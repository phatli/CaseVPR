"""Result visualisation helpers for sequence evaluation."""
from __future__ import annotations

import base64
import copy
import io
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor


ICON_RED_CROSS = base64.b64encode(
    b"""
<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 384 512">
    <style>svg{fill:#ff0000}</style>
    <path d="M342.6 150.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L192 210.7 86.6 105.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3L146.7 256 41.4 361.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L192 301.3 297.4 406.6c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L237.3 256 342.6 150.6z"/>
</svg>
""".strip()
).decode("utf-8")

CIRCLE_GREEN = base64.b64encode(
    b"""
<svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 512 512">
    <style>
        svg {
            fill: #3fe61e;
            stroke: #3fe61e;
            stroke-width: 64;
        }
    </style>
    <path d="M464 256A208 208 0 1 0 48 256a208 208 0 1 0 416 0zM0 256a256 256 0 1 1 512 0A256 256 0 1 1 0 256z"/>
</svg>
""".strip()
).decode("utf-8")

RED_ICON_DATA_URI = f"data:image/svg+xml;base64,{ICON_RED_CROSS}"
GREEN_ICON_DATA_URI = f"data:image/svg+xml;base64,{CIRCLE_GREEN}"


def generate_gif(
    predictions: Sequence[Sequence],
    dataset_folder: str,
    json_path: str,
    target_height: int = 600,
) -> None:
    """Create an animated GIF that overlays matches on a basemap."""
    predictions = list(predictions)
    if not predictions:
        logging.info("No predictions supplied for GIF generation.")
        return

    try:
        import numpy as np
        from PIL import Image
        import utm
        import matplotlib.pyplot as plt
        import geopandas as gpd
        import contextily as ctx
    except ImportError as exc:
        logging.warning(
            "Skipping GIF generation for %s because '%s' is not installed.",
            json_path,
            exc.name if hasattr(exc, "name") else exc,
        )
        return

    dataset_root = Path(dataset_folder)
    try:
        video_frames = _build_gif_frames(
            predictions,
            dataset_root,
            target_height,
            np,
            Image,
            utm,
            plt,
            gpd,
            ctx,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Failed to build GIF for %s: %s", json_path, exc)
        return

    if not video_frames:
        logging.warning("No frames produced for GIF %s.", json_path)
        return

    gif_path = Path(json_path).with_suffix(".gif")
    try:
        video_frames[0].save(
            gif_path,
            format="GIF",
            append_images=video_frames[1:],
            save_all=True,
            duration=200,
            loop=0,
        )
        logging.info("Saved GIF visualisation to %s", gif_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Failed to save GIF %s: %s", gif_path, exc)


def get_html_result_map(
    predictions: Sequence[Sequence],
    dataset_folder: str,
    html_path: str,
    max_entries: int = 250,
) -> None:
    """Create an interactive HTML map of retrieval matches."""
    try:
        import folium
    except ImportError:
        logging.warning(
            "Skipping HTML map generation for %s because 'folium' is not installed.",
            html_path,
        )
        return

    records = list(predictions)[:max_entries]
    if not records:
        logging.info("No predictions supplied for HTML map generation.")
        return

    dataset_root = Path(dataset_folder)

    tasks = [(record, dataset_root) for record in records]
    marker_data: List[Tuple[float, float, bool, str]] = []

    with ThreadPoolExecutor() as executor:
        for result in executor.map(_process_html_item_safe, tasks):
            if result is not None:
                marker_data.append(result)

    if not marker_data:
        logging.warning("No valid entries to plot for %s.", html_path)
        return

    first_lat, first_lon, *_ = marker_data[0]
    fmap = folium.Map(location=[first_lat, first_lon], zoom_start=16)

    for lat, lon, is_correct, popup_html in marker_data:
        icon_uri = GREEN_ICON_DATA_URI if is_correct else RED_ICON_DATA_URI
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=600),
            icon=folium.CustomIcon(icon_image=icon_uri, icon_size=(18, 18)),
        ).add_to(fmap)

    try:
        fmap.save(html_path)
        logging.info("Saved HTML map to %s", html_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Failed to save HTML map %s: %s", html_path, exc)


def save_predictions_json(predictions: Iterable[Sequence], json_path: str) -> None:
    with open(json_path, 'w') as f:
        json.dump(list(predictions), f, indent=2)
    logging.info("Saved predictions to %s", json_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_rel_paths(rel: str) -> List[str]:
    return [segment.strip() for segment in rel.split(',') if segment.strip()]


def _resolve_image_paths(dataset_root: Path, rel_paths: List[str]) -> List[Path]:
    return [dataset_root / rel_path for rel_path in rel_paths]


def _find_gps_coordinates(filename: str, utm_zone: int, hemisphere: str = 'N') -> Tuple[float, float]:
    import utm  # type: ignore

    parts = ".".join(filename.split('.')[:-1]).split('@')
    if utm_zone == 30:
        easting = float(parts[2])
        northing = float(parts[1])
    else:
        easting = float(parts[1])
        northing = float(parts[2])
    lat, lon = utm.to_latlon(easting, northing, utm_zone, hemisphere)
    return lat, lon


def _guess_utm_zone(dataset_root: Path) -> int:
    text = str(dataset_root)
    return 30 if "robotcar" in text else 48


def _load_image_to_buffer(image_paths: List[Path], base_width: int = 500):
    from PIL import Image  # type: ignore

    resample = getattr(Image, "LANCZOS", Image.BICUBIC)

    def resize_image(path: Path, width: int) -> Image.Image:
        with Image.open(path) as img:
            img = img.convert("RGB")
            original_width, original_height = img.size
            aspect_ratio = original_height / original_width
            new_height = int(width * aspect_ratio)
            return img.resize((width, new_height), resample)

    images = [resize_image(path, base_width) for path in image_paths if path.exists()]
    if not images:
        raise FileNotFoundError("None of the expected image paths exist.")

    total_width = sum(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    offset = 0
    for img in images:
        combined.paste(img, (offset, 0))
        offset += img.size[0]

    buf = io.BytesIO()
    combined.save(buf, format='PNG')
    buf.seek(0)
    return buf


def _resize_to_height(image, target_height: int):
    from PIL import Image  # type: ignore

    width, height = image.size
    new_width = int(width * (target_height / height))
    resample = getattr(Image, "LANCZOS", Image.BICUBIC)
    return image.resize((new_width, target_height), resample)


def _resize_to_width(image, target_width: int = 484):
    from PIL import Image  # type: ignore

    width, height = image.size
    new_height = int(height * (target_width / width))
    resample = getattr(Image, "LANCZOS", Image.BICUBIC)
    return image.resize((target_width, new_height), resample)


def _calculate_square_bounding_box(latitudes: List[float], longitudes: List[float], margin_percent: float = 1.0):
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)

    centre_lat = (min_lat + max_lat) / 2
    centre_lon = (min_lon + max_lon) / 2

    lat_range = (max_lat - min_lat) * (1 + margin_percent / 100)
    lon_range = (max_lon - min_lon) * (1 + margin_percent / 100)
    square_size = max(lat_range, lon_range)

    lat_bounds = (centre_lat - square_size / 2, centre_lat + square_size / 2)
    lon_bounds = (centre_lon - square_size / 2, centre_lon + square_size / 2)
    return lat_bounds, lon_bounds


def _plot_on_map(query_lons, query_lats, correctness_flags, boundary_lat, boundary_lon, plt, gpd, ctx):
    fig, ax = plt.subplots(figsize=(8, 8))
    for lon, lat, is_correct in zip(query_lons, query_lats, correctness_flags):
        colour = 'green' if is_correct else 'red'
        marker = 'o' if is_correct else 'x'
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)
        gdf.plot(ax=ax, color=colour, marker=marker)

    for lon in boundary_lon:
        for lat in boundary_lat:
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326")
            gdf = gdf.to_crs(epsg=3857)
            gdf.plot(ax=ax, alpha=0)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")
    ax.set_axis_off()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    buf.seek(0)

    from PIL import Image  # type: ignore

    map_image = Image.open(buf).convert("RGB")
    buf.close()
    plt.close(fig)
    return _resize_to_width(map_image, 484)


def _process_screenshot(
    screenshot,
    boundary_lat,
    boundary_lon,
    target_height,
    np,
    Image,
    plt,
    gpd,
    ctx,
):
    query_lons, query_lats, correctness_flags, retrieval_img = screenshot
    map_img = _plot_on_map(query_lons, query_lats, correctness_flags, boundary_lat, boundary_lon, plt, gpd, ctx)
    map_img_resized = _resize_to_height(map_img, target_height)
    combined = np.hstack([np.array(map_img_resized), np.array(retrieval_img)])
    return Image.fromarray(combined)


def _create_video_frames(
    screenshots,
    boundary_lat,
    boundary_lon,
    target_height,
    np,
    Image,
    plt,
    gpd,
    ctx,
):
    frames = []
    for screenshot in screenshots:
        frames.append(
            _process_screenshot(
                screenshot,
                boundary_lat,
                boundary_lon,
                target_height,
                np,
                Image,
                plt,
                gpd,
                ctx,
            )
        )
    return frames


def _build_gif_frames(
    predictions,
    dataset_root: Path,
    target_height: int,
    np,
    Image,
    utm,
    plt,
    gpd,
    ctx,
):
    query_lons: List[float] = []
    query_lats: List[float] = []
    correctness_flags: List[bool] = []
    screenshots = []
    utm_zone = _guess_utm_zone(dataset_root)

    for query_rel, retrieved_rel, is_correct in predictions:
        query_paths = _split_rel_paths(query_rel)
        retrieved_paths = _split_rel_paths(retrieved_rel)
        if not query_paths or not retrieved_paths:
            logging.warning("Skipping prediction with missing paths: %s", (query_rel, retrieved_rel))
            continue

        query_images = _resolve_image_paths(dataset_root, query_paths)
        retrieved_images = _resolve_image_paths(dataset_root, retrieved_paths)

        try:
            query_buf = _load_image_to_buffer(query_images)
            retrieved_buf = _load_image_to_buffer(retrieved_images)
        except FileNotFoundError as exc:
            logging.warning("Skipping prediction due to missing image: %s", exc)
            continue

        query_img = Image.open(query_buf).convert("RGB")
        retrieved_img = Image.open(retrieved_buf).convert("RGB")

        query_lat, query_lon = _find_gps_coordinates(query_paths[-1], utm_zone)
        query_lons.append(query_lon)
        query_lats.append(query_lat)
        correctness_flags.append(bool(is_correct))

        border_colour = "green" if is_correct else "red"
        border = Image.new("RGB", (retrieved_img.width + 10, retrieved_img.height + 10), border_colour)
        border.paste(retrieved_img, (5, 5))

        retrieval_stack = np.vstack(
            [
                np.array(_resize_to_width(query_img)),
                np.array(_resize_to_width(border)),
            ]
        )
        retrieval_img_resized = _resize_to_height(Image.fromarray(retrieval_stack), target_height)
        screenshots.append(
            [
                copy.deepcopy(query_lons),
                copy.deepcopy(query_lats),
                copy.deepcopy(correctness_flags),
                retrieval_img_resized,
            ]
        )

    if not screenshots:
        return []

    boundary_lat, boundary_lon = _calculate_square_bounding_box(query_lats, query_lons)
    return _create_video_frames(
        screenshots,
        boundary_lat,
        boundary_lon,
        target_height,
        np,
        Image,
        plt,
        gpd,
        ctx,
    )


def _embed_image_buffer(img_buf: io.BytesIO, retrieval_status: str | None = None) -> str:
    img_buf.seek(0)
    encoded = base64.b64encode(img_buf.read()).decode()
    border = ""
    if retrieval_status is not None:
        colour = "green" if retrieval_status == "Correct" else "red"
        border = f"border:2px solid {colour};"
    return (
        "<div style=\"cursor:pointer;transition:transform 0.25s ease;"
        f"{border}\" onclick=\"this.style.transform=this.style.transform==='scale(2)'?'scale(1)':'scale(2)'\">"
        f"<img src=\"data:image/png;base64,{encoded}\" style=\"height:100px;max-width:2000px;\"/></div>"
    )


def _process_html_item(task) -> Tuple[float, float, bool, str]:
    (record, dataset_root) = task
    query_rel, retrieved_rel, is_correct = record
    query_paths = _split_rel_paths(query_rel)
    retrieved_paths = _split_rel_paths(retrieved_rel)
    if not query_paths or not retrieved_paths:
        raise ValueError("Missing query or retrieved paths.")

    utm_zone = _guess_utm_zone(dataset_root)
    query_lat, query_lon = _find_gps_coordinates(query_paths[-1], utm_zone)

    query_files = _resolve_image_paths(dataset_root, query_paths)
    retrieved_files = _resolve_image_paths(dataset_root, retrieved_paths)

    query_buf = _load_image_to_buffer(query_files)
    retrieved_buf = _load_image_to_buffer(retrieved_files)

    retrieval_status = "Correct" if is_correct else "Incorrect"
    query_html = _embed_image_buffer(query_buf)
    retrieved_html = _embed_image_buffer(retrieved_buf, retrieval_status)

    popup_html = f"Query:<br>{query_html}<br>Retrieved:<br>{retrieved_html}<br>"
    return query_lat, query_lon, bool(is_correct), popup_html


def _process_html_item_safe(task):
    try:
        return _process_html_item(task)
    except Exception as exc:
        logging.warning("Skipping HTML entry due to error: %s", exc)
        return None
