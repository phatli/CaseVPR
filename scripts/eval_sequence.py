#!/usr/bin/env python
"""Run sequence-level evaluations for CaseVPR encoders.

The script accepts either a single-evaluation JSON config or a batch config
containing ``model_variants``/``dataset_variants`` (see
``scripts/configs/seq_eval``). Configuration keys map directly to
:class:`casevpr.training.EvaluationConfig` and :class:`casevpr.training.EvalExperiment`.
Batch helpers such as ``base_config`` and meta keys like ``_folder`` /
``_variant_suffix`` are also supported; document any new helpers here when
adding features.

Config reference:
    experiment:
        name – tag used for output folders/logging.
        logs_root – base directory where timestamped runs land.
        seed – random seed forwarded to torch/numpy/cuda.
        deterministic – flip to true for deterministic cuDNN / PyTorch behaviour.
        device – inference device (e.g. "cuda", "cpu").
        save_preds_json – dump per-query predictions to predictions.json.
        save_gif – emit GIF visualisations of retrieval results.
        save_html – emit interactive HTML maps.
        infer_batch_size – mini-batch size for descriptor extraction.
    model:
        seq_encoder – key in hvpr_encoders or seq_encoders (see
            scripts/configs/model_configs.py) such as hvpr_casevpr_224,
            hvpr_casevpr_224_crica, hvpr_casevpr_322, hvpr_seqnet,
            vgg16_seqvlad, jist, svpr.
        checkpoint – optional path to a custom checkpoint. If you leave it out
            we grab the encoder's default ckpt_path from model_configs.py.
        use_best – prefer best_model.pth over last_model.pth when present.
        pca_outdim – optional PCA dimension (None keeps the raw features).
    data:
        dataset_path – formatted dataset root containing split/{queries,database}.
        split – dataset split evaluated (e.g. val, test, custom tag).
        seq_length – frames per sequence descriptor.
        cut_last_frame – drop the final frame from sequences when true.
        val_posDistThr – positive radius (metres) used for recall.
        img_shape – explicit [H, W] resize target; defaults to encoder preference.
        seq_gt_strategy – ground-truth mode {lax, strict, lastframe}.
        reverse – evaluate sequences in reverse order when true.
Batch configs:
    model_variants/models – dict of per-model overrides (supports meta keys
        ``_folder`` / ``_variant_suffix`` for directory naming).
    dataset_variants/datasets – dict of per-dataset overrides (meta keys
        support ``_folder`` or ``_variant``).
    base_config/base_config_path – optional template to merge before applying
        per-variant overrides.
"""
import argparse
import copy
import json
import os
import re
import shutil
import sys
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CASEVPR_ROOT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-2])
if CASEVPR_ROOT_DIR not in sys.path:
    sys.path.insert(0, CASEVPR_ROOT_DIR)

from casevpr.training.eval_runner import EvalExperiment, EvaluationConfig, SequenceEvaluator
from casevpr.training.logging import stop_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECTION_NAMES = {"experiment", "model", "data"}
KEY_TO_SECTION_MAP: Dict[str, Tuple[str, str]] = {
    "name": ("experiment", "name"),
    "logs_root": ("experiment", "logs_root"),
    "seed": ("experiment", "seed"),
    "deterministic": ("experiment", "deterministic"),
    "device": ("experiment", "device"),
    "save_preds_json": ("experiment", "save_preds_json"),
    "save_gif": ("experiment", "save_gif"),
    "save_html": ("experiment", "save_html"),
    "infer_batch_size": ("experiment", "infer_batch_size"),
    "seq_encoder": ("model", "seq_encoder"),
    "checkpoint": ("model", "checkpoint"),
    "resume": ("model", "checkpoint"),
    "use_best": ("model", "use_best"),
    "pca_outdim": ("model", "pca_outdim"),
    "dataset_path": ("data", "dataset_path"),
    "split": ("data", "split"),
    "seq_length": ("data", "seq_length"),
    "cut_last_frame": ("data", "cut_last_frame"),
    "val_posDistThr": ("data", "val_posDistThr"),
    "img_shape": ("data", "img_shape"),
    "seq_gt_strategy": ("data", "seq_gt_strategy"),
    "reverse": ("data", "reverse"),
}
KNOWN_SEQ_ENCODERS = {
    "vgg16_seqvlad",
    "jist",
    "svpr",
    "hvpr_seqnet",
    "hvpr_casevpr_224",
    "hvpr_casevpr_224_crica",
    "hvpr_casevpr_322",
}
LEGACY_ENCODER_ALIASES = {
    "hvpr_seqnet": "hvpr_seqnet",
    "svpr": "svpr",
    "jist": "jist",
    "jist_r18": "jist",
}
PATH_FIELDS = {
    ("experiment", "logs_root"),
    ("model", "checkpoint"),
    ("data", "dataset_path"),
}
WARNED_KEYS: set = set()


def is_batch_config(raw_config: Dict[str, Any]) -> bool:
    if not isinstance(raw_config, dict):
        return False
    if "ds_dicts" in raw_config and "args_dicts" in raw_config:
        return True
    if "datasets" in raw_config and "models" in raw_config:
        return True
    if "dataset_variants" in raw_config and "model_variants" in raw_config:
        return True
    return False


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("_")
    return slug or "run"


def _split_meta(overrides: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta = {k: v for k, v in overrides.items() if isinstance(k, str) and k.startswith("_")}
    clean = {k: v for k, v in overrides.items() if not (isinstance(k, str) and k.startswith("_"))}
    return meta, clean


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _maybe_resolve_path(section: str, field: str, value: Any, config_dir: Path) -> Any:
    if (section, field) in PATH_FIELDS and isinstance(value, str):
        expanded = os.path.expanduser(value)
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = (config_dir / expanded).resolve()
            if not candidate.exists():
                candidate = (PROJECT_ROOT / expanded).resolve()
        return str(candidate)
    return value


def _normalise_existing_paths(config_dict: Dict[str, Any], config_dir: Path) -> None:
    for section, field in PATH_FIELDS:
        section_data = config_dict.get(section)
        if isinstance(section_data, dict) and isinstance(section_data.get(field), str):
            section_data[field] = _maybe_resolve_path(section, field, section_data[field], config_dir)


def _guess_seq_encoder(arch_value: str, overrides: Dict[str, Any]) -> Optional[str]:
    alias = LEGACY_ENCODER_ALIASES.get(arch_value)
    if alias:
        return alias
    if arch_value in KNOWN_SEQ_ENCODERS:
        return arch_value
    pooling = overrides.get("pooling") or overrides.get("aggregation")
    if arch_value == "vgg16" and pooling == "seqvlad":
        return "vgg16_seqvlad"
    return None


def _infer_section_for_key(key: str, value: Any, overrides: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, str]]:
    if key in KEY_TO_SECTION_MAP:
        return KEY_TO_SECTION_MAP[key]
    return None


def _apply_overrides(config_dict: Dict[str, Any], overrides: Dict[str, Any], source: str, config_dir: Path, current_section: Optional[str] = None) -> None:
    for key, value in overrides.items():
        if key == "arch":
            if isinstance(value, str) and "seq_encoder" not in overrides:
                guessed = _guess_seq_encoder(value, overrides)
                if guessed:
                    if current_section == "model":
                        model_section = config_dict
                    else:
                        model_section = config_dict.setdefault("model", {})
                    model_section["seq_encoder"] = guessed
                elif key not in WARNED_KEYS:
                    print(f"Unable to map legacy key '{source}.{key}' to a seq_encoder. Provide 'seq_encoder' explicitly.", file=sys.stderr)
                    WARNED_KEYS.add(key)
            continue
        if key in SECTION_NAMES and isinstance(value, dict):
            target = config_dict.setdefault(key, {})
            _apply_overrides(target, value, f"{source}.{key}", config_dir, current_section=key)
            continue
        section_info = _infer_section_for_key(key, value, overrides)
        if section_info:
            section, field = section_info
            if section == current_section:
                config_dict[field] = _maybe_resolve_path(section, field, value, config_dir)
            else:
                section_dict = config_dict.setdefault(section, {})
                section_dict[field] = _maybe_resolve_path(section, field, value, config_dir)
        else:
            if isinstance(value, dict):
                config_dict[key] = copy.deepcopy(value)
            else:
                if key not in WARNED_KEYS:
                    print(f"Ignoring unrecognised key '{key}' from {source}", file=sys.stderr)
                    WARNED_KEYS.add(key)


def _load_base_config(batch_config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    if "base_config" in batch_config:
        base_cfg = batch_config["base_config"]
        if not isinstance(base_cfg, dict):
            raise TypeError("Expected 'base_config' to be a mapping.")
        return copy.deepcopy(base_cfg)
    base_path_value = batch_config.get("base_config_path")
    if base_path_value:
        base_path = Path(os.path.expanduser(base_path_value))
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"Base config file not found: {base_path}")
        return _load_json(base_path)
    base_cfg = {
        "experiment": copy.deepcopy(batch_config.get("experiment", {})),
        "model": copy.deepcopy(batch_config.get("model", {})),
        "data": copy.deepcopy(batch_config.get("data", {})),
    }
    return base_cfg


def run_single_evaluation(
    raw_config: Dict[str, Any],
    config_path: Optional[Path] = None,
    dry_run: bool = False,
    override_output_folder: Optional[Path] = None,
    copy_config: bool = True,
) -> Dict[str, Any]:
    config_dir = config_path.parent if config_path else Path.cwd()
    _normalise_existing_paths(raw_config, config_dir)
    eval_config = EvaluationConfig.from_dict(raw_config)
    if dry_run:
        return {}
    evaluator = SequenceEvaluator(
        eval_config,
        config_path=config_path if copy_config else None,
        override_output_folder=override_output_folder,
    )
    try:
        return evaluator.evaluate()
    finally:
        stop_logging()


def run_batch_evaluations(
    batch_config: Dict[str, Any],
    config_path: Path,
    dry_run: bool = False,
    override_save_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    model_block_key = "model_variants" if isinstance(batch_config.get("model_variants"), dict) else (
        "models" if isinstance(batch_config.get("models"), dict) else "args_dicts"
    )
    dataset_block_key = "dataset_variants" if isinstance(batch_config.get("dataset_variants"), dict) else (
        "datasets" if isinstance(batch_config.get("datasets"), dict) else "ds_dicts"
    )
    models_block = batch_config.get(model_block_key, {})
    datasets_block = batch_config.get(dataset_block_key, {})
    if not isinstance(models_block, dict) or not isinstance(datasets_block, dict):
        raise ValueError("Batch config must define model and dataset dictionaries (model_variants/models/args_dicts, dataset_variants/datasets/ds_dicts).")

    config_dir = config_path.parent
    base_config = _load_base_config(batch_config, config_path)
    _normalise_existing_paths(base_config, config_dir)

    base_exp_cfg = base_config.get("experiment", {})
    default_logs_root = EvalExperiment().logs_root
    base_logs_root = Path(base_exp_cfg.get("logs_root", default_logs_root))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = _slugify(str(base_exp_cfg.get("name", "seq_eval")))
    overall_folder = base_logs_root / f"{timestamp}-{exp_name}-batch"

    combinations = [(model_key, models_block[model_key], dataset_key, datasets_block[dataset_key]) for model_key in models_block for dataset_key in datasets_block]

    if dry_run:
        print(f"Batch evaluation will produce folder '{overall_folder}' with {len(combinations)} combinations:")
        for model_key, model_overrides_raw, dataset_key, dataset_overrides_raw in combinations:
            model_meta, _ = _split_meta(model_overrides_raw)
            _, _ = _split_meta(dataset_overrides_raw)
            variant_suffix = model_meta.get("_variant_suffix") or model_meta.get("_variant")
            model_label = model_meta.get("_folder") or model_key
            model_dir_name = _slugify(model_label)
            if variant_suffix:
                model_dir_name = f"{model_dir_name}_{_slugify(str(variant_suffix))}"
            dataset_dir_name = _slugify(dataset_key)
            candidate = copy.deepcopy(base_config)
            _, model_overrides = _split_meta(model_overrides_raw)
            _, dataset_overrides = _split_meta(dataset_overrides_raw)
            _apply_overrides(candidate, model_overrides, f"{model_block_key}.{model_key}", config_dir)
            _apply_overrides(candidate, dataset_overrides, f"{dataset_block_key}.{dataset_key}", config_dir)
            EvaluationConfig.from_dict(candidate)
            print(f"  - model '{model_key}' -> folder '{model_dir_name}', dataset '{dataset_key}' -> subdir '{dataset_dir_name}'")
        print("Batch configuration parsed successfully. Exiting due to --dry-run.")
        return {}

    overall_folder.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copy(config_path, overall_folder / config_path.name)

    overall_results: Dict[str, Dict[str, Any]] = {}

    for model_key, model_overrides_raw in models_block.items():
        model_meta, model_overrides = _split_meta(model_overrides_raw)
        variant_suffix = model_meta.get("_variant_suffix") or model_meta.get("_variant")
        model_label = model_meta.get("_folder") or model_key
        model_dir_name = _slugify(model_label)
        if variant_suffix:
            model_dir_name = f"{model_dir_name}_{_slugify(str(variant_suffix))}"
        model_folder = overall_folder / model_dir_name
        model_folder.mkdir(parents=True, exist_ok=True)

        model_results: Dict[str, Any] = {}

        for idx, (dataset_key, dataset_overrides_raw) in enumerate(datasets_block.items()):
            _, dataset_overrides = _split_meta(dataset_overrides_raw)
            dataset_dir_name = _slugify(dataset_key)
            dataset_folder = model_folder / dataset_dir_name
            dataset_folder.mkdir(parents=True, exist_ok=True)

            combined_config = copy.deepcopy(base_config)
            _apply_overrides(combined_config, model_overrides, f"model_variants.{model_key}", config_dir)
            _apply_overrides(combined_config, dataset_overrides, f"dataset_variants.{dataset_key}", config_dir)
            combined_config.setdefault("experiment", {})
            combined_config["experiment"]["logs_root"] = str(model_folder)
            combined_config["experiment"]["name"] = model_dir_name

            try:
                result = run_single_evaluation(
                    combined_config,
                    config_path=config_path,
                    dry_run=False,
                    override_output_folder=model_folder,
                    copy_config=(idx == 0),
                )
            except Exception as exc:
                error_trace = "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))
                error_message = f"{exc.__class__.__name__}: {exc}"
                error_path = dataset_folder / "error.log"
                error_path.write_text(error_trace)
                model_results[dataset_key] = {
                    "error": error_message,
                    "dataset_subdir": dataset_dir_name,
                }
                print(f"Failed model '{model_key}' on dataset '{dataset_key}': {error_message}")
                continue

            _relocate_artifacts(model_folder, dataset_folder, dataset_dir_name)
            model_results[dataset_key] = {
                "recalls": result.get("recalls"),
                "summary": result.get("summary"),
                "dataset_subdir": dataset_dir_name,
            }
            summary = result.get("summary")
            summary_str = summary if isinstance(summary, str) else json.dumps(summary)
            print(f"Completed model '{model_key}' on dataset '{dataset_key}': {summary_str}")

        results_path = model_folder / "results.json"
        with results_path.open("w") as f:
            json.dump(model_results, f, indent=2)

        overall_results[model_dir_name] = model_results

    if override_save_path:
        overall_results_path = override_save_path if override_save_path.is_absolute() else (overall_folder / override_save_path)
    else:
        overall_results_path = overall_folder / "overall_results.json"
    overall_results_path.parent.mkdir(parents=True, exist_ok=True)
    with overall_results_path.open("w") as f:
        json.dump(overall_results, f, indent=2)

    return overall_results


def _relocate_artifacts(model_folder: Path, dataset_folder: Path, dataset_slug: str) -> None:
    dataset_folder.mkdir(parents=True, exist_ok=True)

    artifacts = []
    predictions_src = model_folder / "predictions.json"
    if predictions_src.exists():
        artifacts.append((predictions_src, dataset_folder / f"{dataset_slug}_predictions.json"))

    gif_src = predictions_src.with_suffix(".gif")
    if gif_src.exists():
        artifacts.append((gif_src, dataset_folder / f"{dataset_slug}.gif"))

    html_src = model_folder / f"{model_folder.name}.html"
    if not html_src.exists():
        html_candidates = sorted(model_folder.glob("*.html"))
        if html_candidates:
            html_src = html_candidates[0]
    if html_src.exists():
        artifacts.append((html_src, dataset_folder / f"{dataset_slug}.html"))

    for src, dst in artifacts:
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CaseVPR sequence encoder precision")
    parser.add_argument("--config", required=True, help="Path to evaluation config JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_config = _load_json(config_path)
    if is_batch_config(raw_config):
        results = run_batch_evaluations(raw_config, config_path, dry_run=args.dry_run)
        if args.dry_run:
            return
        print("Batch evaluation summary:")
        print(json.dumps(results, indent=2))
        return

    if args.dry_run:
        run_single_evaluation(raw_config, config_path=config_path, dry_run=True)
        print("Evaluation configuration parsed successfully. Exiting due to --dry-run.")
        return

    results = run_single_evaluation(raw_config, config_path=config_path)
    print("Evaluation summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
