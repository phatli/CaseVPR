#!/usr/bin/env python

import cv2
import numpy as np
from os.path import join

import matplotlib.pyplot as plt

def generate_colormap(num_colors):
    cmap = plt.get_cmap('Accent')
    return [cmap(i)[:3] for i in np.linspace(0, 1, num_colors)]


def add_border(image_numpy, color):
    dim = 2 if color == 'red' else 1
    width = image_numpy.shape[1]
    height = image_numpy.shape[0]
    pad_width = int(0.025*height)
    mask = np.pad(np.ones((height - pad_width * 2, width - pad_width * 2)),
                  ((pad_width, pad_width), (pad_width, pad_width)), 'constant', constant_values=(0, 0))
    mask = np.stack((mask, mask, mask), axis=-1)
    mask0 = 255 * (1 - mask[:, :, dim])
    masked_img = image_numpy * mask
    masked_img[:, :, dim] += mask0
    return np.uint8(masked_img)


def vis_D(D: np.ndarray, retrieved_idx: list, gt_idx: list, start_blocks: list = None, no_sobel=False) -> np.ndarray:
    h = D.shape[0]
    indicator_colomn = 255 * \
        np.ones((h, len(retrieved_idx[0]), 3)).astype(np.uint8)
    gt_colomn = 255 * \
        np.ones((h, len(retrieved_idx[0]), 3)).astype(np.uint8)

    for j in range(1, len(retrieved_idx)):
        for i, idx in enumerate(retrieved_idx[j]):
            indicator_colomn[idx, i] = np.array(
                [[100, 100, 100]]).astype(np.uint8)

    for i, idx in enumerate(retrieved_idx[0]):
        indicator_colomn[idx, i] = np.array(
            [[0, 0, 0]]).astype(np.uint8)

    for i, indices in enumerate(gt_idx):
        gt_colomn[indices, i] = np.array([[0, 255, 0]]).astype(np.uint8)
    if start_blocks is not None:
        start_blocks_col = 255 * \
            np.ones((h, len(start_blocks), 3)).astype(np.uint8)
        colors = generate_colormap(len(start_blocks))
        for i, (start, end) in enumerate(start_blocks):
            # Convert to 0-255 range
            color = (np.array(colors[i]) * 255).astype(np.uint8)
            start_blocks_col[start:end, i] = color

    max_value = np.max(D)
    min_value = np.min(D)

    normalized_heat_map = ((D - min_value) /
                           (max_value - min_value)) * 255
    normalized_heat_map = normalized_heat_map.astype(np.uint8)

    x = cv2.Sobel(normalized_heat_map, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(normalized_heat_map, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    normalized_heat_map_sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    heatmap = cv2.applyColorMap(normalized_heat_map, cv2.COLORMAP_JET)[:h]
    heatmap_sobel = cv2.applyColorMap(
        normalized_heat_map_sobel, cv2.COLORMAP_JET)
    if not no_sobel:
        heatmap = np.concatenate((gt_colomn, heatmap, 255 * np.ones((h, 1, 3)
                                                                    ).astype(np.uint8), heatmap_sobel, indicator_colomn), axis=1)
        if start_blocks is not None:
            heatmap = np.concatenate((
                heatmap,
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                start_blocks_col
            ), axis=1)
    else:
        if start_blocks is not None:
            heatmap = np.concatenate((
                start_blocks_col,
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                heatmap,
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                indicator_colomn,
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                gt_colomn
            ), axis=1)
        else:
            heatmap = np.concatenate((
                heatmap,
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                gt_colomn,
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                255 * np.ones((h, 1, 3)).astype(np.uint8),
                indicator_colomn
            ), axis=1)

    return heatmap


def vis_D_dual(D, retrieved_idx1, retrieved_idx2, ispos1, ispos2, gt_idx):
    h = D.shape[0]
    indicator_colomn1 = 255 * \
        np.ones((h, len(retrieved_idx1[0]), 3)).astype(np.uint8)
    indicator_colomn2 = 255 * \
        np.ones((h, len(retrieved_idx1[0]), 3)).astype(np.uint8)
    gt_colomn = 255 * \
        np.ones((h, len(retrieved_idx1[0]), 3)).astype(np.uint8)

    for j in range(1, len(retrieved_idx1)):
        for i, idx in enumerate(retrieved_idx1[j]):
            indicator_colomn1[idx, i] = np.array(
                [[100, 100, 100]]).astype(np.uint8)
    for j in range(1, len(retrieved_idx2)):
        for i, idx in enumerate(retrieved_idx2[j]):
            indicator_colomn2[idx, i] = np.array(
                [[100, 100, 100]]).astype(np.uint8)

    for i, idx in enumerate(retrieved_idx1[0]):
        indicator_colomn1[idx, i] = np.array(
            [[0, 255, 0] if ispos1 else [255, 0, 0]]).astype(np.uint8)
    for i, idx in enumerate(retrieved_idx2[0]):
        indicator_colomn2[idx, i] = np.array(
            [[0, 255, 0] if ispos2 else [255, 0, 0]]).astype(np.uint8)

    for i, indices in enumerate(gt_idx):
        gt_colomn[indices, i] = np.array([[0, 255, 0]]).astype(np.uint8)

    max_value = np.max(D)
    min_value = np.min(D)

    normalized_heat_map = ((D - min_value) /
                           (max_value - min_value)) * 255
    normalized_heat_map = normalized_heat_map.astype(np.uint8)

    heatmap = cv2.applyColorMap(normalized_heat_map, cv2.COLORMAP_JET)[:h]
    heatmap = np.concatenate((gt_colomn, heatmap, 255 * np.ones((h, 1, 3)).astype(np.uint8),
                             indicator_colomn1, 255 * np.ones((h, 1, 3)).astype(np.uint8), indicator_colomn2), axis=1)
    return heatmap


def vis_seq(decoded_info, img_idx, backend_name, saveSeq, save_dir=None):
    """To visualize sequence of algorithms to debug.

    Args:
        decoded_info (dict): 
            Should include following keys: 
                path, query_timestamp, DD, db_timestamp, gt_idx, retrieved_seq_indices, retrieved_DD, is_pos, start_blocks. 

        img_idx (int): Index of image. 
        backend_name (str): Backend name.
        saveSeq (bool): If True, save the sequence images.
        save_dir (str): Directory to save the sequence images.

    Returns:
        background, D_heapmap_vis: cv2 images of background and depth heatmap.
    """
    dbpath = decoded_info['dbpath']
    querypath = decoded_info['querypath']
    img = cv2.imread(
        join(querypath, f"{decoded_info['query_timestamp'][0]}.png"), cv2.IMREAD_COLOR)
    image_height = img.shape[0]
    image_width = img.shape[1]
    ds = len(decoded_info['query_timestamp'])

    interval = int(0.1 * image_width)
    if ds > 7:
        background_width = ((interval+image_width) * 2 + interval) * 2
        background_height = int(1.2 * image_height) * int(ds/2) + interval
    else:
        background_width = ((interval+image_width) * 2 + interval)
        background_height = int(1.2 * image_height) * int(ds) + interval

    background = np.ones(
        (background_height, background_width, 3), np.uint8) * 255

    x, y = interval, interval
    DD = np.array(decoded_info["DD"])
    start_blocks = decoded_info.get("start_blocks", None)

    # For vis_seqslam
    if decoded_info["db_timestamp"] is not None:
        if "+" in backend_name:
            gt_idx = decoded_info["gt_idx"]
            retrieved_idx1, retrieved_idx2 = decoded_info["retrieved_seq_indices"]
            retreived_DD1, retrieved_DD2 = decoded_info['retrieved_DD']
            is_pos1 = eval(decoded_info["is_pos"])
            is_pos2 = retrieved_idx2[0][-1] in gt_idx[-1]
            D_heatmap = vis_D_dual(
                DD, retrieved_idx1, retrieved_idx2, is_pos1, is_pos2, gt_idx)
            D_heatmap_vis = D_heatmap[:50]

            for i in range(ds):
                img_query = cv2.imread(
                    join(querypath, f"{decoded_info['query_timestamp'][-(i+1)]}.png"), cv2.IMREAD_COLOR)
                img_db = cv2.imread(
                    join(dbpath, f"{decoded_info['db_timestamp'][-(i+1)]}.png"), cv2.IMREAD_COLOR)
                img_c = np.concatenate((img_query, img_db), 1)
                img = np.ones(
                    (int(1.2 * img_c.shape[0]), img_c.shape[1], 3), np.uint8) * 255
                img[:img_c.shape[0], :img_c.shape[1]] = img_c
                cv2.putText(img, f"idx: {retrieved_idx1[0][-(i+1)]},difference:{retreived_DD1[-(i+1)]:.3f}",
                            (5, img_c.shape[0]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                background[y:y+img.shape[0], x:x+img.shape[1]] = img
                if ds > 7:
                    if i != int(ds/2) - 1:
                        y += img.shape[0]
                    else:
                        x = 5 * interval + 2 * image_width
                        y = interval
                else:
                    y += img.shape[0]
            if saveSeq:
                cv2.imwrite(
                    save_dir + f"/{img_idx:04d}_seq.png", background)
                cv2.imwrite(
                    save_dir + f"/{img_idx:04d}_{'TP' if is_pos1 else 'FP'}_{'TP' if is_pos2 else 'FP'}_D.png", cv2.cvtColor(D_heatmap, cv2.COLOR_RGB2BGR))

        else:
            is_pos = eval(decoded_info["is_pos"])
            retrieved_idx = decoded_info["retrieved_seq_indices"]
            gt_idx = decoded_info["gt_idx"]
            D_heatmap = vis_D(DD, retrieved_idx, gt_idx,
                              start_blocks=start_blocks)
            D_heatmap_vis = D_heatmap[:50]
            for i in range(ds):
                img_query = cv2.imread(join(
                    querypath, f"{decoded_info['query_timestamp'][-(i+1)]}.png"), cv2.IMREAD_COLOR)
                img_db = cv2.imread(
                    join(dbpath, f"{decoded_info['db_timestamp'][-(i+1)]}.png"), cv2.IMREAD_COLOR)
                # print(decoded_info['query_timestamp'][-(i+1)],decoded_info['db_timestamp'][-(i+1)],i)
                img_c = np.concatenate((img_query, img_db), 1)
                # img_c = add_border(img_c,'green' if decoded_info['retrieved_DD'][-(i+1)] * 10000 < 0.2 else 'red')
                img = np.ones(
                    (int(1.2 * img_c.shape[0]), img_c.shape[1], 3), np.uint8) * 255
                img[:img_c.shape[0], :img_c.shape[1]] = img_c
                cv2.putText(img, f"idx: {decoded_info['retrieved_seq_indices'][0][-(i+1)]},difference:{decoded_info['retrieved_DD'][-(i+1)]:.3f}",
                            (5, img_c.shape[0]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                background[y:y+img.shape[0], x:x+img.shape[1]] = img
                if ds > 7:
                    if i != int(ds/2) - 1:
                        y += img.shape[0]
                    else:
                        x = 5 * interval + 2 * image_width
                        y = interval
                else:
                    y += img.shape[0]
            if saveSeq:
                cv2.imwrite(
                    save_dir + f"/{img_idx:04d}_seq.png", background)
                cv2.imwrite(
                    save_dir + f"/{img_idx:04d}_{'TP' if is_pos else 'FP'}_D.png", cv2.cvtColor(D_heatmap, cv2.COLOR_RGB2BGR))
    else:
        background = cv2.putText(background, "No Loop Detected!", (20, int(
            2/3 * background.shape[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
        D_heatmap_vis = None

    return background, D_heatmap_vis
