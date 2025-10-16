"""Evaluation utilities for sequence descriptor training."""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import faiss
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from .utils import extract_sequence_descriptor


def test(
    args,
    eval_ds,
    model,
    epoch: int = 0,
    pca: Optional[PCA] = None,
    writer=None,
    tboard_name: str = "test",
    seq_gt_strategy: Optional[str] = None,
    return_preds: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[np.ndarray, str]:
    if not seq_gt_strategy:
        seq_gt_strategy = args.seq_gt_strategy
    model = model.eval()
    outputdim = model.module.meta['outputdim']
    seq_len = args.seq_length
    n_gpus = args.n_gpus

    query_num = eval_ds.queries_num
    gallery_num = eval_ds.database_num
    all_features = np.empty((query_num + gallery_num, outputdim), dtype=np.float32)

    with torch.no_grad():
        logging.debug("Extracting gallery features for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=4,
            batch_size=args.infer_batch_size,
            pin_memory=(args.device == "cuda"),
            shuffle=args.test_shuffle,
            generator=generator,
        )
        inference_times = []

        for images, indices, _ in tqdm(database_dataloader, ncols=100, desc="Extracting gallery features..."):
            torch.cuda.synchronize()
            start_time = time.time()
            images = images.contiguous().view(-1, 3, args.img_shape[0], args.img_shape[1])
            if (images.shape[0] % (seq_len * n_gpus) != 0) and n_gpus > 1:
                model.module = model.module.to(args.device)
                for sequence in range(images.shape[0] // seq_len):
                    n_seq = sequence * seq_len
                    seq_images = images[n_seq: n_seq + seq_len].to(args.device)
                    outputs = model.module(seq_images)
                    features = extract_sequence_descriptor(outputs).cpu().numpy()
                    if pca:
                        features = pca.transform(features)
                    all_features[indices.numpy()[sequence], :] = features
                model = model.cuda()
            else:
                outputs = model(images.to(args.device))
                features = extract_sequence_descriptor(outputs).cpu().numpy()
                if pca:
                    features = pca.transform(features)
                all_features[indices.numpy(), :] = features
            torch.cuda.synchronize()
            inference_times.append(time.time() - start_time)

        if inference_times:
            average_time = sum(inference_times) / len(inference_times)
            logging.info("Average inference time: %.4f s", average_time)

        logging.debug("Extracting queries features for evaluation/testing")
        queries_subset_ds = Subset(
            eval_ds,
            list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)),
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds,
            num_workers=4,
            batch_size=args.infer_batch_size,
            pin_memory=(args.device == "cuda"),
            generator=generator,
        )

        for images, _, indices in tqdm(queries_dataloader, ncols=100, desc="Extracting queries features..."):
            images = images.contiguous().view(-1, 3, args.img_shape[0], args.img_shape[1])

            if (images.shape[0] % (seq_len * n_gpus) != 0) and n_gpus > 1:
                model.module = model.module.to(args.device)
                for sequence in range(images.shape[0] // seq_len):
                    n_seq = sequence * seq_len
                    seq_images = images[n_seq: n_seq + seq_len].to(args.device)
                    outputs = model.module(seq_images)
                    features = extract_sequence_descriptor(outputs).cpu().numpy()
                    if pca:
                        features = pca.transform(features)
                    all_features[indices.numpy()[sequence], :] = features
                model = model.cuda()
            else:
                outputs = model(images.to(args.device))
                features = extract_sequence_descriptor(outputs).cpu().numpy()
                if pca:
                    features = pca.transform(features)
                all_features[indices.numpy(), :] = features

    torch.cuda.empty_cache()
    queries_features = all_features[eval_ds.database_num:]
    gallery_features = all_features[:eval_ds.database_num]

    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatL2(res, outputdim)
    faiss_index.add(gallery_features)

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_features, 10)

    positives_per_query = eval_ds.pIdx
    recall_values = [1, 5, 10]
    recalls = np.zeros(len(recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / len(eval_ds.qIdx) * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])

    if writer:
        for val, rec in zip(recall_values, recalls):
            writer.add_scalar(f"{tboard_name}_{seq_gt_strategy}/recall_{val}", float(rec), epoch)
    if return_preds:
        return recalls, recalls_str, predictions
    return recalls, recalls_str
