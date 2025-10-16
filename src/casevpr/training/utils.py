"""Utility helpers for CaseVPR training."""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


__all__ = [
    "RAMEfficient2DMatrix",
    "configure_transform",
    "align_pooling_state_dict",
    "freeze_all_except_attn",
    "load_pretrained_backbone",
    "resume_train",
    "save_checkpoint",
    "extract_sequence_descriptor",
]


class RAMEfficient2DMatrix:
    """Sparse row-major storage for large feature caches."""

    def __init__(self, shape: Sequence[int], dtype=np.float32) -> None:
        rows, cols = shape
        self.shape = (int(rows), int(cols))
        self.dtype = dtype
        self._matrix = [None] * self.shape[0]

    def __setitem__(self, indexes: Iterable[int], values) -> None:  # type: ignore[override]
        values = np.asarray(values, dtype=self.dtype)
        if values.shape[1] != self.shape[1]:
            raise ValueError(f"Expected {self.shape[1]} columns, got {values.shape[1]}")
        for idx, val in zip(indexes, values):
            self._matrix[int(idx)] = np.asarray(val, dtype=self.dtype)

    def __getitem__(self, index):  # type: ignore[override]
        if hasattr(index, "__len__"):
            return np.array([self._matrix[int(i)] for i in index], dtype=self.dtype)
        return np.array(self._matrix[int(index)], dtype=self.dtype)


def configure_transform(image_dim: tuple[int, int], meta: Dict[str, Any]) -> transforms.Compose:
    normalize = transforms.Normalize(mean=meta["mean"], std=meta["std"])
    return transforms.Compose([
        transforms.Resize(image_dim, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        normalize,
    ])


def align_pooling_state_dict(model_or_keys: Any, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Align pooling prefix between checkpoints saved with `pool.` vs `pooling.`."""
    if hasattr(model_or_keys, "state_dict"):
        model_keys = list(model_or_keys.state_dict().keys())
    else:
        model_keys = list(model_or_keys)

    expects_pool = any(key.startswith("pool.") for key in model_keys)
    expects_pooling = any(key.startswith("pooling.") for key in model_keys)
    has_pool = any(key.startswith("pool.") for key in state_dict)
    has_pooling = any(key.startswith("pooling.") for key in state_dict)

    if (expects_pool and has_pool) or (expects_pooling and has_pooling) or (not expects_pool and not expects_pooling):
        return state_dict

    aligned: Dict[str, torch.Tensor] = {}
    if expects_pooling and has_pool:
        prefix_len = len("pool.")
        for key, value in state_dict.items():
            if key.startswith("pool."):
                aligned["pooling." + key[prefix_len:]] = value
            else:
                aligned[key] = value
        return aligned

    if expects_pool and has_pooling:
        prefix_len = len("pooling.")
        for key, value in state_dict.items():
            if key.startswith("pooling."):
                aligned["pool." + key[prefix_len:]] = value
            else:
                aligned[key] = value
        return aligned

    return state_dict


def save_checkpoint(args: Any, state: Dict[str, Any], is_best: bool, filename: str) -> None:
    model_path = os.path.join(args.output_folder, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(args.output_folder, "best_model.pth"))
    if getattr(args, "save_every_epoch", False):
        shutil.copyfile(
            model_path,
            os.path.join(args.output_folder, f"epoch_{state['epoch_num']}.pth"),
        )


def resume_train(
    args: Any,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = False,
):
    import logging

    logging.debug("Loading checkpoint: %s", args.resume)
    checkpoint = torch.load(args.resume)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    state_dict = align_pooling_state_dict(model, state_dict)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_overall_recall = checkpoint.get("best_overall_recall", 0.0)
    not_improved_num = checkpoint.get("not_improved_num", 0)
    logging.debug(
        "Loaded checkpoint: start_epoch_num = %d, current_overall_recall = %.1f",
        start_epoch_num + 1,
        best_overall_recall,
    )

    if args.resume.endswith("last_model.pth"):
        best_src = args.resume.replace("last_model.pth", "best_model.pth")
        best_dst = os.path.join(args.output_folder, "best_model.pth")
        if os.path.exists(best_src) and best_src != best_dst:
            shutil.copy(best_src, best_dst)

    return model, optimizer, best_overall_recall, start_epoch_num + 1, not_improved_num


def load_pretrained_backbone(args: Any, model: torch.nn.Module) -> torch.nn.Module:
    import logging

    logging.debug("Loading checkpoint: %s", args.pretrain_model)
    checkpoint = torch.load(args.pretrain_model)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = align_pooling_state_dict(model, state_dict)
    result = model.load_state_dict(state_dict, strict=False)
    for missing_key in result.missing_keys:
        logging.debug("Missing key: %s in pretrained model %s", missing_key, args.pretrain_model)
    return model


def freeze_all_except_attn(model: torch.nn.Module) -> torch.nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.pooling.parameters():
        param.requires_grad = True
    return model


def extract_sequence_descriptor(output):
    """Return the tensor that represents the sequence descriptor."""
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            desc = extract_sequence_descriptor(item)
            if desc is not None:
                return desc
        return None
    return output
