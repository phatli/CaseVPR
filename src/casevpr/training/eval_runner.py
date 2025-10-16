"""Config-driven sequence evaluation runner."""
from __future__ import annotations

import copy
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from casevpr.utils import ARGS as DEFAULT_ARGS
from scripts.configs.model_configs import hvpr_encoders, seq_encoders

from .datasets import BaseDataset, PCADataset
from .evaluation import test
from .logging import setup_logging
from .result_visualization import generate_gif, get_html_result_map, save_predictions_json
from .utils import configure_transform, extract_sequence_descriptor, align_pooling_state_dict


@dataclass
class EvalExperiment:
    name: str = "eval"
    logs_root: str = "output/seq_eval_logs"
    seed: int = 43
    deterministic: bool = False
    device: str = "cuda"
    save_preds_json: bool = True
    save_gif: bool = False
    save_html: bool = False
    infer_batch_size: Optional[int] = None


@dataclass
class EvalModel:
    seq_encoder: str = "vgg16_seqvlad"
    checkpoint: str = ""
    use_best: bool = True
    pca_outdim: Optional[int] = None


@dataclass
class EvalData:
    dataset_path: str = ""
    split: str = "test"
    seq_length: int = 14
    cut_last_frame: bool = False
    val_posDistThr: int = 25
    img_shape: Optional[list[int]] = None
    seq_gt_strategy: str = "lax"
    reverse: bool = False


@dataclass
class EvaluationConfig:
    experiment: EvalExperiment
    model: EvalModel
    data: EvalData

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "EvaluationConfig":
        return cls(
            experiment=EvalExperiment(**cfg.get("experiment", {})),
            model=EvalModel(**cfg.get("model", {})),
            data=EvalData(**cfg.get("data", {})),
        )


class SequenceEvaluator:
    def __init__(
        self,
        config: EvaluationConfig,
        config_path: Optional[Path] = None,
        override_output_folder: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.override_output_folder = Path(override_output_folder) if override_output_folder else None
        self.model_family = "seq"
        self.args = self._build_args()
        self.output_folder = Path(self.args.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        setup_logging(str(self.output_folder), console="info")
        logging.info("Initialised SequenceEvaluator with config: %s", json.dumps(self._config_dict(), indent=2))
        if config_path and config_path.exists():
            dest = (self.output_folder / config_path.name).resolve()
            src = config_path.resolve()
            if dest != src:
                shutil.copy(src, dest)
        config_dump_path = self.output_folder / "config_used.json"
        config_dump_path.write_text(json.dumps(self._config_dict(), indent=2))

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "experiment": vars(self.config.experiment),
            "model": {**vars(self.config.model), "family": self.model_family},
            "data": vars(self.config.data),
        }

    def _build_args(self):
        cfg = self.config
        args = copy.deepcopy(DEFAULT_ARGS)
        exp = cfg.experiment
        data = cfg.data
        model_cfg = cfg.model

        args.exp_name = exp.name
        args.device = exp.device
        args.seed = exp.seed
        args.deterministic = exp.deterministic
        args.dataset_path = data.dataset_path
        args.seq_length = data.seq_length
        args.cut_last_frame = data.cut_last_frame
        args.val_posDistThr = data.val_posDistThr
        args.img_shape = None
        args.seq_gt_strategy = data.seq_gt_strategy
        args.reverse = data.reverse
        args.infer_batch_size = DEFAULT_ARGS.infer_batch_size
        if exp.infer_batch_size is not None:
            args.infer_batch_size = exp.infer_batch_size
        args.test_shuffle = False
        args.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if model_cfg.seq_encoder in seq_encoders:
            self.model_family = "seq"
            self.model_entry = copy.deepcopy(seq_encoders[model_cfg.seq_encoder])
            self.model_class_args = copy.deepcopy(self.model_entry.get("class_args", {}))
            self.model_class_args.setdefault("seq_length", args.seq_length)
            args.arch = self.model_class_args.get("arch", args.arch)
            args.pooling = self.model_class_args.get("pooling", args.pooling)
            args.aggregation = self.model_class_args.get("aggregation", args.aggregation)
            encoder_img_shape = self.model_entry.get("img_shape")
            if args.img_shape is None and encoder_img_shape:
                args.img_shape = list(encoder_img_shape)
        elif model_cfg.seq_encoder in hvpr_encoders:
            self.model_family = "hvpr"
            self.model_entry = copy.deepcopy(hvpr_encoders[model_cfg.seq_encoder])
            self.model_class_args = copy.deepcopy(self.model_entry.get("class_args", {}))
            self.model_class_args.setdefault("seq_len", args.seq_length)
            args.arch = model_cfg.seq_encoder
            args.pooling = "hvpr"
            args.aggregation = self.model_class_args.get("encoder_type", "hvpr")
            encoder_img_shape = self.model_entry.get("img_shape")
            if args.img_shape is None and encoder_img_shape:
                args.img_shape = list(encoder_img_shape)
            existing_encode_type = args.get("encode_type") if isinstance(args, dict) else None
            args.encode_type = self.model_class_args.get("encoder_type", existing_encode_type)
            features_dim = self.model_entry.get("features_dim")
            if features_dim is None:
                features_dim = self.model_class_args.get("features_dim", 768 * 14)
            args.features_dim = features_dim
        else:
            raise KeyError(f"Unknown sequence encoder '{model_cfg.seq_encoder}'")

        if args.img_shape is None:
            args.img_shape = [384, 384]

        if self.override_output_folder is not None:
            self.override_output_folder.mkdir(parents=True, exist_ok=True)
            args.output_folder = str(self.override_output_folder)
        else:
            dataset_tag = Path(args.dataset_path).name or "dataset"
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            run_name = f"{timestamp}-{args.arch}-{args.pooling}-{args.aggregation}-{dataset_tag}-{args.exp_name}"
            args.output_folder = str(Path(exp.logs_root) / run_name)
            os.makedirs(args.output_folder, exist_ok=True)

        checkpoint_override = model_cfg.checkpoint or self.model_entry.get("ckpt_path")
        if not checkpoint_override:
            raise ValueError(f"No checkpoint provided for encoder '{model_cfg.seq_encoder}' and no default ckpt_path found.")
        self.checkpoint_path = Path(checkpoint_override)
        if not model_cfg.checkpoint:
            logging.info("Using default checkpoint %s for encoder %s", self.checkpoint_path, model_cfg.seq_encoder)
        self.use_best = model_cfg.use_best
        self.pca_outdim = model_cfg.pca_outdim

        return args

    def _load_model(self):
        args = self.args
        model_entry = copy.deepcopy(self.model_entry)
        model_cls = model_entry["class"]
        if self.model_family == "seq":
            for key, value in self.model_class_args.items():
                setattr(args, key, value)
            model = model_cls(args)
        else:
            class_args = copy.deepcopy(self.model_class_args)
            class_args.setdefault("seq_len", args.seq_length)
            model = model_cls(**class_args)
            model_meta_dim = getattr(model, "meta", {}).get("outputdim")
            configured_dim = self.model_entry.get("features_dim") or class_args.get("features_dim")
            if model_meta_dim is not None:
                args.features_dim = model_meta_dim
            elif configured_dim is not None:
                args.features_dim = configured_dim
                if not hasattr(model, "meta") or not isinstance(model.meta, dict):
                    model.meta = {}
                model.meta['outputdim'] = args.features_dim
            else:
                embed_dim = getattr(model.encoder, 'embed_dim', 768)
                args.features_dim = embed_dim * 14
                model.meta = {'outputdim': args.features_dim}

        checkpoint_path = self._resolve_checkpoint_path()
        state_dict = torch.load(checkpoint_path)["model_state_dict"]
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = align_pooling_state_dict(model, state_dict)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            logging.warning("Missing checkpoint parameters: %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            logging.warning("Unexpected checkpoint parameters: %s", load_result.unexpected_keys)
        model = model.to(args.device)
        model = torch.nn.DataParallel(model)
        if hasattr(model.module, 'meta') and 'outputdim' in model.module.meta:
            args.features_dim = model.module.meta['outputdim']
        return model

    def _resolve_checkpoint_path(self) -> Path:
        if self.checkpoint_path.is_file():
            if self.use_best:
                best_path = self.checkpoint_path.parent / "best_model.pth"
                return best_path if best_path.exists() else self.checkpoint_path
            return self.checkpoint_path
        elif self.checkpoint_path.is_dir():
            target = self.checkpoint_path / ("best_model.pth" if self.use_best else "last_model.pth")
            if not target.exists():
                raise FileNotFoundError(f"Checkpoint not found: {target}")
            return target
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {self.checkpoint_path}")

    def _compute_pca(self, model) -> Optional[PCA]:
        if self.pca_outdim is None:
            return None
        args = self.args
        pca_path = Path(self.checkpoint_path).parent / f"pca{self.pca_outdim}.pkl"
        full_features_dim = args.features_dim
        args.features_dim = self.pca_outdim
        if pca_path.exists():
            pca = torch.load(pca_path)
            logging.info("Loaded PCA from %s", pca_path)
        else:
            logging.info('Computing PCA descriptors...')
            pca_ds = PCADataset(
                dataset_folder=args.dataset_path,
                split='train',
                base_transform=self.transform,
                seq_len=args.seq_length,
            )
            num_images = min(len(pca_ds), 2 ** 14)
            if num_images < len(pca_ds):
                idxs = np.random.choice(len(pca_ds), num_images, replace=False)
            else:
                idxs = np.arange(len(pca_ds))
            subset_ds = Subset(pca_ds, idxs.tolist())
            dl = DataLoader(subset_ds, args.infer_batch_size)
            pca_features = np.empty([len(idxs), full_features_dim])
            with torch.no_grad():
                for i, sequences in enumerate(tqdm(dl, ncols=100, desc="PCA features")):
                    if len(sequences.shape) == 5:
                        sequences = sequences.view(-1, 3, args.img_shape[0], args.img_shape[1])
                    outputs = model(sequences.to(args.device))
                    features = extract_sequence_descriptor(outputs).cpu().numpy()
                    start = i * args.infer_batch_size
                    pca_features[start:start + len(features)] = features
            pca = PCA(self.pca_outdim)
            pca.fit(pca_features)
            torch.save(pca, pca_path)
            logging.info("Saved PCA to %s", pca_path)
        args.features_dim = self.pca_outdim
        return pca

    def evaluate(self) -> Dict[str, Any]:
        args = self.args
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = not args.deterministic
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except RuntimeError:
                torch.use_deterministic_algorithms(True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        else:
            torch.backends.cudnn.deterministic = False
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        self.transform = configure_transform((args.img_shape[0], args.img_shape[1]), meta)

        eval_ds = BaseDataset(
            dataset_folder=args.dataset_path,
            split=self.config.data.split,
            base_transform=self.transform,
            seq_len=args.seq_length,
            pos_thresh=args.val_posDistThr,
            cut_last_frame=args.cut_last_frame,
            reverse_frames=args.reverse,
            seq_gt_strategy=args.seq_gt_strategy,
        )
        logging.info("Evaluation dataset: %s", eval_ds)

        model = self._load_model()
        pca = self._compute_pca(model)
        data_generator = None
        if args.seed is not None:
            data_generator = torch.Generator()
            data_generator.manual_seed(args.seed)

        recalls, recalls_str, predictions = test(
            args,
            eval_ds,
            model,
            pca=pca,
            generator=data_generator,
            return_preds=True,
        )
        if pca is not None:
            model.module.meta['outputdim'] = self.pca_outdim
        logging.info("Output dimension of the model is %s", model.module.meta['outputdim'])
        logging.info("Recalls on %s set: %s", self.config.data.split, recalls_str)

        outputs = {
            "recalls": recalls.tolist(),
            "summary": recalls_str,
            "output_folder": str(self.output_folder),
        }

        if self.config.experiment.save_preds_json:
            preds_json_path = self.output_folder / "predictions.json"
            preds_records = []
            for i, pred in enumerate(predictions):
                q_path = eval_ds.q_paths[eval_ds.qIdx[i]]
                db_path = eval_ds.db_paths[pred[0]]
                is_correct = int(pred[0] in eval_ds.pIdx[i])
                preds_records.append([q_path, db_path, bool(is_correct)])
            save_predictions_json(preds_records, str(preds_json_path))
            if self.config.experiment.save_gif:
                generate_gif(preds_records, eval_ds.dataset_folder, str(preds_json_path))
            if self.config.experiment.save_html:
                html_path = self.output_folder / f"{self.output_folder.name}.html"
                get_html_result_map(preds_records, eval_ds.dataset_folder, str(html_path))

        return outputs
