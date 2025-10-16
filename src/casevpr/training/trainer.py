"""Sequence descriptor training for CaseVPR."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import torch
import torch.nn as nn
try:  # Prefer tensorboardX but fall back to PyTorch SummaryWriter.
    from tensorboardX import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - runtime fallback
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except ImportError:
        SummaryWriter = None  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm

from casevpr.utils import ARGS as DEFAULT_ARGS, AttributeDict
from scripts.configs.model_configs import hvpr_encoders, seq_encoders

from .datasets import BaseDataset, TrainDataset, collate_fn
from .evaluation import test
from .logging import setup_logging
from .utils import (
    configure_transform,
    extract_sequence_descriptor,
    freeze_all_except_attn,
    load_pretrained_backbone,
    resume_train,
    save_checkpoint,
)


@dataclass
class ExperimentSection:
    name: str = "default"
    logs_root: str = "output/train_logs"
    seed: int = 43
    deterministic: bool = False
    device: str = "cuda"
    resume: Optional[str] = None
    pretrain_model: Optional[str] = None
    save_every_epoch: bool = False
    only_train_attn: bool = False
    tensorboard: bool = True
    debug: bool = False


@dataclass
class ModelSection:
    seq_encoder: str = "vgg16_seqvlad"
    args_override: Dict[str, Any] = field(default_factory=dict)
    family: Optional[str] = None


@dataclass
class DataSection:
    dataset_path: str = ""
    city: Any = ""
    seq_length: int = 14
    cut_last_frame: bool = False
    train_posDistThr: int = 10
    val_posDistThr: int = 25
    negDistThr: int = 25
    addtest_posDistThr: Optional[int] = None
    img_shape: Optional[List[int]] = None
    seq_gt_strategy: str = "lax"
    neg_seq_gt_strategy: str = "lax"
    reverse: bool = False
    cached_train_dataset: Optional[str] = None


@dataclass
class LoaderSection:
    train_batch_size: int = 4
    infer_batch_size: int = 8
    num_workers: int = 8
    queries_per_epoch: int = 5000
    n_gpus: Optional[int] = None
    test_shuffle: bool = False
    batch_split_size: Optional[int] = None


@dataclass
class MiningSection:
    cached_negatives: int = 1000
    cached_queries: int = 1000
    nNeg: int = 5
    augseq: bool = False
    augseq_max_step: int = 2
    augseq_prob: float = 0.3
    seq_neg_mining: bool = False
    seq_neg_mining_prob: float = 0.3


@dataclass
class OptimSection:
    optim: str = "adam"
    lr: float = 1e-5
    weight_decay: float = 0.0
    multiple_lr: bool = False
    lr_encoder: Optional[float] = None
    lr_pooling: Optional[float] = None
    lr_aggregator: Optional[float] = None
    criterion: str = "triplet"
    margin: float = 0.1
    epochs: int = 100
    patience: int = 3


@dataclass
class EvalSplit:
    split: str
    tboard_name: str
    seq_gt_strategy: Optional[str] = None


@dataclass
class AdditionalTestset:
    path: str
    split: str = "test"
    tboard_name: Optional[str] = None
    seq_gt_strategy: Optional[str] = None


@dataclass
class EvaluationSection:
    val: EvalSplit = field(default_factory=lambda: EvalSplit("val", "val"))
    additional_testsets: List[AdditionalTestset] = field(default_factory=list)
    additional_gt_strategies: List[str] = field(default_factory=list)
    only_test_current_strategy: bool = False


@dataclass
class TrainingConfig:
    experiment: ExperimentSection
    model: ModelSection
    data: DataSection
    loader: LoaderSection
    mining: MiningSection
    optimization: OptimSection
    evaluation: EvaluationSection

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TrainingConfig":
        experiment = ExperimentSection(**cfg.get("experiment", {}))
        model = ModelSection(**cfg.get("model", {}))
        data = DataSection(**cfg.get("data", {}))
        loader = LoaderSection(**cfg.get("loader", {}))
        mining = MiningSection(**cfg.get("mining", {}))
        optimization = OptimSection(**cfg.get("optimization", {}))
        eval_cfg = cfg.get("evaluation", {})
        val_cfg = eval_cfg.get("val")
        splits_legacy = eval_cfg.get("splits")
        legacy_entries: List[EvalSplit] = []
        if val_cfg is None and splits_legacy is not None:
            if isinstance(splits_legacy, dict):
                for key, value in splits_legacy.items():
                    if isinstance(value, dict):
                        legacy_entries.append(EvalSplit(
                            split=str(value.get("split", key)),
                            tboard_name=str(value.get("tboard_name", key)),
                            seq_gt_strategy=value.get("seq_gt_strategy"),
                        ))
                    elif isinstance(value, str):
                        legacy_entries.append(EvalSplit(split=str(value), tboard_name=str(key)))
                    else:
                        raise TypeError(f"Unsupported legacy splits entry type: {type(value)}")
            elif isinstance(splits_legacy, list):
                for entry in splits_legacy:
                    if isinstance(entry, dict):
                        legacy_entries.append(EvalSplit(
                            split=str(entry.get("split", "val")),
                            tboard_name=str(entry.get("tboard_name", entry.get("split", "val"))),
                            seq_gt_strategy=entry.get("seq_gt_strategy"),
                        ))
                    else:
                        legacy_entries.append(EvalSplit(str(entry), str(entry)))
            else:
                raise TypeError("evaluation.splits must be list or dict when provided.")

            if legacy_entries:
                val_cfg = {
                    "split": legacy_entries[0].split,
                    "tboard_name": legacy_entries[0].tboard_name,
                    "seq_gt_strategy": legacy_entries[0].seq_gt_strategy,
                }
                if len(legacy_entries) > 1 and "test" not in eval_cfg:
                    eval_cfg["test"] = {
                        "split": legacy_entries[1].split,
                        "tboard_name": legacy_entries[1].tboard_name,
                        "seq_gt_strategy": legacy_entries[1].seq_gt_strategy,
                    }

        if val_cfg is None:
            val_split = EvalSplit("val", "val")
        elif isinstance(val_cfg, dict):
            val_split = EvalSplit(
                split=str(val_cfg.get("split", "val")),
                tboard_name=str(val_cfg.get("tboard_name", "val")),
                seq_gt_strategy=val_cfg.get("seq_gt_strategy"),
            )
        elif isinstance(val_cfg, str):
            val_split = EvalSplit(split=val_cfg, tboard_name="val")
        else:
            raise TypeError("evaluation.val must be a dict or string if provided.")

        add_tests_cfg = eval_cfg.get("additional_testsets", [])
        additional_testsets: List[AdditionalTestset] = []
        for item in add_tests_cfg:
            if isinstance(item, str):
                additional_testsets.append(AdditionalTestset(path=item))
            elif isinstance(item, dict):
                path_value = item.get("path") or item.get("dataset_path")
                if not path_value:
                    raise ValueError("Each additional_testsets entry must include a 'path'.")
                additional_testsets.append(
                    AdditionalTestset(
                        path=path_value,
                        split=item.get("split", "test"),
                        tboard_name=item.get("tboard_name"),
                        seq_gt_strategy=item.get("seq_gt_strategy"),
                    )
                )
            else:
                raise TypeError(f"Unsupported additional_testsets entry type: {type(item)}")
        evaluation = EvaluationSection(
            val=val_split,
            additional_testsets=additional_testsets,
            additional_gt_strategies=eval_cfg.get("additional_gt_strategies", []),
            only_test_current_strategy=eval_cfg.get("only_test_current_strategy", False),
        )
        return cls(experiment, model, data, loader, mining, optimization, evaluation)


class SequenceTrainer:
    def __init__(self, config: TrainingConfig, config_path: Optional[Path] = None) -> None:
        self.config = config
        self.config_path = config_path
        self.model_family = "seq"
        self.args = self._build_args()
        self.output_folder = Path(self.args.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        setup_logging(str(self.output_folder))
        logging.info("Initialised SequenceTrainer with config: %s", json.dumps(self._config_dict(), indent=2))
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
            "model": {
                "seq_encoder": self.config.model.seq_encoder,
                "args_override": self.config.model.args_override,
                "family": getattr(self, "model_family", "seq"),
            },
            "data": vars(self.config.data),
            "loader": vars(self.config.loader),
            "mining": vars(self.config.mining),
            "optimization": vars(self.config.optimization),
            "evaluation": {
                "val": vars(self.config.evaluation.val),
                "additional_testsets": [
                    {
                        "path": testset.path,
                        "split": testset.split,
                        **({"tboard_name": testset.tboard_name} if testset.tboard_name is not None else {}),
                        **({"seq_gt_strategy": testset.seq_gt_strategy} if testset.seq_gt_strategy is not None else {}),
                    }
                    for testset in self.config.evaluation.additional_testsets
                ],
                "additional_gt_strategies": self.config.evaluation.additional_gt_strategies,
                "only_test_current_strategy": self.config.evaluation.only_test_current_strategy,
            },
        }

    def _build_args(self) -> AttributeDict:
        cfg = self.config
        args = copy.deepcopy(DEFAULT_ARGS)
        exp = cfg.experiment
        data = cfg.data
        loader = cfg.loader
        mining = cfg.mining
        opt = cfg.optimization
        model_cfg = cfg.model
        eval_cfg = cfg.evaluation

        # Experiment level overrides
        args.seed = exp.seed
        args.deterministic = exp.deterministic
        args.device = exp.device
        args.exp_name = exp.name
        args.resume = exp.resume
        args.pretrain_model = exp.pretrain_model
        args.save_every_epoch = exp.save_every_epoch
        args.only_train_attn = exp.only_train_attn

        # Data overrides
        args.dataset_path = data.dataset_path
        args.city = data.city
        args.seq_length = data.seq_length
        args.cut_last_frame = data.cut_last_frame
        args.train_posDistThr = data.train_posDistThr
        args.val_posDistThr = data.val_posDistThr
        args.negDistThr = data.negDistThr
        args.addtest_posDistThr = data.addtest_posDistThr
        args.img_shape = list(data.img_shape) if data.img_shape is not None else None
        args.seq_gt_strategy = data.seq_gt_strategy
        args.neg_seq_gt_strategy = data.neg_seq_gt_strategy
        args.reverse = data.reverse
        args.cached_train_dataset = data.cached_train_dataset

        # Loader overrides
        args.train_batch_size = loader.train_batch_size
        args.infer_batch_size = loader.infer_batch_size
        args.queries_per_epoch = loader.queries_per_epoch
        args.test_shuffle = loader.test_shuffle
        if loader.n_gpus is not None:
            args.n_gpus = loader.n_gpus
        else:
            args.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        args.num_workers = loader.num_workers
        args.batch_split_size = loader.batch_split_size

        # Mining overrides
        args.cached_negatives = mining.cached_negatives
        args.cached_queries = mining.cached_queries
        args.nNeg = mining.nNeg
        args.augseq = mining.augseq
        args.augseq_max_step = mining.augseq_max_step
        args.augseq_prob = mining.augseq_prob
        args.seq_neg_mining = mining.seq_neg_mining
        args.seq_neg_mining_prob = mining.seq_neg_mining_prob

        # Optimisation overrides
        args.optim = opt.optim
        args.lr = opt.lr
        args.weight_decay = opt.weight_decay
        args.multiple_lr = opt.multiple_lr
        args.lr_encoder = opt.lr_encoder
        args.lr_pooling = opt.lr_pooling
        args.lr_aggregator = opt.lr_aggregator
        args.criterion = opt.criterion
        args.margin = opt.margin
        args.epochs_num = opt.epochs
        args.patience = opt.patience

        # Evaluation overrides
        args.add_testsets = cfg.evaluation.additional_testsets
        args.only_test_current_strategy = eval_cfg.only_test_current_strategy
        args.add_test_gt_strategy = eval_cfg.additional_gt_strategies

        # Model overrides
        if model_cfg.seq_encoder in seq_encoders:
            self.model_family = "seq"
            self.model_entry = copy.deepcopy(seq_encoders[model_cfg.seq_encoder])
            self.model_class_args = copy.deepcopy(self.model_entry.get("class_args", {}))
            self.model_class_args.update(model_cfg.args_override)
            if "seq_length" not in model_cfg.args_override:
                self.model_class_args["seq_length"] = args.seq_length
            args.arch = self.model_class_args.get("arch", args.arch)
            args.pooling = self.model_class_args.get("pooling", args.pooling)
            args.aggregation = self.model_class_args.get("aggregation", args.aggregation)
            if "encode_type" in self.model_class_args:
                args.encode_type = self.model_class_args["encode_type"]
            encoder_img_shape = self.model_entry.get("img_shape")
            if data.img_shape is None and encoder_img_shape:
                args.img_shape = list(encoder_img_shape)
        elif model_cfg.seq_encoder in hvpr_encoders:
            self.model_family = "hvpr"
            self.model_entry = copy.deepcopy(hvpr_encoders[model_cfg.seq_encoder])
            self.model_class_args = copy.deepcopy(self.model_entry.get("class_args", {}))
            self.model_class_args.update(model_cfg.args_override)
            self.model_class_args.setdefault("seq_len", args.seq_length)
            args.arch = model_cfg.seq_encoder
            args.pooling = "hvpr"
            args.aggregation = self.model_class_args.get("encoder_type", "hvpr")
            current_encode_type = args.get("encode_type") if isinstance(args, dict) else None
            args.encode_type = self.model_class_args.get("encoder_type", current_encode_type)
            encoder_img_shape = self.model_entry.get("img_shape")
            if data.img_shape is None and encoder_img_shape:
                args.img_shape = list(encoder_img_shape)
            features_dim = self.model_entry.get("features_dim")
            if features_dim is None:
                features_dim = self.model_class_args.get("features_dim", 768 * 14)
            args.features_dim = features_dim
        else:
            raise KeyError(f"Unknown sequence encoder '{model_cfg.seq_encoder}'")

        if args.img_shape is None:
            args.img_shape = data.img_shape or [384, 384]

        dataset_tag = Path(args.dataset_path).name or "dataset"
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_name = f"{timestamp}-{args.arch}-{args.pooling}-{args.aggregation}-{dataset_tag}-{args.exp_name}"
        seed_dir = Path(exp.logs_root) / run_name / str(args.seed)
        if args.resume:
            resume_path = Path(args.resume).resolve()
            args.output_folder = str(resume_path.parent)
        else:
            args.output_folder = str(seed_dir)
        os.makedirs(args.output_folder, exist_ok=True)

        return args

    def train(self) -> Dict[str, Any]:
        args = self.args
        config_dict = self._config_dict()
        safe_args = {k: v for k, v in args.items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
        logging.info("Training arguments: %s", safe_args)
        start_time = datetime.now()

        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        img_shape = (args.img_shape[0], args.img_shape[1])
        transform = configure_transform(image_dim=img_shape, meta=meta)

        if args.cached_train_dataset:
            logging.info("Retrieving train set from cached: %s", args.cached_train_dataset)
            triplets_ds = torch.load(args.cached_train_dataset)
            triplets_ds.bs = args.infer_batch_size
            triplets_ds.seq_len = args.seq_length
            triplets_ds.n_gpus = args.n_gpus
            triplets_ds.transform = transform
            triplets_ds.img_shape = args.img_shape
            triplets_ds.base_transform = transform
            triplets_ds.nNeg = args.nNeg
            triplets_ds.cut_last_frame = args.cut_last_frame
        else:
            logging.info("Loading train set...")
            triplets_ds = TrainDataset(
                cities=args.city,
                dataset_folder=args.dataset_path,
                split='train',
                base_transform=transform,
                seq_len=args.seq_length,
                cut_last_frame=args.cut_last_frame,
                pos_thresh=args.train_posDistThr,
                neg_thresh=args.negDistThr,
                infer_batch_size=args.infer_batch_size,
                n_gpus=args.n_gpus,
                img_shape=args.img_shape,
                cached_negatives=args.cached_negatives,
                cached_queries=args.cached_queries,
                nNeg=args.nNeg,
                seq_gt_strategy=args.seq_gt_strategy,
                neg_seq_gt_strategy=args.neg_seq_gt_strategy,
                augseq=args.augseq,
                augseq_max_step=args.augseq_max_step,
                augseq_prob=args.augseq_prob,
                seq_neg_mining=args.seq_neg_mining,
                seq_neg_mining_prob=args.seq_neg_mining_prob,
            )
        logging.info("Train set: %s", triplets_ds)

        testsets = []
        primary_splits: List[EvalSplit] = [self.config.evaluation.val]
        for split_cfg in primary_splits:
            split_name = split_cfg.split
            ds = BaseDataset(
                dataset_folder=args.dataset_path,
                split=split_name,
                base_transform=transform,
                seq_len=args.seq_length,
                cut_last_frame=args.cut_last_frame,
                pos_thresh=args.val_posDistThr,
                seq_gt_strategy=split_cfg.seq_gt_strategy or args.seq_gt_strategy,
            )
            testsets.append({
                "ds": ds,
                "tboard_name": split_cfg.tboard_name,
                "seq_gt_strategy": split_cfg.seq_gt_strategy or args.seq_gt_strategy,
            })
            logging.info("%s set: %s", split_name.title(), ds)

        if args.add_testsets:
            posDistThr = args.addtest_posDistThr if args.addtest_posDistThr is not None else args.val_posDistThr
            for testset_cfg in args.add_testsets:
                ds = BaseDataset(
                    dataset_folder=testset_cfg.path,
                    split=testset_cfg.split,
                    base_transform=transform,
                    seq_len=args.seq_length,
                    cut_last_frame=args.cut_last_frame,
                    pos_thresh=posDistThr,
                    seq_gt_strategy=testset_cfg.seq_gt_strategy or args.seq_gt_strategy,
                )
                testsets.append({
                    "ds": ds,
                    "tboard_name": testset_cfg.tboard_name or f"test_{Path(testset_cfg.path).name}_{testset_cfg.split}",
                    "seq_gt_strategy": testset_cfg.seq_gt_strategy or args.seq_gt_strategy,
                })
                logging.info("Additional test set: %s", ds)

        if not args.only_test_current_strategy:
            strategy_comps = args.add_test_gt_strategy or (["lax"] if args.seq_gt_strategy == "strict" else ["strict"]) + ["lastframe"]
            for strategy_comp in strategy_comps:
                for split_cfg in primary_splits:
                    ds = BaseDataset(
                        dataset_folder=args.dataset_path,
                        split=split_cfg.split,
                        base_transform=transform,
                        seq_len=args.seq_length,
                        cut_last_frame=args.cut_last_frame,
                        pos_thresh=args.val_posDistThr,
                        seq_gt_strategy=strategy_comp,
                    )
                    testsets.append({
                        "ds": ds,
                        "tboard_name": split_cfg.tboard_name,
                        "seq_gt_strategy": strategy_comp,
                    })
                    logging.info("Complementary %s set (%s): %s", split_cfg.split, strategy_comp, ds)

                    if args.add_testsets:
                        posDistThr = args.addtest_posDistThr if args.addtest_posDistThr is not None else args.val_posDistThr
                        for testset_cfg in args.add_testsets:
                            ds_extra = BaseDataset(
                                dataset_folder=testset_cfg.path,
                                split=testset_cfg.split,
                                base_transform=transform,
                                seq_len=args.seq_length,
                                cut_last_frame=args.cut_last_frame,
                                pos_thresh=posDistThr,
                                seq_gt_strategy=testset_cfg.seq_gt_strategy or strategy_comp,
                            )
                            testsets.append({
                                "ds": ds_extra,
                                "tboard_name": testset_cfg.tboard_name or f"test_{Path(testset_cfg.path).name}_{testset_cfg.split}",
                                "seq_gt_strategy": testset_cfg.seq_gt_strategy or strategy_comp,
                            })
                            logging.info("Complementary additional test set (%s): %s", strategy_comp, ds_extra)

        if self.model_family == "seq":
            model_entry = copy.deepcopy(self.model_entry)
            model_cls = model_entry["class"]
            class_args = copy.deepcopy(self.model_class_args)
            for key, value in class_args.items():
                setattr(args, key, value)
            model = model_cls(args)
        else:
            model_entry = copy.deepcopy(self.model_entry)
            model_cls = model_entry["class"]
            class_args = copy.deepcopy(self.model_class_args)
            class_args.setdefault("seq_len", args.seq_length)
            model = model_cls(**class_args)
            embed_dim = getattr(model.encoder, 'embed_dim', 768)
            args.features_dim = embed_dim * 14
            model.meta = {'outputdim': args.features_dim}

        model_params_json_path = Path(args.output_folder) / "model_params.json"
        model_params = {name: {'shape': list(param.shape), 'requires_grad': param.requires_grad} for name, param in model.named_parameters()}
        model_params_json_path.write_text(json.dumps(model_params, indent=2))

        model = model.to(args.device)
        if self.model_family == "seq":
            if args.pooling in ["netvlad"] and not args.resume and not args.pretrain_model:
                triplets_ds.is_inference = True
                model.pooling.initialize_netvlad_layer(args, triplets_ds, model.encoder)
                triplets_ds.is_inference = False

            if args.aggregation in ["seqvlad", "seqsral", "seqattnvlad"] and not args.resume:
                triplets_ds.is_inference = True
                model.aggregator.initialize_seqvlad_layer(args, triplets_ds, model.encoder)
                triplets_ds.is_inference = False

            if args.only_train_attn:
                model = freeze_all_except_attn(model)

        triplets_ds.features_dim = args.features_dim
        logging.info("Output dimension of the model is %s", getattr(model, 'meta', {}).get('outputdim', args.features_dim))

        if args.optim == "adam":
            optimizer_cls = torch.optim.Adam
        elif args.optim == "adamw":
            optimizer_cls = torch.optim.AdamW
        elif args.optim == "sgd":
            optimizer_cls = torch.optim.SGD
        else:
            raise ValueError(f"Optimizer {args.optim} not supported")

        if args.multiple_lr:
            param_groups = []
            if hasattr(model, "encoder"):
                param_groups.append({'params': model.encoder.parameters(), 'lr': args.lr_encoder or args.lr})
            pooling_module = None
            if hasattr(model, "pooling"):
                pooling_module = model.pooling
            elif hasattr(model, "pool"):
                pooling_module = model.pool
            if pooling_module is not None:
                param_groups.append({'params': pooling_module.parameters(), 'lr': args.lr_pooling or args.lr})
            if hasattr(model, "aggregator"):
                param_groups.append({'params': model.aggregator.parameters(), 'lr': args.lr_aggregator or args.lr})
            if not param_groups:
                raise ValueError("multiple_lr enabled but no parameter groups were collected for the model.")
            optimizer = optimizer_cls(
                param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            additional_kwargs = {}
            if args.optim == "sgd":
                additional_kwargs['momentum'] = 0.9
            optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **additional_kwargs)

        if args.criterion == "triplet":
            criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
        else:
            raise ValueError(f"Criterion {args.criterion} not supported in CaseVPR trainer")

        if args.resume:
            model, optimizer, best_recall, start_epoch_num, not_improved_num = resume_train(args, model, optimizer)
            logging.info("Resuming from epoch %d with best overall_recall %.1f", start_epoch_num, best_recall)
        elif args.pretrain_model and args.arch != 'Official-timesf':
            model = load_pretrained_backbone(args, model)
            best_recall = start_epoch_num = not_improved_num = 0
        else:
            best_recall = start_epoch_num = not_improved_num = 0

        writer = None
        if self.config.experiment.tensorboard:
            if SummaryWriter is None:
                logging.warning(
                    "TensorBoard logging requested but tensorboardX / torch.utils.tensorboard "
                    "is not available. Continuing without TensorBoard."
                )
            else:
                writer = SummaryWriter(args.output_folder)
        if args.n_gpus and args.n_gpus > 0:
            device_ids = list(range(min(args.n_gpus, torch.cuda.device_count())))
        else:
            device_ids = None
        model = torch.nn.DataParallel(model, device_ids=device_ids)

        if writer and not args.resume and args.exp_name != "debug":
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(name, param.data, 0)

        epoch_history = []
        for epoch_num in range(start_epoch_num, args.epochs_num):
            logging.info("Start training epoch: [%02d], mem usage: %s%%", epoch_num, psutil.virtual_memory().percent)
            for ii, param_group in enumerate(optimizer.param_groups):
                if writer:
                    writer.add_scalar(f'Learning_rate/{ii}', param_group['lr'], epoch_num)

            epoch_start_time = datetime.now()
            epoch_losses = np.zeros((0, 1), dtype=np.float32)
            loops_num = math.ceil(args.queries_per_epoch / args.cached_queries)

            for loop_num in range(loops_num):
                logging.debug("Cache: %d / %d", loop_num + 1, loops_num)
                triplets_ds.compute_triplets(model)
                triplets_dl = DataLoader(
                    dataset=triplets_ds,
                    num_workers=args.num_workers,
                    batch_size=args.train_batch_size,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,
                )

                model = model.train()
                for images, _, _ in tqdm(triplets_dl, ncols=100, desc="Training..."):
                    optimizer.zero_grad()
                    per_triplet_count = args.nNeg + 2
                    split_size = args.batch_split_size or args.train_batch_size
                    split_size = max(1, min(split_size, args.train_batch_size))

                    images = images.reshape(args.train_batch_size, per_triplet_count, args.seq_length, 3, *img_shape)

                    batch_loss_value = 0.0
                    for split_start in range(0, args.train_batch_size, split_size):
                        split_end = min(split_start + split_size, args.train_batch_size)
                        current_split = split_end - split_start
                        chunk = images[split_start:split_end].reshape(-1, 3, *img_shape)
                        outputs = model(chunk.to(args.device))
                        features = extract_sequence_descriptor(outputs)
                        features = features.reshape(current_split, -1, args.features_dim)

                        chunk_loss = 0.0
                        for b in range(current_split):
                            query = features[b:b + 1, 0]
                            pos = features[b:b + 1, 1]
                            negatives = features[b, 2:]
                            chunk_loss = chunk_loss + criterion_triplet(query, pos, negatives)

                        chunk_loss = chunk_loss / (args.train_batch_size * args.nNeg)
                        chunk_loss.backward()
                        batch_loss_value += chunk_loss.detach().cpu().item()
                        del features, outputs, chunk

                    optimizer.step()

                    batch_loss = batch_loss_value
                    epoch_losses = np.append(epoch_losses, batch_loss)

                torch.cuda.empty_cache()
                logging.debug(
                    "Epoch[%02d](%d/%d): current batch triplet loss = %.4f, average epoch triplet loss = %.4f, mem usage: %s%%",
                    epoch_num,
                    loop_num + 1,
                    loops_num,
                    batch_loss,
                    epoch_losses.mean(),
                    psutil.virtual_memory().percent,
                )

            if writer:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(name, param.data, epoch_num + 1)

            logging.info(
                "Finished epoch %02d in %s, average epoch triplet loss = %.4f",
                epoch_num,
                str(datetime.now() - epoch_start_time)[:-7],
                epoch_losses.mean(),
            )

            recalls = None
            for i, test_dict in enumerate(testsets):
                recalls_, recalls_str_ = test(
                    args,
                    test_dict["ds"],
                    model,
                    writer=writer,
                    epoch=epoch_num + 1,
                    tboard_name=test_dict["tboard_name"],
                    seq_gt_strategy=test_dict["seq_gt_strategy"],
                )
                logging.info("Recalls on %s_%s set: %s", test_dict['tboard_name'], test_dict['seq_gt_strategy'], recalls_str_)
                if i == 0:
                    recalls = recalls_

            assert recalls is not None
            overall_recall = (recalls[0] + recalls[1]) / 2
            logging.info('Overall recall: %.4f', overall_recall)
            if writer:
                writer.add_scalar('overall_recall', overall_recall, epoch_num)
            is_best = overall_recall > best_recall

            if is_best:
                logging.info(
                    "Improved: previous best overall_recall = %.1f, current R@5 = %.1f, current R@1 = %.1f, overall_recall = %.1f",
                    best_recall,
                    recalls[1],
                    recalls[0],
                    overall_recall,
                )
                best_recall = overall_recall
                not_improved_num = 0
            else:
                not_improved_num += 1
                logging.info(
                    "Not improved: %d / %d: best overall_recall = %.1f, current R@5 = %.1f, current R@1 = %.1f, current overall_recall = %.1f",
                    not_improved_num,
                    args.patience,
                    best_recall,
                    recalls[1],
                    recalls[0],
                    overall_recall,
                )

            save_checkpoint(
                args,
                {
                    "epoch_num": epoch_num,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "recalls": recalls,
                    "best_overall_recall": best_recall,
                    "not_improved_num": not_improved_num,
                    "config": config_dict,
                },
                is_best,
                filename="last_model.pth",
            )

            epoch_history.append({
                "epoch": epoch_num,
                "loss": float(epoch_losses.mean()),
                "overall_recall": float(overall_recall),
                "metric": recalls.tolist(),
                "is_best": bool(is_best),
            })

            if not is_best and not_improved_num >= args.patience:
                logging.info("Performance did not improve for %d epochs. Stop training.", not_improved_num)
                break

        logging.info("Best overall_recall: %.1f", best_recall)
        logging.info("Training completed in %s", str(datetime.now() - start_time)[:-7])

        best_model_path = Path(args.output_folder) / "best_model.pth"
        if best_model_path.exists():
            best_model_state_dict = torch.load(best_model_path)["model_state_dict"]
            model.load_state_dict(best_model_state_dict)

            for test_dict in testsets:
                _, recalls_str_ = test(args, test_dict["ds"], model)
                logging.info("Recalls on %s_%s set: %s", test_dict['tboard_name'], test_dict['seq_gt_strategy'], recalls_str_)
        else:
            logging.warning("best_model.pth not found in %s; skipping final evaluation.", args.output_folder)

        if writer:
            writer.close()

        return {
            "best_overall_recall": best_recall,
            "epoch_history": epoch_history,
            "output_folder": str(self.output_folder),
        }
