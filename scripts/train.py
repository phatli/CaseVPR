#!/usr/bin/env python
"""Train CaseVPR sequence encoders.

Config reference:
    experiment:
        name – tag appended to run folders & TensorBoard logs.
        logs_root – root directory where runs/<seed> are created.
        seed – forwarded to torch/numpy/cuda for reproducibility.
        deterministic – toggles cuDNN deterministic mode when true.
        device – training device string passed through to args.device.
        resume – optional checkpoint path to resume from.
        pretrain_model – checkpoint loaded before training if not resuming.
        save_every_epoch – also dumps epoch_{n}.pth for each checkpoint.
        only_train_attn – freezes model except attention blocks.
    model:
        seq_encoder – key in hvpr_encoders or seq_encoders (see
            scripts/configs/model_configs.py) such as hvpr_casevpr_224,
            hvpr_casevpr_224_crica, hvpr_casevpr_322, hvpr_seqnet,
            vgg16_seqvlad, jist, svpr.
        args_override – per-model kwargs (e.g. seq_len, encoder_type ∈
            {b_sd_c, bs_d_c, sd_b_c}).
    data:
        dataset_path – formatted dataset root containing split/{database,queries}.
        city – optional string/list filter applied by TrainDataset.
        seq_length – frames per sequence fed to the encoder.
        cut_last_frame – drops final image from each stored sequence.
        train_posDistThr – radius (metres) for positives during mining.
        val_posDistThr – positive radius used for validation/test recall.
        negDistThr – radius for non-negatives excluded from sampling.
        addtest_posDistThr – overrides val_posDistThr for add. testsets.
        img_shape – resize target [H, W]; falls back to encoder defaults.
        seq_gt_strategy – {lax, strict, lastframe} positive labelling.
        neg_seq_gt_strategy – {lax, strict} non-negative mining regime.
        reverse – flips sequence order for evaluation if requested.
        cached_train_dataset – path to pre-cached TrainDataset state.
    loader:
        train_batch_size – batch size for triplet loss loops.
        infer_batch_size – batch size for descriptor extraction/mining.
        num_workers – DataLoader workers for training caches.
        queries_per_epoch – queries processed each epoch before refresh.
        n_gpus – manual GPU count (auto when null).
        test_shuffle – shuffle gallery batches in evaluation loaders.
        batch_split_size – optional micro-batch size for gradient accumulation within each training batch.
    mining:
        cached_negatives – database sequences cached each refresh.
        cached_queries – queries cached each refresh.
        nNeg – negatives sampled per query when forming triplets.
        augseq – enable frame-replacement augmentation.
        augseq_max_step – max frame offset for augmentation swaps.
        augseq_prob – probability of applying augmentation to a sample.
        seq_neg_mining – enable GPS-based sequence hard-negative mining.
        seq_neg_mining_prob – chance to replace a negative with mined seq.
    optimization:
        optim – optimiser type {adam, adamw, sgd}.
        lr – base learning rate applied to optimiser/groups.
        lr_encoder/lr_pooling/lr_aggregator – per-module LRs when
            multiple_lr is true.
        multiple_lr – split optimiser into encoder/pooling/aggregator
            parameter groups.
        weight_decay – optimiser weight decay factor.
        criterion – currently supports "triplet" loss.
        margin – triplet loss margin in descriptor space.
        epochs – maximum training epochs.
        patience – early-stopping patience (epochs w/out improvement).
    evaluation:
        val – descriptor for the primary validation split
            ({split, tboard_name?, seq_gt_strategy?}); this split drives
            checkpoint selection and early stopping.
        additional_testsets – extra datasets evaluated post-epoch. Accepts
            strings (treated as paths with split="test") or objects with
            {path, split?, tboard_name?, seq_gt_strategy?}.
        additional_gt_strategies – extra GT strategies (lax/strict/lastframe).
        only_test_current_strategy – skip complementary sweeps when true.
CLI:
    --dry-run – validate the config & exit before any training begins.
    --gpu-id – optionally mask training to a single GPU index for the run.
    --resume-logdir – resume from an existing output folder using its stored config.
"""
import argparse
import json
import os
import sys
from pathlib import Path

CASEVPR_ROOT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-2])
if CASEVPR_ROOT_DIR not in sys.path:
    sys.path.insert(0, CASEVPR_ROOT_DIR)


# argparse helpers ---------------------------------------------------------
def non_negative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("gpu_id must be non-negative.")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sequence encoders for CaseVPR")
    parser.add_argument("--config", default=None, help="Path to the training config JSON file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and exit without running training",
    )
    parser.add_argument(
        "--resume-logdir",
        default=None,
        help="Resume from an output folder that contains config_used.json and checkpoints.",
    )
    parser.add_argument(
        "--gpu-id",
        type=non_negative_int,
        default=None,
        help="Select a single GPU by index (overrides config/device visibility).",
    )
    args = parser.parse_args()
    if args.resume_logdir:
        if args.config:
            parser.error("--config cannot be used together with --resume-logdir.")
    elif not args.config:
        parser.error("--config is required unless --resume-logdir is provided.")
    return args


def main() -> None:
    args = parse_args()
    resume_logdir = Path(args.resume_logdir).expanduser().resolve() if args.resume_logdir else None

    selected_device = None
    selected_gpu_count = None
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        selected_device = "cuda"
        selected_gpu_count = 1

    from casevpr.training import TrainingConfig, SequenceTrainer
    from casevpr.training.logging import stop_logging

    if resume_logdir:
        if not resume_logdir.exists():
            raise FileNotFoundError(f"Resume folder not found: {resume_logdir}")
        resume_config_path = resume_logdir / "config_used.json"
        if not resume_config_path.exists():
            raise FileNotFoundError(f"config_used.json not found in {resume_logdir}")
        config_path = resume_config_path.resolve()
        with config_path.open() as f:
            raw_config = json.load(f)
    else:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open() as f:
            raw_config = json.load(f)

    training_config = TrainingConfig.from_dict(raw_config)

    if resume_logdir:
        resume_checkpoint = resume_logdir / "last_model.pth"
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"last_model.pth not found in {resume_logdir}")
        training_config.experiment.resume = str(resume_checkpoint.resolve())

    if selected_device is not None:
        training_config.experiment.device = selected_device
    if selected_gpu_count is not None:
        training_config.loader.n_gpus = selected_gpu_count

    if args.dry_run:
        print("Configuration parsed successfully. Exiting due to --dry-run.")
        return

    trainer = SequenceTrainer(training_config, config_path=config_path)
    result = trainer.train()
    stop_logging()

    print("Training finished. Summary:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
