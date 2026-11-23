"""Command-line parameters for the PriorFuse five-fold experiment."""

import argparse
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train PriorFuse with patient-level stratified five-fold cross-validation."
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--data-root", type=Path, default=PROJECT_ROOT,
                       help="Directory containing label/, train/ and test/.")
    paths.add_argument("--clip-weights", type=Path, default=PROJECT_ROOT / "weights/ViT-B-16.pt")
    paths.add_argument("--brainiac-weights", type=Path, default=PROJECT_ROOT / "MFM/BrainIAC.ckpt")
    paths.add_argument("--medclip-dir", type=Path, default=PROJECT_ROOT / "MFM/medclip-vit",
                       help="Directory for the official MedCLIP ViT checkpoint. Downloaded on first use.")
    paths.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results",
                       help="Parent directory for a new experiment run.")

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--num-workers", type=int, default=0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--lora-rank", type=int, default=8)
    training.add_argument("--code-number", type=int, default=16)
    training.add_argument("--temperature", type=float, default=0.07)
    training.add_argument("--label-smoothing", type=float, default=0.05,
                          help="Binary target smoothing in [0, 0.5).")

    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size - 1, args.lora_rank, args.code_number) < 1:
        parser.error("epochs, rank and codebook size must be positive and batch size must be at least 2")
    if args.num_workers < 0 or not 0 <= args.seed < 2**32 - 5:
        parser.error("num-workers must be nonnegative and seed must be in [0, 2**32 - 5)")
    if not 0 <= args.label_smoothing < 0.5:
        parser.error("label smoothing must be in [0, 0.5)")
    loss_options = (args.lr, args.temperature, args.weight_decay)
    if not (all(map(math.isfinite, loss_options)) and args.lr > 0 and args.temperature > 0
            and args.weight_decay >= 0):
        parser.error("invalid learning rate, temperature or weight decay")
    return args
