from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import load_config, set_seed, train_model_bundle
from src.preprocessing import load_dataset


def train(args):
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    train_frame = load_dataset(args.train)
    print(
        f"Training samples: {len(train_frame)}; "
        f"labels: {train_frame['label'].value_counts().sort_index().to_dict()}"
    )

    bundle = train_model_bundle(train_frame, config, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    print(f"Saved model: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the rumor detection model.")
    parser.add_argument("--train", type=Path, default=Path("data/split/train.csv"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/best_model.joblib"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models/cache"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
