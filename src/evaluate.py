from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import score_models
from src.preprocessing import load_dataset


MODEL_THRESHOLDS = {
    "word_tfidf_logistic_regression": 0.0,
    "word_char_tfidf_linear_svc": 0.0,
    "tfidf_ensemble": 0.0,
    "bertweet_linear_svc": 0.0,
    "mpnet_logistic_regression": 0.0,
}


def metric_row(name, labels, predictions, threshold):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix(labels, predictions, output_path):
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]),
        display_labels=["non-rumor", "rumor"],
    )
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_title("Final Model Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=200)
    plt.close(display.figure_)


def evaluate(args):
    validation = load_dataset(args.val)
    labels = validation["label"].to_numpy()
    bundle = joblib.load(args.model)
    scores = score_models(
        bundle,
        validation["text"],
        cache_dir=args.cache_dir,
        cache_key="val",
    )

    rows = []
    predictions_by_model = {}
    for name, model_scores in scores.items():
        threshold = (
            float(bundle["threshold"])
            if name == "final_weighted_fusion"
            else MODEL_THRESHOLDS[name]
        )
        predictions = (np.asarray(model_scores) >= threshold).astype(int)
        predictions_by_model[name] = predictions
        rows.append(metric_row(name, labels, predictions, threshold))

    comparison = pd.DataFrame(rows).sort_values(
        ["accuracy", "f1"],
        ascending=False,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)

    final_name = "final_weighted_fusion"
    final_predictions = predictions_by_model[final_name]
    final_metrics = next(row for row in rows if row["model"] == final_name)
    report = classification_report(
        labels,
        final_predictions,
        labels=[0, 1],
        target_names=["non-rumor", "rumor"],
        digits=4,
    )
    (args.results_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )
    save_confusion_matrix(
        labels,
        final_predictions,
        args.results_dir / "confusion_matrix.png",
    )

    output = {
        "model": final_name,
        "model_path": str(args.model),
        "validation_size": int(len(validation)),
        "threshold": float(bundle["threshold"]),
        "weights": bundle["weights"],
        "metrics": {
            key: value
            for key, value in final_metrics.items()
            if key not in {"model", "threshold"}
        },
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    display_columns = [column for column in comparison.columns if column != "threshold"]
    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print(f"\nSaved evaluation results to: {args.results_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rumor detection models.")
    parser.add_argument("--val", type=Path, default=Path("data/split/val.csv"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best_model.joblib"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models/cache"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
