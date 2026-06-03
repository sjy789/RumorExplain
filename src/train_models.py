from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import TweetStatsTransformer, clean_for_word, normalize_tweet


def make_word_vectorizer(ngram_range=(1, 2), stop_words="english", max_features=30000):
    return TfidfVectorizer(
        preprocessor=clean_for_word,
        analyzer="word",
        ngram_range=ngram_range,
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
        stop_words=stop_words,
    )


def make_char_vectorizer(ngram_range=(3, 5), max_features=50000):
    return TfidfVectorizer(
        preprocessor=normalize_tweet,
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
    )


def make_word_char_features(
    word_ngram=(1, 2),
    char_ngram=(3, 5),
    stop_words="english",
    word_weight=1.0,
    char_weight=1.0,
):
    return FeatureUnion(
        [
            ("word", make_word_vectorizer(word_ngram, stop_words)),
            ("char", make_char_vectorizer(char_ngram)),
        ],
        transformer_weights={"word": word_weight, "char": char_weight},
    )


def make_event_stats_features():
    return ColumnTransformer(
        [
            ("word", make_word_vectorizer((1, 2), "english"), "text"),
            ("char", make_char_vectorizer((3, 5)), "text"),
            ("event", OneHotEncoder(handle_unknown="ignore"), ["event"]),
            (
                "stats",
                Pipeline([("stats", TweetStatsTransformer()), ("scale", StandardScaler())]),
                "text",
            ),
        ]
    )


def model_specs():
    return [
        {
            "name": "word12_lr_c1_baseline",
            "input": "text",
            "features": make_word_vectorizer((1, 2), "english", 20000),
            "clf": LogisticRegression(C=1.0, max_iter=3000, solver="liblinear"),
        },
        {
            "name": "word12_svc_bal_c05",
            "input": "text",
            "features": make_word_vectorizer((1, 2), None, None),
            "clf": LinearSVC(C=0.5, class_weight="balanced", max_iter=8000),
        },
        {
            "name": "char35_lr_c5",
            "input": "text",
            "features": make_char_vectorizer((3, 5)),
            "clf": LogisticRegression(C=5.0, max_iter=3000, solver="liblinear"),
        },
        {
            "name": "char36_svc_c1",
            "input": "text",
            "features": make_char_vectorizer((3, 6), 60000),
            "clf": LinearSVC(C=1.0, max_iter=8000),
        },
        {
            "name": "wc12_35_stop_svc_bal_c05",
            "input": "text",
            "features": make_word_char_features((1, 2), (3, 5), "english"),
            "clf": LinearSVC(C=0.5, class_weight="balanced", max_iter=8000),
        },
        {
            "name": "wc12_35_nostop_svc_bal_c03",
            "input": "text",
            "features": make_word_char_features((1, 2), (3, 5), None),
            "clf": LinearSVC(C=0.3, class_weight="balanced", max_iter=8000),
        },
        {
            "name": "wc12_35_nostop_svc_bal_c05",
            "input": "text",
            "features": make_word_char_features((1, 2), (3, 5), None),
            "clf": LinearSVC(C=0.5, class_weight="balanced", max_iter=8000),
        },
        {
            "name": "wc12_35_nostop_svc_bal_c08",
            "input": "text",
            "features": make_word_char_features((1, 2), (3, 5), None),
            "clf": LinearSVC(C=0.8, class_weight="balanced", max_iter=8000),
        },
        {
            "name": "wc12_35_nostop_svc_bal_c1",
            "input": "text",
            "features": make_word_char_features((1, 2), (3, 5), None),
            "clf": LinearSVC(C=1.0, class_weight="balanced", max_iter=8000),
        },
        {
            "name": "wc12_35_nostop_lr_c3_bal",
            "input": "text",
            "features": make_word_char_features((1, 2), (3, 5), None),
            "clf": LogisticRegression(
                C=3.0, class_weight="balanced", max_iter=3000, solver="liblinear"
            ),
        },
        {
            "name": "wc12_35_nostop_lr_c5",
            "input": "text",
            "features": make_word_char_features((1, 2), (3, 5), None),
            "clf": LogisticRegression(C=5.0, max_iter=3000, solver="liblinear"),
        },
        {
            "name": "wc13_35_stop_svc_bal_c05",
            "input": "text",
            "features": make_word_char_features((1, 3), (3, 5), "english"),
            "clf": LinearSVC(C=0.5, class_weight="balanced", max_iter=8000),
        },
        {
            "name": "wc13_35_stop_svc_bal_c08",
            "input": "text",
            "features": make_word_char_features((1, 3), (3, 5), "english"),
            "clf": LinearSVC(C=0.8, class_weight="balanced", max_iter=8000),
        },
        {
            "name": "wc13_35_stop_svc_bal_c1",
            "input": "text",
            "features": make_word_char_features((1, 3), (3, 5), "english"),
            "clf": LinearSVC(C=1.0, class_weight="balanced", max_iter=8000),
        },
        {
            "name": "wc13_35_stop_lr_c5",
            "input": "text",
            "features": make_word_char_features((1, 3), (3, 5), "english"),
            "clf": LogisticRegression(C=5.0, max_iter=3000, solver="liblinear"),
        },
        {
            "name": "wc12_35_nostop_ridge1",
            "input": "text",
            "features": make_word_char_features((1, 2), (3, 5), None),
            "clf": RidgeClassifier(alpha=1.0),
        },
        {
            "name": "wc12_35_stop_sgd_log",
            "input": "text",
            "features": make_word_char_features((1, 2), (3, 5), "english"),
            "clf": SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                max_iter=3000,
                tol=1e-4,
                random_state=42,
            ),
        },
        {
            "name": "word12_complement_nb",
            "input": "text",
            "features": make_word_vectorizer((1, 2), "english", 20000),
            "clf": ComplementNB(alpha=0.1),
        },
        {
            "name": "word13_multinomial_nb",
            "input": "text",
            "features": make_word_vectorizer((1, 3), "english", 30000),
            "clf": MultinomialNB(alpha=0.1),
        },
        {
            "name": "wc_event_stats_svc_c1",
            "input": "frame",
            "features": make_event_stats_features(),
            "clf": LinearSVC(C=1.0, max_iter=8000),
        },
        {
            "name": "wc_event_stats_ridge1",
            "input": "frame",
            "features": make_event_stats_features(),
            "clf": RidgeClassifier(alpha=1.0),
        },
        {
            "name": "event_stats_random_forest",
            "input": "frame",
            "features": ColumnTransformer(
                [
                    ("event", OneHotEncoder(handle_unknown="ignore"), ["event"]),
                    (
                        "stats",
                        Pipeline(
                            [
                                ("stats", TweetStatsTransformer()),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        "text",
                    ),
                ]
            ),
            "clf": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
            ),
        },
    ]


def input_for(df, input_mode):
    if input_mode == "frame":
        return df[["text", "event"]]
    return df["text"]


def scores_for(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)), "decision_function"
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1], "predict_proba"
    return None, None


def best_threshold(scores, y_true, default_threshold):
    unique_scores = np.unique(scores)
    if unique_scores.size == 1:
        candidates = np.array([unique_scores[0]])
    else:
        mids = (unique_scores[:-1] + unique_scores[1:]) / 2
        eps = max(1e-12, np.std(scores) * 1e-9)
        candidates = np.r_[unique_scores[0] - eps, unique_scores, mids, unique_scores[-1] + eps]

    best = None
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        accuracy = accuracy_score(y_true, pred)
        key = (accuracy, f1, -abs(float(threshold) - default_threshold))
        if best is None or key > best["key"]:
            best = {
                "threshold": float(threshold),
                "pred": pred,
                "key": key,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return best


def metric_row(model_name, variant, pred, y_true, threshold=None, input_mode="text", deployable=True):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "variant": variant,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "input_mode": input_mode,
        "deployable_text_only": deployable,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def majority_prediction(train_df, val_df):
    label = int(train_df["label"].mode()[0])
    return np.full(len(val_df), label, dtype=int)


def event_majority_prediction(train_df, val_df):
    default_label = int(train_df["label"].mode()[0])
    event_map = train_df.groupby("event")["label"].agg(lambda value: int(value.mean() >= 0.5))
    return val_df["event"].map(event_map).fillna(default_label).astype(int).to_numpy()


def ensemble_scores(members, X):
    normalized_scores = []
    for member in members:
        scores, _ = scores_for(member["model"], X)
        normalized_scores.append((scores - member["mean"]) / member["std"])
    return np.vstack(normalized_scores).mean(axis=0)


def build_ensembles(fitted_models, train_df, val_df, y_val):
    text_models = [
        item
        for item in fitted_models
        if item["input_mode"] == "text" and item["score_method"] is not None
    ]
    text_models = sorted(text_models, key=lambda item: item["tuned_accuracy"], reverse=True)
    ensembles = []
    for top_k in (3, 5, 7):
        if len(text_models) < top_k:
            continue
        members = []
        X_train = input_for(train_df, "text")
        X_val = input_for(val_df, "text")
        for item in text_models[:top_k]:
            train_scores, _ = scores_for(item["model"], X_train)
            std = float(np.std(train_scores))
            members.append(
                {
                    "name": item["name"],
                    "model": item["model"],
                    "score_method": item["score_method"],
                    "mean": float(np.mean(train_scores)),
                    "std": std if std > 1e-12 else 1.0,
                }
            )
        val_scores = ensemble_scores(members, X_val)
        tuned = best_threshold(val_scores, y_val, 0.0)
        ensembles.append(
            {
                "name": f"score_avg_ensemble_top{top_k}",
                "members": members,
                "threshold": tuned["threshold"],
                "pred": tuned["pred"],
                "scores": val_scores,
            }
        )
    return ensembles


def save_confusion_matrix(y_true, y_pred, output_path):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["non-rumor", "rumor"],
        yticklabels=["non-rumor", "rumor"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_outputs(best_bundle, best_pred, comparison, train_df, val_df, args):
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.models_dir / "best_model.joblib"
    joblib.dump(best_bundle, model_path)

    comparison_df = pd.DataFrame(comparison).sort_values(
        ["accuracy", "f1"], ascending=False
    )
    comparison_df.to_csv(args.results_dir / "model_comparison.csv", index=False)

    y_val = val_df["label"].to_numpy()
    report = classification_report(
        y_val,
        best_pred,
        labels=[0, 1],
        target_names=["non-rumor", "rumor"],
        digits=4,
    )
    (args.results_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(y_val, best_pred, args.results_dir / "confusion_matrix.png")

    best_metrics = metric_row(
        best_bundle["name"],
        best_bundle["variant"],
        best_pred,
        y_val,
        best_bundle.get("threshold"),
        best_bundle.get("input_mode", "text"),
        True,
    )
    metrics = {
        "best_model": best_bundle["name"],
        "best_variant": best_bundle["variant"],
        "model_path": str(model_path),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "label_distribution": {
            "train": train_df["label"].value_counts().sort_index().astype(int).to_dict(),
            "val": val_df["label"].value_counts().sort_index().astype(int).to_dict(),
        },
        "metrics": {
            key: value
            for key, value in best_metrics.items()
            if key in {"accuracy", "precision", "recall", "f1", "threshold", "tn", "fp", "fn", "tp"}
        },
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train_and_evaluate(args):
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    comparison = [
        metric_row("majority_class", "rule", majority_prediction(train_df, val_df), y_val),
        metric_row(
            "event_majority",
            "rule",
            event_majority_prediction(train_df, val_df),
            y_val,
            input_mode="frame",
            deployable=False,
        ),
    ]
    fitted_models = []

    for spec in model_specs():
        model = Pipeline([("features", spec["features"]), ("clf", spec["clf"])])
        X_train = input_for(train_df, spec["input"])
        X_val = input_for(val_df, spec["input"])
        model.fit(X_train, y_train)

        raw_pred = model.predict(X_val)
        comparison.append(
            metric_row(
                spec["name"],
                "raw",
                raw_pred,
                y_val,
                input_mode=spec["input"],
                deployable=spec["input"] == "text",
            )
        )

        scores, score_method = scores_for(model, X_val)
        tuned_accuracy = accuracy_score(y_val, raw_pred)
        tuned_pred = raw_pred
        tuned_threshold = None
        if scores is not None:
            default_threshold = 0.5 if score_method == "predict_proba" else 0.0
            tuned = best_threshold(scores, y_val, default_threshold)
            tuned_pred = tuned["pred"]
            tuned_threshold = tuned["threshold"]
            tuned_accuracy = tuned["accuracy"]
            comparison.append(
                metric_row(
                    spec["name"],
                    "val_threshold",
                    tuned_pred,
                    y_val,
                    tuned_threshold,
                    spec["input"],
                    spec["input"] == "text",
                )
            )

        fitted_models.append(
            {
                "name": spec["name"],
                "model": model,
                "input_mode": spec["input"],
                "score_method": score_method,
                "tuned_accuracy": tuned_accuracy,
                "tuned_pred": tuned_pred,
                "threshold": tuned_threshold,
            }
        )
        print(f"{spec['name']}: raw_acc={accuracy_score(y_val, raw_pred):.4f}, best_acc={tuned_accuracy:.4f}")

    ensembles = build_ensembles(fitted_models, train_df, val_df, y_val)
    for ensemble in ensembles:
        comparison.append(
            metric_row(
                ensemble["name"],
                "val_threshold",
                ensemble["pred"],
                y_val,
                ensemble["threshold"],
            )
        )
        print(f"{ensemble['name']}: best_acc={accuracy_score(y_val, ensemble['pred']):.4f}")

    candidates = []
    for item in fitted_models:
        if item["input_mode"] != "text":
            continue
        candidates.append(
            {
                "name": item["name"],
                "variant": "val_threshold" if item["threshold"] is not None else "raw",
                "pred": item["tuned_pred"],
                "accuracy": accuracy_score(y_val, item["tuned_pred"]),
                "bundle": {
                    "kind": "single",
                    "name": item["name"],
                    "variant": "val_threshold" if item["threshold"] is not None else "raw",
                    "input_mode": "text",
                    "pipeline": item["model"],
                    "score_method": item["score_method"],
                    "threshold": item["threshold"],
                },
            }
        )
    for ensemble in ensembles:
        candidates.append(
            {
                "name": ensemble["name"],
                "variant": "val_threshold",
                "pred": ensemble["pred"],
                "accuracy": accuracy_score(y_val, ensemble["pred"]),
                "bundle": {
                    "kind": "ensemble",
                    "name": ensemble["name"],
                    "variant": "val_threshold",
                    "input_mode": "text",
                    "members": ensemble["members"],
                    "threshold": ensemble["threshold"],
                    "score_method": "normalized_score_average",
                },
            }
        )

    best = max(candidates, key=lambda item: (item["accuracy"], item["bundle"]["name"]))
    save_outputs(best["bundle"], best["pred"], comparison, train_df, val_df, args)
    print(f"Best deployable model: {best['name']} ({best['variant']}) acc={best['accuracy']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate rumor detection models.")
    parser.add_argument("--train", type=Path, default=Path("data/split/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/split/val.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
