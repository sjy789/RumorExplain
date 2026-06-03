from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _as_text_input(texts):
    if isinstance(texts, str):
        return pd.Series([texts])
    return pd.Series(list(texts)).astype(str)


def _scores_for(model, X, method):
    if method == "decision_function":
        return np.asarray(model.decision_function(X))
    if method == "predict_proba":
        return np.asarray(model.predict_proba(X))[:, 1]
    raise ValueError(f"Unsupported score method: {method}")


def predict_bundle(bundle, texts):
    X = _as_text_input(texts)
    if bundle["kind"] == "single":
        if bundle.get("threshold") is None:
            return np.asarray(bundle["pipeline"].predict(X)).astype(int)
        scores = _scores_for(bundle["pipeline"], X, bundle["score_method"])
        return (scores >= bundle["threshold"]).astype(int)

    normalized_scores = []
    for member in bundle["members"]:
        scores = _scores_for(member["model"], X, member["score_method"])
        normalized_scores.append((scores - member["mean"]) / member["std"])
    scores = np.vstack(normalized_scores).mean(axis=0)
    return (scores >= bundle["threshold"]).astype(int)


def _linear_terms_from_pipeline(pipeline, text, label, top_k):
    features = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_") or not hasattr(features, "get_feature_names_out"):
        return []

    X = features.transform(pd.Series([text]))
    coef = clf.coef_[0]
    contrib = X.multiply(coef).toarray().ravel()
    if label == 1:
        candidate_idx = np.where(contrib > 0)[0]
        ranked = candidate_idx[np.argsort(contrib[candidate_idx])[::-1]]
    else:
        candidate_idx = np.where(contrib < 0)[0]
        ranked = candidate_idx[np.argsort(contrib[candidate_idx])]

    names = features.get_feature_names_out()
    terms = []
    for idx in ranked[:top_k]:
        term = names[idx].replace("word__", "").replace("char__", "")
        term = term.replace("URLTOKEN", "url").replace("USERTOKEN", "user")
        if term.strip() and term not in terms:
            terms.append(term)
    return terms


class RumourDetectClass:
    def __init__(self, model_path="models/best_model.joblib"):
        self.bundle = joblib.load(model_path)

    def classify(self, text: str) -> int:
        return int(predict_bundle(self.bundle, text)[0])

    def explain(self, text: str, top_k: int = 8) -> str:
        label = self.classify(text)
        pipeline = None
        if self.bundle["kind"] == "single":
            pipeline = self.bundle["pipeline"]
        elif self.bundle.get("members"):
            pipeline = self.bundle["members"][0]["model"]

        terms = _linear_terms_from_pipeline(pipeline, text, label, top_k) if pipeline else []
        label_text = "rumor" if label == 1 else "non-rumor"
        if not terms:
            return (
                f"The model predicts {label_text}. The decision is based on "
                "the combined TF-IDF word and character n-gram score."
            )
        return f"The model predicts {label_text}. Main supporting features: {', '.join(terms)}."


def main():
    parser = argparse.ArgumentParser(description="Classify one rumor-detection text.")
    parser.add_argument("text")
    parser.add_argument("--model", default="models/best_model.joblib")
    args = parser.parse_args()

    detector = RumourDetectClass(args.model)
    label = detector.classify(args.text)
    print(label)
    print(detector.explain(args.text))


if __name__ == "__main__":
    main()
