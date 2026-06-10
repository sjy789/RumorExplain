from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import normalize_scores, score_models


class RumourDetectClass:
    def __init__(self, model_path="models/best_model.joblib"):
        self.bundle = joblib.load(model_path)
        self.runtime = {}
        self._last_text = None
        self._last_result = None

    def score(self, text):
        return self.details(text)["score"]

    def classify(self, text):
        return self.details(text)["label"]

    def details(self, text):
        if text == self._last_text:
            return self._last_result

        scores = score_models(self.bundle, [text], runtime=self.runtime)
        contributions = {}
        for name, weight in self.bundle["weights"].items():
            component_score = scores[
                {
                    "tfidf": "tfidf_ensemble",
                    "bertweet": "bertweet_linear_svc",
                    "mpnet": "mpnet_logistic_regression",
                }[name]
            ]
            contributions[name] = float(
                weight
                * normalize_scores(
                    component_score,
                    self.bundle["components"][name]["stats"],
                )[0]
            )
        final_score = float(sum(contributions.values()))
        result = {
            "label": int(final_score >= self.bundle["threshold"]),
            "score": final_score,
            "threshold": float(self.bundle["threshold"]),
            "contributions": contributions,
        }
        self._last_text = text
        self._last_result = result
        return result

    def explain(self, text):
        result = self.details(text)
        label = "rumor" if result["label"] == 1 else "non-rumor"
        strongest = max(
            result["contributions"],
            key=lambda name: abs(result["contributions"][name]),
        )
        return (
            f"The model predicts {label} with fusion score {result['score']:.4f} "
            f"(threshold {result['threshold']:.2f}). The strongest component "
            f"contribution came from {strongest}."
        )


def main():
    parser = argparse.ArgumentParser(description="Classify one text.")
    parser.add_argument("text")
    parser.add_argument(
        "--model",
        default="models/best_model.joblib",
    )
    args = parser.parse_args()

    detector = RumourDetectClass(args.model)
    result = detector.details(args.text)
    print(result["label"])
    print(detector.explain(args.text))


if __name__ == "__main__":
    main()
