from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import normalize_scores, score_models
from src.explanation import build_evidence, fallback_explanation
from src.llm_explainer import LLMExplainer, LLMExplanationError


class RumourDetectClass:
    def __init__(self, model_path="models/best_model.joblib"):
        self.bundle = joblib.load(model_path)
        self.runtime = {}
        self._last_text = None
        self._last_result = None
        self._llm_explainers = {}

    def score(self, text):
        return self.details(text)["score"]

    def classify(self, text):
        return self.details(text)["label"]

    def evidence(self, text):
        return build_evidence(self.bundle, text, self.details(text))

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

    def explain_result(
        self,
        text,
        use_llm=True,
        refresh=False,
        llm_config="configs/llm_config.json",
    ):
        evidence = self.evidence(text)
        if use_llm:
            try:
                if llm_config not in self._llm_explainers:
                    self._llm_explainers[llm_config] = LLMExplainer.from_env(
                        llm_config
                    )
                explainer = self._llm_explainers[llm_config]
                explanation = explainer.explain(
                    evidence,
                    refresh=refresh,
                )
                return explanation, explainer.model
            except LLMExplanationError:
                pass
        return fallback_explanation(evidence), "local_fallback"

    def explain(
        self,
        text,
        use_llm=True,
        refresh=False,
        llm_config="configs/llm_config.json",
    ):
        explanation, _ = self.explain_result(
            text,
            use_llm=use_llm,
            refresh=refresh,
            llm_config=llm_config,
        )
        return explanation

    def predict_with_explanation(
        self,
        text,
        use_llm=True,
        refresh=False,
        llm_config="configs/llm_config.json",
    ):
        result = self.details(text)
        explanation, source = self.explain_result(
            text,
            use_llm=use_llm,
            refresh=refresh,
            llm_config=llm_config,
        )
        return {
            "label": result["label"],
            "explanation": explanation,
            "explanation_source": source,
        }


def main():
    parser = argparse.ArgumentParser(description="Classify one text.")
    parser.add_argument("text")
    parser.add_argument(
        "--model",
        default="models/best_model.joblib",
    )
    parser.add_argument(
        "--llm-config",
        default="configs/llm_config.json",
    )
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--refresh-explanation", action="store_true")
    parser.add_argument("--show-source", action="store_true")
    args = parser.parse_args()

    detector = RumourDetectClass(args.model)
    result = detector.predict_with_explanation(
        args.text,
        use_llm=not args.no_llm,
        refresh=args.refresh_explanation,
        llm_config=args.llm_config,
    )
    label_name = "谣言" if result["label"] == 1 else "非谣言"
    print(f"检测结果：{result['label']}（{label_name}）")
    print(f"判断依据：{result['explanation']}")
    if args.show_source:
        print(f"解释来源：{result['explanation_source']}")


if __name__ == "__main__":
    main()
