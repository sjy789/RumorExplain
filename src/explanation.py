from __future__ import annotations

import numpy as np


COMPONENT_NAMES = {
    "tfidf": "TF-IDF文本特征模型",
    "bertweet": "BERTweet语义模型",
    "mpnet": "MPNet语义模型",
}


def _clean_feature_name(name):
    term = name.split("__", 1)[-1]
    replacements = {
        "urltoken": "[链接]",
        "usertoken": "[用户提及]",
    }
    return replacements.get(term, term).strip()


def extract_tfidf_terms(bundle, text, label, top_k=8):
    aggregate = {}
    members = bundle["components"]["tfidf"]["members"]
    for member in members:
        pipeline = member["model"]
        features = pipeline.named_steps["features"]
        classifier = pipeline.named_steps["classifier"]
        matrix = features.transform([text]).tocsr()
        names = features.get_feature_names_out()

        for index, value in zip(matrix.indices, matrix.data):
            name = names[index]
            if not name.startswith("word__"):
                continue
            contribution = float(value * classifier.coef_[0, index])
            term = _clean_feature_name(name)
            if len(term) < 2:
                continue
            aggregate[term] = aggregate.get(term, 0.0) + contribution

    direction = 1.0 if label == 1 else -1.0
    aligned = {
        term: contribution * direction
        for term, contribution in aggregate.items()
    }
    supporting = sorted(
        ((term, value) for term, value in aligned.items() if value > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    opposing = sorted(
        ((term, -value) for term, value in aligned.items() if value < 0),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "supporting_terms": [term for term, _ in supporting[:top_k]],
        "opposing_terms": [term for term, _ in opposing[: max(3, top_k // 2)]],
    }


def _margin_level(score, threshold):
    margin = abs(float(score) - float(threshold))
    if margin < 0.25:
        return "较弱"
    if margin < 0.75:
        return "中等"
    return "较强"


def build_evidence(bundle, text, result):
    label = int(result["label"])
    terms = extract_tfidf_terms(bundle, text, label)
    components = []
    for name, contribution in result["contributions"].items():
        components.append(
            {
                "name": COMPONENT_NAMES[name],
                "direction": "支持谣言" if contribution >= 0 else "支持非谣言",
                "strength": round(abs(float(contribution)), 4),
            }
        )

    return {
        "text": str(text),
        "label": label,
        "label_name": "谣言" if label == 1 else "非谣言",
        "decision_strength": _margin_level(result["score"], result["threshold"]),
        "components": components,
        **terms,
    }


def fallback_explanation(evidence):
    label_name = evidence["label_name"]
    supporting = evidence["supporting_terms"]
    agreeing = [
        item["name"]
        for item in evidence["components"]
        if (item["direction"] == "支持谣言") == (evidence["label"] == 1)
    ]

    parts = [f"模型将该文本判断为{label_name}。"]
    if supporting:
        terms = "、".join(f"“{term}”" for term in supporting[:5])
        parts.append(f"文本中影响较大的判别特征包括{terms}。")
    if agreeing:
        parts.append(f"{'、'.join(agreeing)}的判断方向与最终结果一致。")
    else:
        parts.append("各子模型存在一定分歧，最终结果由加权融合得出。")
    parts.append("该依据反映模型学习到的语言模式，不等同于外部事实核验。")
    return "".join(parts)
