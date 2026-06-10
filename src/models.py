from __future__ import annotations

import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

from src.preprocessing import clean_for_word, normalize_tweet


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardizer(scores):
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return mean, std if std > 1e-12 else 1.0


def normalize_scores(scores, stats):
    return (np.asarray(scores, dtype=float) - stats[0]) / stats[1]


def estimator_scores(model, features):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    return np.asarray(model.predict_proba(features), dtype=float)[:, 1]


def make_word_vectorizer(stop_words=None, max_features=30000):
    return TfidfVectorizer(
        preprocessor=clean_for_word,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
        stop_words=stop_words,
    )


def make_char_vectorizer():
    return TfidfVectorizer(
        preprocessor=normalize_tweet,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
    )


def make_word_char_svc(c_value, stop_words=None):
    features = FeatureUnion(
        [
            ("word", make_word_vectorizer(stop_words=stop_words)),
            ("char", make_char_vectorizer()),
        ]
    )
    classifier = LinearSVC(
        C=c_value,
        class_weight="balanced",
        max_iter=8000,
        random_state=42,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def train_word_logistic_regression(texts, labels):
    return Pipeline(
        [
            (
                "features",
                make_word_vectorizer(stop_words="english", max_features=20000),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    ).fit(texts, labels)


def train_tfidf_ensemble(texts, labels):
    specifications = [
        ("word_char_svc_c08", 0.8, None),
        ("word_char_svc_c10", 1.0, None),
        ("word_char_svc_stop_c05", 0.5, "english"),
    ]
    members = []
    for name, c_value, stop_words in specifications:
        model = make_word_char_svc(c_value, stop_words).fit(texts, labels)
        scores = estimator_scores(model, texts)
        members.append(
            {
                "name": name,
                "model": model,
                "stats": standardizer(scores),
            }
        )
    return {"kind": "tfidf_ensemble", "members": members}


def score_tfidf_ensemble(component, texts):
    member_scores = [
        normalize_scores(estimator_scores(member["model"], texts), member["stats"])
        for member in component["members"]
    ]
    return np.vstack(member_scores).mean(axis=0)


def _texts_digest(texts):
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_embedding_cache(path, texts):
    if path is None or not Path(path).exists():
        return None
    cached = np.load(path, allow_pickle=False)
    digest = str(cached["digest"].item())
    if digest != _texts_digest(texts):
        return None
    return cached["embeddings"]


def _save_embedding_cache(path, texts, embeddings):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings),
        digest=np.asarray(_texts_digest(texts)),
    )


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def encode_bertweet(
    texts,
    config,
    cache_path=None,
    device=None,
    runtime=None,
):
    texts = list(map(str, texts))
    cached = _load_embedding_cache(cache_path, texts)
    if cached is not None:
        return cached

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = runtime if runtime is not None else {}
    key = ("bertweet", config["model_name"], str(device))
    if key not in runtime:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            normalization=True,
        )
        model = AutoModel.from_pretrained(config["model_name"]).to(device)
        model.eval()
        runtime[key] = tokenizer, model
    tokenizer, model = runtime[key]

    batches = []
    batch_size = int(config["batch_size"])
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=int(config["max_length"]),
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            batches.append(
                _mean_pool(hidden, encoded["attention_mask"]).cpu().numpy()
            )

    embeddings = np.vstack(batches)
    embeddings /= np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
    )
    _save_embedding_cache(cache_path, texts, embeddings)
    if runtime is None:
        del model
        gc.collect()
    return embeddings


def encode_mpnet(
    texts,
    config,
    cache_path=None,
    device=None,
    runtime=None,
):
    texts = list(map(str, texts))
    cached = _load_embedding_cache(cache_path, texts)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = runtime if runtime is not None else {}
    key = ("mpnet", config["model_name"], str(device))
    if key not in runtime:
        runtime[key] = SentenceTransformer(config["model_name"], device=str(device))
    model = runtime[key]
    embeddings = model.encode(
        texts,
        batch_size=int(config["batch_size"]),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    _save_embedding_cache(cache_path, texts, embeddings)
    if runtime is None:
        del model
        gc.collect()
    return embeddings


def train_model_bundle(train_frame, config, cache_dir):
    texts = train_frame["text"]
    labels = train_frame["label"].to_numpy()
    cache_dir = Path(cache_dir)

    word_lr = train_word_logistic_regression(texts, labels)
    tfidf = train_tfidf_ensemble(texts, labels)
    tfidf_train_scores = score_tfidf_ensemble(tfidf, texts)
    tfidf["stats"] = standardizer(tfidf_train_scores)

    bertweet_embeddings = encode_bertweet(
        texts,
        config["bertweet"],
        cache_dir / "bertweet_train.npz",
    )
    bertweet_classifier = LinearSVC(
        C=float(config["bertweet"]["classifier_c"]),
        max_iter=8000,
        random_state=int(config["seed"]),
    ).fit(bertweet_embeddings, labels)
    bertweet_scores = estimator_scores(bertweet_classifier, bertweet_embeddings)

    mpnet_embeddings = encode_mpnet(
        texts,
        config["mpnet"],
        cache_dir / "mpnet_train.npz",
    )
    mpnet_classifier = LogisticRegression(
        C=float(config["mpnet"]["classifier_c"]),
        max_iter=5000,
        solver="liblinear",
        random_state=int(config["seed"]),
    ).fit(mpnet_embeddings, labels)
    mpnet_scores = estimator_scores(mpnet_classifier, mpnet_embeddings)

    return {
        "format_version": 1,
        "name": "tfidf_bertweet_mpnet_fusion",
        "seed": int(config["seed"]),
        "threshold": float(config["threshold"]),
        "weights": config["weights"],
        "baseline": {"word_lr": word_lr},
        "components": {
            "tfidf": tfidf,
            "bertweet": {
                "kind": "bertweet",
                "config": config["bertweet"],
                "classifier": bertweet_classifier,
                "stats": standardizer(bertweet_scores),
            },
            "mpnet": {
                "kind": "mpnet",
                "config": config["mpnet"],
                "classifier": mpnet_classifier,
                "stats": standardizer(mpnet_scores),
            },
        },
    }


def score_models(bundle, texts, cache_dir=None, cache_key=None, runtime=None):
    texts = list(map(str, texts))
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    runtime = runtime if runtime is not None else {}

    tfidf_component = bundle["components"]["tfidf"]
    tfidf_scores = score_tfidf_ensemble(tfidf_component, texts)

    bertweet_component = bundle["components"]["bertweet"]
    bertweet_cache = (
        cache_dir / f"bertweet_{cache_key}.npz"
        if cache_dir is not None and cache_key
        else None
    )
    bertweet_embeddings = encode_bertweet(
        texts,
        bertweet_component["config"],
        bertweet_cache,
        runtime=runtime,
    )
    bertweet_scores = estimator_scores(
        bertweet_component["classifier"],
        bertweet_embeddings,
    )

    mpnet_component = bundle["components"]["mpnet"]
    mpnet_cache = (
        cache_dir / f"mpnet_{cache_key}.npz"
        if cache_dir is not None and cache_key
        else None
    )
    mpnet_embeddings = encode_mpnet(
        texts,
        mpnet_component["config"],
        mpnet_cache,
        runtime=runtime,
    )
    mpnet_scores = estimator_scores(mpnet_component["classifier"], mpnet_embeddings)

    component_scores = {
        "tfidf": tfidf_scores,
        "bertweet": bertweet_scores,
        "mpnet": mpnet_scores,
    }
    fusion_scores = np.zeros(len(texts), dtype=float)
    for name, weight in bundle["weights"].items():
        fusion_scores += float(weight) * normalize_scores(
            component_scores[name],
            bundle["components"][name]["stats"],
        )

    return {
        "word_tfidf_logistic_regression": estimator_scores(
            bundle["baseline"]["word_lr"],
            texts,
        ),
        "word_char_tfidf_linear_svc": estimator_scores(
            tfidf_component["members"][0]["model"],
            texts,
        ),
        "tfidf_ensemble": tfidf_scores,
        "bertweet_linear_svc": bertweet_scores,
        "mpnet_logistic_regression": mpnet_scores,
        "final_weighted_fusion": fusion_scores,
    }
