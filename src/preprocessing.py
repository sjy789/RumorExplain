import html
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")


def normalize_tweet(text):
    text = html.unescape(str(text))
    text = URL_RE.sub(" URLTOKEN ", text)
    text = MENTION_RE.sub(" USERTOKEN ", text)
    return text.lower()


def clean_for_word(text):
    text = normalize_tweet(text).replace("'", "")
    text = re.sub(r"[^a-z0-9#_\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_text_series(values):
    if isinstance(values, pd.DataFrame):
        if "text" in values.columns:
            return values["text"].astype(str)
        return values.iloc[:, 0].astype(str)
    if isinstance(values, pd.Series):
        return values.astype(str)
    array = np.asarray(values)
    if array.ndim > 1:
        array = array[:, 0]
    return pd.Series(array).astype(str)


class TweetStatsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text = to_text_series(X)
        alpha_counts = text.apply(lambda value: max(1, sum(ch.isalpha() for ch in value)))
        return np.column_stack(
            [
                text.str.len(),
                text.str.split().str.len(),
                text.str.count(URL_RE),
                text.str.count(MENTION_RE),
                text.str.count(HASHTAG_RE),
                text.str.count("!"),
                text.str.count(r"\?"),
                text.str.count(r"\d"),
                text.str.lower().str.startswith("rt ").astype(float),
                text.apply(lambda value: sum(ch.isupper() for ch in value)) / alpha_counts,
            ]
        ).astype(float)
