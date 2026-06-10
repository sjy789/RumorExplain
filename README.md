# Rumor Detection Model

This project trains and evaluates the rumor classification module for the AI
introduction course project. The submitted workflow contains fixed,
reproducible model settings rather than the exploratory parameter search.

## Data

Place the supplied split files at:

```text
data/split/train.csv
data/split/val.csv
```

Both files must contain `id`, `text`, `label`, and `event`. Label `0` means
non-rumor and label `1` means rumor.

## Model

The final classifier combines three normalized scores:

```text
0.60 * word/character TF-IDF ensemble
0.20 * frozen BERTweet mean-pooled features + LinearSVC
0.20 * MPNet sentence embeddings + LogisticRegression
```

The decision threshold is fixed at `0.03`. All fixed parameters are recorded
in `configs/model_config.json`.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Training uses only `train.csv` and saves:

```text
models/best_model.joblib
```

The first run downloads BERTweet and MPNet. Generated embedding caches are
stored under `models/cache/`.

## Evaluate

```bash
python src/evaluate.py
```

Evaluation reads the fixed model and evaluates it once on `val.csv`. It also
compares representative TF-IDF, BERTweet, MPNet, and fusion models, then
generates:

```text
results/metrics.json
results/classification_report.txt
results/model_comparison.csv
results/confusion_matrix.png
```

Expected final validation accuracy with the supplied split is approximately
`0.8953` (`359/401`).

The exploratory runs used to choose the fixed architecture are summarized in
`docs/experiment_summary.md`. This document is not used by training or
evaluation.

## Predict

```bash
python src/predict.py "Breaking news example text"
```

Python usage:

```python
from src.predict import RumourDetectClass

detector = RumourDetectClass("models/best_model.joblib")
label = detector.classify("Breaking news example text")
explanation = detector.explain("Breaking news example text")
```
