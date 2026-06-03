# Rumor Detection Model Training

This repository contains the model training and evaluation code for the rumor
detection part of the AI introduction course project.

## Data

The dataset is placed under:

```text
data/split/train.csv
data/split/val.csv
```

Each file contains `id,text,label,event`. Label `0` means non-rumor and label
`1` means rumor.

## Train And Evaluate

```bash
pip install -r requirements.txt
python src/train_models.py
```

The script compares multiple classifiers, tunes decision thresholds on
`val.csv`, tries score-average ensembles, and writes:

```text
models/best_model.joblib
results/metrics.json
results/classification_report.txt
results/model_comparison.csv
results/confusion_matrix.png
```

## Predict

```bash
python src/predict.py "Breaking news example text"
```

The `RumourDetectClass` class in `src/predict.py` exposes:

```python
detector = RumourDetectClass("models/best_model.joblib")
label = detector.classify(text)
reason = detector.explain(text)
```
