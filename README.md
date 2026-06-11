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

The classifier always determines the label. The explanation module sends the
fixed label and model evidence to the SJTU large-language-model API, which
turns that evidence into a short Chinese explanation. The LLM cannot change
the classification result.

Apply for an API key through the SJTU service, connect to the campus network
or SJTU VPN, and set these variables in the same terminal used to run Python:

```powershell
$env:SJTU_API_KEY="your_api_key"
$env:SJTU_BASE_URL="https://models.sjtu.edu.cn/api/v1"
$env:SJTU_LLM_MODEL="deepseek-chat"
```

Do not write the real API key into source code or commit it to GitHub. Check
the API connection with:

```bash
python src/llm_explainer.py
```

Run classification and explanation with:

```bash
python src/predict.py "Breaking news example text"
```

Example output:

```text
检测结果：1（谣言）
判断依据：该文本包含……，这些语言特征与模型判断方向一致……
```

If the API is unavailable, prediction still works and automatically uses a
local evidence-based explanation. To force offline explanation:

```bash
python src/predict.py --no-llm "Breaking news example text"
```

Python usage:

```python
from src.predict import RumourDetectClass

detector = RumourDetectClass("models/best_model.joblib")
label = detector.classify("Breaking news example text")
explanation = detector.explain("Breaking news example text")
result = detector.predict_with_explanation("Breaking news example text")
```

`result` contains `label`, `explanation`, and `explanation_source`. When the
SJTU API is used, `explanation_source` records the actual model name, such as
`deepseek-chat`; otherwise it is `local_fallback`. API responses are cached
under `.cache/` to reduce repeated requests. The client also respects the
configured limit of 10 requests per minute.
