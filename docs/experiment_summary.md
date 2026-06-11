# 实验结果摘要

最终提交的训练流程使用已经确定的模型和参数，不会在运行时重复进行大规模搜索。但是我们在实验阶段一共完成了 274 种模型、参数、阈值与融合组合。下表从主要模型类别中选取具有代表性的结果进行展示，并非完整实验记录，便于更加清晰地展示：

| Model family | Representative validation accuracy |
|---|---:|
| Word TF-IDF + LogisticRegression | 0.8279 |
| Word/character TF-IDF + LinearSVC | 0.8728 |
| NB-SVM | 0.8703 |
| Five-fold OOF stacking | 0.8703 |
| MPNet embeddings + LogisticRegression | 0.8404 |
| BERTweet mean embeddings + LinearSVC | 0.8379 |
| SetFit | 0.8504 |
| Fine-tuned BERTweet | 0.8429 |
| Fine-tuned DeBERTa-v3-small | 0.8204 |
| Fixed TF-IDF/BERTweet/MPNet fusion | **0.8953** |

最终选择的融合方案权重分别为 `0.60`、`0.20` 和 `0.20`，分类阈值固定为 `0.03`。
