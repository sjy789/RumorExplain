# Experiment Summary

The final submitted workflow is fixed and does not repeat this search. 
The complete exploration covered 274 model, parameter, threshold, and fusion
combinations. 
The table below selects representative results from the main model families rather than listing every experiment:

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

The selected fusion uses weights `0.60`, `0.20`, and `0.20`.
The final decisionthreshold is fixed at `0.03`.
