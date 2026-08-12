# Multi-Label Toxic Comment Classification

Multi-label toxicity classification of Wikipedia talk-page comments, comparing a classical
TF-IDF baseline against fine-tuned transformer models, followed by per-label threshold
calibration and attention-based interpretability analysis.

Each comment may carry any combination of six independent labels:
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
— 159,571 Wikipedia talk-page comments, each labelled independently by human raters.

The dataset is severely imbalanced: `threat` and `identity_hate` appear in well under 1% of
comments, which drives most of the methodological decisions below.

## Results

Mean column-wise ROC AUC is the competition metric. F1 is reported at the default 0.5 cut-off.

| Model | Mean ROC AUC | Micro F1 | Macro F1 |
|---|---|---|---|
| TF-IDF + One-vs-Rest Logistic Regression | 0.9772 | 0.7036 | 0.5860 |
| **BERT** (`bert-base-uncased`) | **0.9895** | **0.7230** | **0.6266** |
| XLM-RoBERTa (`xlm-roberta-base`) | 0.9823 | 0.6486 | 0.5058 |

BERT was the stronger model. XLM-RoBERTa underperformed on the rare labels in particular,
where its precision collapsed (`threat` F1 = 0.2249).

### Threshold calibration

AUC stayed above 0.98 for every label while F1 fell as low as 0.42 on the rare ones — a
threshold artefact rather than a ranking failure. Thresholds were tuned per label on the
**validation** set only, then applied unchanged to the test set.

| Label | Support | AUC | F1 @ 0.5 | Threshold | F1 tuned | Gain |
|---|---|---|---|---|---|---|
| toxic | 2296 | 0.9868 | 0.7972 | 0.94 | 0.8320 | +0.0347 |
| severe_toxic | 233 | 0.9923 | 0.4226 | 0.94 | 0.5290 | +0.1064 |
| obscene | 1271 | 0.9934 | 0.7788 | 0.95 | 0.8491 | +0.0703 |
| threat | 84 | 0.9829 | 0.5500 | 0.92 | 0.5591 | +0.0091 |
| insult | 1204 | 0.9895 | 0.7141 | 0.94 | 0.7837 | +0.0696 |
| identity_hate | 208 | 0.9891 | 0.5038 | 0.92 | 0.6065 | +0.1027 |

Aggregate effect: micro F1 **0.7260 → 0.7921**, macro F1 **0.6277 → 0.6932**. The gains are
largest on exactly the rare labels the fixed cut-off handled worst.

### Interpretability

Token-level attributions are taken from final-layer attention directed at the `[CLS]`
position, averaged over heads. Attention shows where the model looked, not a causal account
of why it decided — it is a widely used but contested proxy, and is reported as indicative
only.

## Repository contents

| File | Description |
|---|---|
| `updated-fyp-nb1.ipynb` | Data quality checks, visualisation, TF-IDF + OvR logistic regression baseline |
| `updated-fyp-nb2 (1).ipynb` | BERT and XLM-RoBERTa fine-tuning, comparison against the baseline |
| `updated-fyp-nb3 (2).ipynb` | Per-label threshold optimisation and attention-based interpretability |
| `project-report.docx` | Full written report |

## Running the notebooks

The notebooks were written for and executed in Kaggle Notebooks, and locate their input file
automatically under `/kaggle/input/`.

1. `+ Add Input` → add `jigsaw-toxic-comment-classification-challenge`
2. Settings → Accelerator → **GPU**, and Internet → **On**
3. Run All

Notebook 2 takes roughly 2–4 hours for both models on the full dataset; its `SAMPLE_SIZE`
constant caps the training data if you want a faster run. Notebook 3 retrains BERT briefly so
that it is self-contained, and takes 40–60 minutes.

A fixed random seed of 42 controls all partitioning and initialisation, and every split is
stratified so class representation stays consistent across partitions.

### Implementation notes

- **Multi-label, not multi-class** — `BCEWithLogitsLoss`, sigmoid + threshold rather than
  softmax/argmax, `problem_type="multi_label_classification"`, and float label vectors.
- The Transformers library renamed `evaluation_strategy` to `eval_strategy` between versions;
  the notebooks detect at runtime which name the installed version expects.
- Interpretability requires `attn_implementation="eager"` at model instantiation — the
  default attention backend does not return attention weights.

## Stack

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · matplotlib · seaborn
