# sequence-models-spam`r`n`r`nDetecting spam models

## Overview

A small, reproducible comparison of TF-IDF + Logistic Regression, a bidirectional RNN, a bidirectional LSTM, and a Transformer encoder on the SMS Spam Collection. Implementation is intentionally direct so the data flow, masking, training loop, and checkpoint selection are easy to inspect.

## Data and split

The experiment reads the UCI SMS Spam Collection file with `encoding="latin-1"`. It was donated by Almeida and Hidalgo to the [UCI Machine Learning Repository](https://doi.org/10.24432/C5CC84), which provides the dataset under CC BY 4.0. If absent, `experiment.py` downloads the public UCI ZIP. The executed parser loaded 5,572 records: 4,825 ham and 747 spam. The split produced 3,900 training, 836 validation, and 836 test records. The raw file has 5,574 LF characters because one message contains two embedded line breaks; the tab-delimited parser treats that body as one record. It expects the tab-separated label/text format and maps ham to 0 and spam to 1. The split is stratified 70/15/15 with `random_state=42`. A lowercase `[a-z0-9]+` regex tokenizer is used. Vocabulary counts are computed from training messages only; vocabulary size is capped at 8,000 (including PAD/UNK), and messages are truncated or padded to length 40.

## Models

The baseline is `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)` plus class-weight-balanced Logistic Regression. Neural models use PyTorch CPU, 64-dimensional embeddings, hidden size 64, masked mean pooling, Adam, batch size 64, 8 epochs, BCEWithLogitsLoss with training-set negative/positive ratio as `pos_weight`, and gradient clipping at 1.0. BiRNN and BiLSTM are bidirectional. The Transformer has learned token and position embeddings, two encoder layers, four heads, feed-forward size 128, dropout 0.1, and a key-padding mask. Each neural architecture uses seeds 0, 1, and 2. Best epoch is selected only by validation PR-AUC.

## Results

Metrics are calculated on the held-out test partition at probability threshold 0.5. Accuracy is not used as a headline metric. Mean and population standard deviation are across the three independent seed runs; the baseline is one run.

<!-- RESULTS_START -->

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9327354260089686 | 0.9369369369369369 | 0.9285714285714286 | 0.9901218429360694 | 0.973980566375551 |
| BiRNN | 0.8863952000084975 ± 0.03059158600574668 | 0.8757001846973688 ± 0.08941692574231234 | 0.9077380952380952 ± 0.04394292577450419 | 0.9843256708760851 ± 0.005364225640593288 | 0.9387492120918711 ± 0.029174835690283788 |
| BiLSTM | 0.9182037966767739 ± 0.0008776001714708933 | 0.9379835029113609 ± 0.03655204741068986 | 0.9017857142857144 ± 0.033407655239053015 | 0.984572316495659 ± 0.0041148567132432 | 0.9550253088676673 ± 0.013291833289063557 |
| Transformer | 0.899213550389347 ± 0.024680372862573418 | 0.9642287624223403 ± 0.018576246208903153 | 0.8452380952380952 ± 0.05520011009223632 | 0.9807780847145487 ± 0.006166110343695519 | 0.9463831796993002 ± 0.010842959620361662 |
<!-- RESULTS_END -->

## Findings

<!-- FINDINGS_START -->

- BiLSTM had the highest mean test PR-AUC among the neural models: 0.9550253088676673 (population standard deviation 0.013291833289063557).
- BiLSTM had the highest mean test F1 among the neural models: 0.9182037966767739 (population standard deviation 0.0008776001714708933).
- BiRNN had the largest across-seed standard deviation in test PR-AUC: 0.029174835690283788; these three runs share the same fixed split.
- The single TF-IDF + Logistic Regression baseline measured test F1 0.9327354260089686 and PR-AUC 0.973980566375551.
<!-- FINDINGS_END -->

## Reproduction: hand-written scaled dot-product attention

The attention block follows the scaled dot-product multi-head attention idea in [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/abs/1706.03762). This implementation computes Q, K, and V with linear projections, reshapes into four heads, scales QKᵀ by √16, masks padded keys, applies softmax, and combines the values. Two encoder blocks keep the remaining architecture aligned with `nn.TransformerEncoder`.

| Model | Test F1 (mean ± population std) | Test PR-AUC (mean ± population std) |
|---|---:|---:|
| Hand-written attention | 0.8976169953523513 ± 0.010058719381301814 | 0.9442024150935638 ± 0.004840488617443233 |
| `nn.TransformerEncoder` | 0.899213550389347 ± 0.024680372862573418 | 0.9463831796993002 ± 0.010842959620361662 |

**Matched:** embedding width 64, two layers, four heads, feed-forward width 128, dropout 0.1, learned positional embeddings, padding mask, masked mean pooling, same train/validation/test split, seeds 0–2, Adam/BCE training, validation PR-AUC checkpoint selection, and test metrics at 0.5.

**Different:** Q/K/V projection, head splitting, score scaling, key masking, softmax, and value aggregation are written with explicit tensor operations. PyTorch may use fused attention kernels, and parameter initialization/random-number consumption differs, so weights and outputs are not numerically identical. The measured test metrics above compare task performance, not layer-by-layer numerical equivalence.

Manual run IDs and all full-precision metrics are in `results/reproduction.json`; each seed is also tracked in MLflow.

## Limitations

Small dataset, one domain, one fixed split, class imbalance, no external validation, and not production-grade. Three seeds measure variation from initialization and training on this one split; they do not measure dataset-to-dataset uncertainty. Duplicate messages are not removed.

## How to run

Python 3.12 is used by the Docker image; the local runner was developed with Python 3.11. Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python experiment.py
pytest -q
uvicorn app:app --host 0.0.0.0 --port 8000
```

Docker commands, when Docker Engine is available:

```bash
docker build -t sequence-models-spam .
docker run -p 8000:8000 sequence-models-spam
```

Open `notebooks/sequence_models_spam.ipynb` from the repository root and run its experiment cell for the same workflow. MLflow uses `sqlite:///mlflow.db`; experiment outputs are in `results/`; model artifacts are in `artifacts/best_transformer/`. The saved Transformer is selected by highest test PR-AUC across its seeds only for the serving artifact after all runs are complete. This is not used to select the reported aggregate model performance. Its seed, metric, and run ID are written in `meta.json` / `results.json` and tagged in MLflow.

## Reproduction record

README result cells are generated from `results/results.json` and `results/summary.csv`. The final resume line is printed by `experiment.py` only after all runs complete. The CSV and JSON retain full precision. The script prints the raw run metrics and summary table to standard output. The registered artifact metadata documents that test PR-AUC was used to choose one Transformer seed for deployment, while epoch selection remains validation-only.



