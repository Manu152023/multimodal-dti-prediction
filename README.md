# Multimodal AI for Drug-Target Interaction Prediction

A multimodal deep learning pipeline that predicts drug-target binding
affinity (pKd) from molecular structure (SMILES) and protein sequence,
with a built-in explainability layer via cross-attention.

Dataset: [BindingDB for Drug-Target Interaction](https://www.kaggle.com/datasets/rajaryan2315/bindingdb-for-drug-target-interaction)
(`bindingdb_highconf_pKd_clean.csv`, ~1.59 GB), columns: `compound_iso_smiles`,
`target_sequence`, `affinity`.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This has been validated to work with `rdkit>=2023.9.1`'s modern
`rdFingerprintGenerator` API (no deprecated calls).

## 2. Verify your environment (run this first!)

Before touching the real 1.59 GB file, confirm the whole pipeline works
using small synthetic data:

```bash
python smoke_test.py
```

This generates synthetic SMILES/sequences/affinities, runs preprocessing,
builds the model, and does one full training step. Takes ~10-30 seconds.
If this passes, your environment is correctly set up.

## 3. Get the real data

1. Download `bindingdb_highconf_pKd_clean.csv` from the Kaggle link above.
2. Place it somewhere accessible, e.g. `data_raw/bindingdb_highconf_pKd_clean.csv`.

If the full file is too large to train on comfortably (1.59 GB, likely
hundreds of thousands of rows), start with a subsample using `--sample_frac`
(see below) to validate the full pipeline before committing to a full run.

## 4. Preprocess

```bash
python data_preprocessing.py \
    --csv data_raw/bindingdb_highconf_pKd_clean.csv \
    --out data/ \
    --sample_frac 0.2   # optional: use 20% of rows for a faster first pass
```

This cleans the data (drops invalid SMILES, deduplicates, filters
implausible affinity values), converts each SMILES to a 1024-bit Morgan
fingerprint and each protein sequence to a fixed-length integer encoding,
and saves `train.pkl` / `val.pkl` / `test.pkl` (70/15/15 split) to `data/`.

## 5. Train

```bash
python train.py --data_dir data/ --epochs 30 --batch_size 64 --lr 1e-3
```

Trains the multimodal model with early stopping (patience=7 by default)
and saves the best checkpoint (by validation RMSE) to `checkpoints/best_model.pt`.
Uses GPU automatically if available (`torch.cuda.is_available()`).

## 6. Evaluate

```bash
python evaluate.py --data_dir data/ --checkpoint checkpoints/best_model.pt
```

Reports RMSE, MSE, MAE, Pearson correlation, Spearman correlation, and the
**Concordance Index (CI)** — the standard ranking metric used in DTI
benchmarks (DeepDTA, GraphDTA, etc.) so your results are directly
comparable to prior published work. Also saves a predicted-vs-actual
scatter plot to `results/pred_vs_actual.png`.

## 7. Explainability case studies

```bash
python explainability.py \
    --checkpoint checkpoints/best_model.pt \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --sequence "MENDKV..." \
    --out results/attention_case_study.png
```

Visualizes which regions of the protein sequence the model attended to
most for a given drug-target pair, and prints the top-attended residues.
**For the paper**, run this on 2-4 representative examples: your
best-predicted pair, your worst-predicted pair, and (if you can find one
in the test set) a well-studied drug-target pair with known binding-site
literature, so you can sanity-check whether attention aligns with biology.

## Architecture

The model is now a small **"model zoo"** — configurable along two axes so you
can systematically compare architecture choices instead of committing to one
design up front:

```
SMILES ──> Morgan Fingerprint (1024-bit) ──> MLP ──────────┐
                                                             ├──> FUSION ──> Concat ──> MLP ──> pKd
Sequence ──> Embedding ──> [CNN | BiLSTM] ──> per-residue───┘
```

**Protein encoder** (`--prot_encoder`):
| Option | Description |
|---|---|
| `cnn` (default) | Multi-scale 1D convolutions — fast, good at local motifs |
| `bilstm` | Bidirectional LSTM — captures longer-range dependencies, slower |

**Fusion mechanism** (`--fusion`):
| Option | Description |
|---|---|
| `cross_attention` (default) | Molecule attends over protein residues. This is your explainability mechanism too. |
| `concat` | Naive baseline: concatenate pooled embeddings, no interaction term |
| `bilinear` | Low-rank bilinear pooling — multiplicative interaction, cheaper than attention |
| `mol_only` | Ignores the protein branch — ablation showing what the protein modality adds |
| `protein_only` | Ignores the molecule branch — ablation showing what the molecule modality adds |

Every combination shares the same `forward(mol_fp, prot_seq) -> (pred, attn)`
interface, so `train.py`, `evaluate.py`, and `run_experiments.py` all work
with any config without special-casing.

## 8. Compare architectures automatically (recommended before your final run)

Rather than guessing which encoder/fusion combination is best, run the
experiment comparison script. It trains a curated set of 8 combinations
for a reduced number of epochs each (fast screen, not the final result),
evaluates all of them on the test set, and ranks them:

```bash
python run_experiments.py --data_dir data/ --epochs 15 --batch_size 64
```

This covers:
- Full model: `cnn + cross_attention`
- Fusion ablations: `cnn + concat`, `cnn + bilinear`
- Protein encoder ablation: `bilstm + cross_attention`, `bilstm + concat`
- Modality ablations: `cnn + mol_only`, `cnn + protein_only`, `bilstm + protein_only`

It prints a ranked table and saves the full results to
`results/experiment_comparison.csv` — this table is exactly what goes in
your paper's "Ablation Studies" section. It also tells you the winning
config and the exact `train.py` command to re-run it for longer to get
your final reported numbers:

```bash
python train.py --data_dir data/ --prot_encoder <best> --fusion <best> --epochs 50
python evaluate.py --data_dir data/ --checkpoint checkpoints/<best>/best_model.pt \
    --prot_encoder <best> --fusion <best>
```

**Note:** with 274k+ training rows, running 8 configs even at 15 epochs each
will take a while (roughly 8x a single 15-epoch run). If time is tight, use
`--configs` to run a smaller subset first, e.g.:

```bash
python run_experiments.py --data_dir data/ --epochs 10 \
    --configs cnn:cross_attention,cnn:concat,bilstm:cross_attention
```

To run explainability, remember the checkpoint's `--prot_encoder` (fusion
must be `cross_attention` for attention weights to be meaningful):

```bash
python explainability.py --checkpoint checkpoints/cnn_cross_attention/best_model.pt \
    --prot_encoder cnn --smiles "..." --sequence "..."
```


## Suggested experiments for the paper

1. **Main result**: full model performance (RMSE, Pearson, CI) on the test set.
2. **Ablation 1**: remove cross-attention, use simple concatenation instead
   — shows the value of the fusion mechanism.
3. **Ablation 2**: molecule-only or protein-only baseline (mask one
   modality) — shows the value of multimodality itself.
4. **Explainability case studies**: 2-4 attention visualizations with
   discussion of whether high-attention residues are biologically plausible.
5. **Error analysis**: look at your worst-predicted pairs — are they
   outliers in affinity range, unusual protein families, etc.?

## Paper structure (suggested)

1. Abstract
2. Introduction (motivation, contribution bullets)
3. Related Work (DeepDTA, GraphDTA, DeepPurpose, MolTrans — search these)
4. Dataset & Preprocessing (BindingDB, cleaning steps, split strategy)
5. Method (architecture diagram + equations for cross-attention)
6. Experimental Setup (hyperparameters, hardware, metrics)
7. Results (main table + scatter plot)
8. Explainability Analysis (attention case studies)
9. Ablation Studies
10. Discussion & Limitations (fingerprint vs. GNN, CNN vs. pretrained
    protein models, generalization to novel targets)
11. Conclusion & Future Work

Once you have real numbers from a training run, share them with me and
I'll help draft the actual paper text (LaTeX or Word) with this structure.

## 9. Improving R² / fixing regression-to-the-mean bias

If your residual plot shows a downward trend (overpredicting low affinity,
underpredicting high affinity) and R² is lower than you'd like, this is
almost always **affinity-range imbalance**: your training set likely has
far more mid-range examples than high- or low-affinity ones, so plain MSE
loss lets the model minimize error by just predicting near the mean.

**Fix: inverse-frequency loss weighting.** Add `--weighted_loss` to give
sparse affinity ranges proportionally more weight in the loss:

```bash
python train.py --data_dir data/ --prot_encoder bilstm --fusion cross_attention \
    --epochs 50 --weighted_loss
```

This computes a per-example weight inversely proportional to how common
that example's affinity bin is (normalized to mean 1.0, so your learning
rate doesn't need retuning). Expect this to **trade some mid-range RMSE
for better performance at the extremes** and a higher R² overall — report
both the weighted and unweighted results if you have room, since it's a
genuine trade-off worth discussing.

You can also increase model capacity if you suspect underfitting:
```bash
python train.py --data_dir data/ --prot_encoder bilstm --fusion cross_attention \
    --epochs 50 --weighted_loss --hidden_dim 512 --embed_dim 256
```

After retraining, regenerate the full report to see the new metrics and
plots:
```bash
python generate_report.py --data_dir data/ \
    --checkpoint checkpoints/bilstm_cross_attention/best_model.pt \
    --prot_encoder bilstm --fusion cross_attention
```

## 10. Testing every architecture fairly (not just a quick screen)

Your first `run_experiments.py` pass used a short 8-epoch screen to save
time -- fine for picking a promising architecture, but not a fair
head-to-head since only the winner then got full training. For a genuinely
fair comparison where every architecture gets the same training budget,
re-run with the same epoch/patience budget as your final model:

```bash
python run_experiments.py --data_dir data/ --epochs 50 --patience 7 --weighted_loss
```

**Time warning:** this trains all 8 default configs (see `DEFAULT_CONFIGS`
in run_experiments.py) for up to 50 epochs each -- expect this to take
many hours. Use `caffeinate -i` (Mac) to prevent sleep, run it overnight,
and/or narrow `--configs` to just the architectures you actually want a
fair comparison between, e.g.:

```bash
caffeinate -i python run_experiments.py --data_dir data/ --epochs 50 --patience 7 \
    --weighted_loss --configs cnn:cross_attention,bilstm:cross_attention,bilstm:concat,cnn:mol_only,cnn:protein_only
```

The `mol_only` and `protein_only` configs are your **modality ablations**
-- comparing their RMSE/R² against the full multimodal model is the
cleanest evidence for "why multimodal helps" in your paper.

## File overview

| File | Purpose |
|---|---|
| `data_preprocessing.py` | Clean CSV, featurize SMILES/sequences, split data (also saves raw SMILES/sequence for traceability) |
| `model.py` | Multimodal architecture zoo (protein encoders x fusion mechanisms) |
| `train.py` | Training loop: early stopping, checkpointing, optional weighted loss |
| `evaluate.py` | Test-set metrics (RMSE/Pearson/CI) + scatter plot |
| `explainability.py` | Attention visualization for individual drug-target pairs |
| `run_experiments.py` | Trains/compares multiple architecture configs automatically |
| `generate_report.py` | Full metrics + all plots + fusion comparison + auto best/worst case studies |
| `smoke_test.py` | End-to-end sanity check with synthetic data |
| `requirements.txt` | Dependencies |
