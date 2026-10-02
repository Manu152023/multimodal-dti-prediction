"""
compare_all_models.py
----------------------
Computes the FULL metric set (RMSE, MAE, R2, Pearson, Spearman, CI, and
accuracy-within-threshold) for every checkpoint you've trained, and prints
one consolidated table -- covering both the weighted-vs-unweighted
experiment and the fusion/architecture comparison, in a single run.

This does NOT generate plots (that's generate_report.py's job, and it's
slower) -- this is the fast way to just see all the numbers side by side.

Default model list assumes the checkpoint layout from this conversation's
earlier steps. Edit the MODELS list below, or pass --models to override:

    python compare_all_models.py

    # Or specify your own set:
    python compare_all_models.py --models \
        "baseline:checkpoints/bilstm_cross_attention/best_model.pt:bilstm:cross_attention:256:128" \
        "weighted:checkpoints/bilstm_cross_attention_weighted/best_model.pt:bilstm:cross_attention:256:128" \
        "large:checkpoints/bilstm_cross_attention_large/best_model.pt:bilstm:cross_attention:512:256"

Each --models entry is: name:checkpoint_path:prot_encoder:fusion:hidden_dim:embed_dim
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch

from model import MultimodalDTIModel
from generate_report import compute_full_metrics

# Edit this list to match the checkpoints you actually have, or override
# entirely with --models on the command line.
DEFAULT_MODELS = [
    # name              checkpoint path                                              prot_enc  fusion            hidden embed
    ("baseline_unweighted", "checkpoints/bilstm_cross_attention/best_model.pt",        "bilstm", "cross_attention", 256, 128),
    ("weighted_loss",       "checkpoints/bilstm_cross_attention_weighted/best_model.pt", "bilstm", "cross_attention", 256, 128),
    ("larger_capacity",     "checkpoints/bilstm_cross_attention_large/best_model.pt",   "bilstm", "cross_attention", 512, 256),
    ("fusion_cnn_crossattn", "checkpoints/cnn_cross_attention/best_model.pt",           "cnn",    "cross_attention", 256, 128),
    ("fusion_cnn_concat",   "checkpoints/cnn_concat/best_model.pt",                     "cnn",    "concat",          256, 128),
]


def parse_models_arg(models_arg):
    if not models_arg:
        return DEFAULT_MODELS
    parsed = []
    for entry in models_arg:
        name, path, prot_enc, fusion, hidden, embed = entry.split(":")
        parsed.append((name, path, prot_enc, fusion, int(hidden), int(embed)))
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Compare full metrics across every trained checkpoint")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--models", nargs="+", default=None,
                         help="Override the default model list. Each entry: "
                              "name:checkpoint_path:prot_encoder:fusion:hidden_dim:embed_dim")
    parser.add_argument("--out_csv", default="results/all_models_comparison.csv")
    args = parser.parse_args()

    models = parse_models_arg(args.models)

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
    print(f"Using device: {device}\n")

    with open(os.path.join(args.data_dir, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    mol_feats = torch.tensor(test_data["mol_feats"], dtype=torch.float32)
    prot_feats = torch.tensor(test_data["prot_feats"], dtype=torch.long)
    targets = np.asarray(test_data["labels"])
    print(f"Test set size: {len(targets)}\n")

    rows = []

    # Naive baseline: always predict the training set's mean pKd. This has
    # ZERO learned signal -- if your trained models don't clearly beat this,
    # something is genuinely wrong. If they do, that's concrete proof the
    # model has learned real structure-affinity patterns, not noise.
    with open(os.path.join(args.data_dir, "train.pkl"), "rb") as f:
        train_labels = np.asarray(pickle.load(f)["labels"])
    mean_baseline_preds = np.full_like(targets, fill_value=train_labels.mean())
    baseline_metrics = compute_full_metrics(mean_baseline_preds, targets)
    baseline_metrics["model"] = "naive_mean_baseline (no learning)"
    baseline_metrics["checkpoint"] = "N/A"
    baseline_metrics["prot_encoder"] = "N/A"
    baseline_metrics["fusion"] = "N/A"
    rows.append(baseline_metrics)
    print(f"Naive mean-baseline computed (predicts {train_labels.mean():.3f} for every example)\n")

    for name, ckpt_path, prot_enc, fusion, hidden_dim, embed_dim in models:
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"Evaluating: {name}  ({ckpt_path})")
        model = MultimodalDTIModel(prot_encoder_type=prot_enc, fusion_type=fusion,
                                    hidden_dim=hidden_dim, embed_dim=embed_dim).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        all_preds = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(targets), batch_size):
                fp_batch = mol_feats[i:i + batch_size].to(device)
                seq_batch = prot_feats[i:i + batch_size].to(device)
                preds, _ = model(fp_batch, seq_batch)
                all_preds.append(preds.cpu().numpy())
        preds = np.concatenate(all_preds)

        metrics = compute_full_metrics(preds, targets)
        metrics["model"] = name
        metrics["checkpoint"] = ckpt_path
        metrics["prot_encoder"] = prot_enc
        metrics["fusion"] = fusion
        rows.append(metrics)

    if not rows:
        print("\nNo checkpoints found. Check your paths / edit DEFAULT_MODELS in this script.")
        return

    df = pd.DataFrame(rows)
    col_order = ["model", "prot_encoder", "fusion", "rmse", "mae", "r2", "pearson",
                 "spearman", "concordance_index", "accuracy_within_0.5_pKd",
                 "accuracy_within_1.0_pKd", "accuracy_within_2.0_pKd", "checkpoint"]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values("rmse")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("\n" + "=" * 100)
    print("ALL MODELS -- FULL METRIC COMPARISON (ranked by RMSE, lower is better)")
    print("=" * 100)
    display_cols = ["model", "rmse", "mae", "r2", "pearson", "concordance_index",
                     "accuracy_within_1.0_pKd"]
    print(df[display_cols].to_string(index=False))
    print(f"\nFull table (all metrics) saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
