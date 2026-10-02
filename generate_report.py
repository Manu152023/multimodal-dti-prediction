"""
generate_report.py
-------------------
Produces the complete set of results you need before writing the paper:
    1. Full metrics for your final trained model (RMSE, MSE, MAE, R2,
       Pearson, Spearman, Concordance Index).
    2. A fusion/architecture comparison chart from run_experiments.py's
       results (this IS your multimodal fusion ablation figure).
    3. Error-analysis plots: predicted-vs-actual scatter, residual plot,
       error distribution histogram, performance broken down by affinity
       range.
    4. Automatic best-case and worst-case explainability figures, found by
       actually searching your test set (not a placeholder example).

Usage:
    python generate_report.py --data_dir data/ \
        --checkpoint checkpoints/bilstm_cross_attention/best_model.pt \
        --prot_encoder bilstm --fusion cross_attention

Everything is saved under results/report/.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from model import MultimodalDTIModel, VALID_PROT_ENCODERS, VALID_FUSIONS
from train import get_device
from evaluate import concordance_index
from explainability import get_attention_for_pair, plot_attention


def compute_full_metrics(preds: np.ndarray, targets: np.ndarray, max_ci_samples=3000) -> dict:
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    pearson_corr, _ = pearsonr(preds, targets)
    spearman_corr, _ = spearmanr(preds, targets)

    if len(targets) > max_ci_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(targets), max_ci_samples, replace=False)
        ci = concordance_index(targets[idx], preds[idx])
    else:
        ci = concordance_index(targets, preds)

    # "Accuracy"-style metrics: % of predictions within a given error
    # threshold of the true pKd. This is the standard regression substitute
    # for classification accuracy, and is reported in DTI papers (e.g. "X%
    # of predictions within 1 log unit of true affinity") precisely because
    # R^2/RMSE alone aren't intuitive to a non-technical reader.
    abs_error = np.abs(preds - targets)
    accuracy_within_0_5 = float(np.mean(abs_error <= 0.5))
    accuracy_within_1_0 = float(np.mean(abs_error <= 1.0))
    accuracy_within_2_0 = float(np.mean(abs_error <= 2.0))

    return {
        "rmse": rmse, "mse": mse, "mae": mae, "r2": r2,
        "pearson": float(pearson_corr), "spearman": float(spearman_corr),
        "concordance_index": ci,
        "accuracy_within_0.5_pKd": accuracy_within_0_5,
        "accuracy_within_1.0_pKd": accuracy_within_1_0,
        "accuracy_within_2.0_pKd": accuracy_within_2_0,
    }


def plot_scatter(preds, targets, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.25, s=8)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual pKd")
    plt.ylabel("Predicted pKd")
    plt.title("Predicted vs Actual Binding Affinity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_residuals(preds, targets, out_path):
    residuals = preds - targets
    plt.figure(figsize=(7, 5))
    plt.scatter(targets, residuals, alpha=0.25, s=8)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Actual pKd")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_error_histogram(preds, targets, out_path):
    errors = np.abs(preds - targets)
    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Absolute Error (|Predicted - Actual|)")
    plt.ylabel("Count")
    plt.title("Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_performance_by_range(preds, targets, out_path, n_bins=5):
    bins = np.linspace(targets.min(), targets.max(), n_bins + 1)
    bin_labels, bin_rmses, bin_counts = [], [], []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (targets >= bins[i]) & (targets < bins[i + 1])
        else:
            mask = (targets >= bins[i]) & (targets <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_rmse = float(np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2)))
        bin_labels.append(f"{bins[i]:.1f}-{bins[i+1]:.1f}")
        bin_rmses.append(bin_rmse)
        bin_counts.append(int(mask.sum()))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(bin_labels, bin_rmses, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Actual pKd range")
    ax1.set_ylabel("RMSE", color="steelblue")
    ax1.set_title("Model Error by Affinity Range")
    for i, count in enumerate(bin_counts):
        ax1.annotate(f"n={count}", (i, bin_rmses[i]), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fusion_comparison(comparison_csv, out_path):
    """The key 'multimodal fusion result' figure: compares all architecture
    configurations you ran through run_experiments.py on RMSE, Pearson, and
    Concordance Index side by side.
    """
    df = pd.read_csv(comparison_csv)
    df = df.sort_values("test_rmse")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("test_rmse", "Test RMSE (lower is better)"),
               ("test_pearson", "Test Pearson r (higher is better)"),
               ("test_ci", "Concordance Index (higher is better)")]

    for ax, (col, title) in zip(axes, metrics):
        colors = ["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(df))]
        ax.bar(df["config"], df[col], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        for i, v in enumerate(df[col]):
            ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=8)

    plt.suptitle("Architecture / Fusion Mechanism Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def find_best_worst_examples(preds, targets, n=1):
    errors = np.abs(preds - targets)
    best_idx = np.argsort(errors)[:n]
    worst_idx = np.argsort(errors)[-n:]
    return best_idx, worst_idx


def main():
    parser = argparse.ArgumentParser(description="Generate the full results/report suite")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--checkpoint", required=True, help="Path to your final trained model checkpoint")
    parser.add_argument("--prot_encoder", choices=VALID_PROT_ENCODERS, required=True)
    parser.add_argument("--fusion", choices=VALID_FUSIONS, required=True)
    parser.add_argument("--comparison_csv", default="results/experiment_comparison.csv",
                         help="Output of run_experiments.py, for the fusion comparison chart")
    parser.add_argument("--out_dir", default="results/report/")
    parser.add_argument("--hidden_dim", type=int, default=256,
                         help="Must match the --hidden_dim the checkpoint was trained with")
    parser.add_argument("--embed_dim", type=int, default=128,
                         help="Must match the --embed_dim the checkpoint was trained with")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}\n")

    # ---- 1. Load final model and run inference on the full test set ----
    model = MultimodalDTIModel(prot_encoder_type=args.prot_encoder, fusion_type=args.fusion,
                                hidden_dim=args.hidden_dim, embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with open(os.path.join(args.data_dir, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)

    mol_feats = torch.tensor(test_data["mol_feats"], dtype=torch.float32)
    prot_feats = torch.tensor(test_data["prot_feats"], dtype=torch.long)
    labels = test_data["labels"]
    has_raw_text = "smiles" in test_data and "sequence" in test_data

    print(f"Running inference on {len(labels)} test examples ...")
    all_preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            fp_batch = mol_feats[i:i + batch_size].to(device)
            seq_batch = prot_feats[i:i + batch_size].to(device)
            preds, _ = model(fp_batch, seq_batch)
            all_preds.append(preds.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.asarray(labels)

    # ---- 2. Full metrics ----
    metrics = compute_full_metrics(preds, targets)
    print("\n=== Final Model -- Full Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:>20}: {v:.4f}")

    with open(os.path.join(args.out_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- 3. Error-analysis plots ----
    print("\nGenerating plots ...")
    plot_scatter(preds, targets, os.path.join(args.out_dir, "pred_vs_actual.png"))
    plot_residuals(preds, targets, os.path.join(args.out_dir, "residuals.png"))
    plot_error_histogram(preds, targets, os.path.join(args.out_dir, "error_histogram.png"))
    plot_performance_by_range(preds, targets, os.path.join(args.out_dir, "performance_by_range.png"))

    # ---- 4. Fusion / architecture comparison chart ----
    if os.path.exists(args.comparison_csv):
        plot_fusion_comparison(args.comparison_csv, os.path.join(args.out_dir, "fusion_comparison.png"))
    else:
        print(f"\n[!] {args.comparison_csv} not found -- skipping fusion comparison chart. "
              f"Run run_experiments.py first if you want this figure.")

    # ---- 5. Best/worst explainability case studies ----
    if has_raw_text:
        print("\nFinding best- and worst-predicted examples for case studies ...")
        best_idx, worst_idx = find_best_worst_examples(preds, targets, n=1)

        for label, idx_arr in [("best", best_idx), ("worst", worst_idx)]:
            idx = idx_arr[0]
            smiles = test_data["smiles"][idx]
            sequence = test_data["sequence"][idx]
            actual = float(targets[idx])
            predicted = float(preds[idx])

            pred_check, attn = get_attention_for_pair(model, smiles, sequence, device)
            out_png = os.path.join(args.out_dir, f"case_study_{label}.png")
            plot_attention(sequence, attn, out_png, top_k=15)

            print(f"\n[{label.upper()} case] actual pKd={actual:.3f}  predicted pKd={predicted:.3f}  "
                  f"|error|={abs(actual - predicted):.3f}")
            print(f"  SMILES: {smiles}")
            print(f"  Sequence (first 60 aa): {sequence[:60]}...")
    else:
        print("\n[!] test.pkl doesn't contain raw SMILES/sequence strings (from an older "
              "preprocessing run). Re-run data_preprocessing.py with the updated script to "
              "enable automatic best/worst case studies.")

    print(f"\nAll outputs saved to: {args.out_dir}")
    print("\nFor your paper, use:")
    print("  - final_metrics.json          -> Results table")
    print("  - pred_vs_actual.png          -> Results figure")
    print("  - residuals.png, error_histogram.png, performance_by_range.png -> Error analysis")
    print("  - fusion_comparison.png       -> Ablation Studies figure")
    print("  - case_study_best.png / case_study_worst.png -> Explainability figures")


if __name__ == "__main__":
    main()
