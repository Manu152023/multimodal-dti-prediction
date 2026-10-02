"""
evaluate.py
-----------
Loads a trained checkpoint and evaluates it on the held-out test split.
Produces standard DTI regression metrics plus a predicted-vs-actual
scatter plot and a Concordance Index (CI), which is the standard ranking
metric used across DTI benchmarks (DeepDTA, GraphDTA, etc.), alongside
RMSE/MSE/Pearson so results are directly comparable to prior work.

Usage:
    python evaluate.py --data_dir data/ --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from model import MultimodalDTIModel, VALID_PROT_ENCODERS, VALID_FUSIONS
from train import DTIDataset, get_device
from torch.utils.data import DataLoader


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard CI metric used in DTI literature (Gonen & Heller, 2005).
    Measures the probability that, for a randomly chosen pair with
    different true affinities, the model ranks them in the correct order.
    """
    order = np.argsort(y_true)
    y_true, y_pred = y_true[order], y_pred[order]
    n = len(y_true)

    concordant, total_pairs = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[j] != y_true[i]:
                total_pairs += 1
                if y_pred[j] > y_pred[i]:
                    concordant += 1
                elif y_pred[j] == y_pred[i]:
                    concordant += 0.5
    return concordant / total_pairs if total_pairs > 0 else float("nan")


def evaluate(model, loader, device, max_ci_samples=2000):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for mol_fp, prot_seq, labels, _weights in loader:
            mol_fp, prot_seq = mol_fp.to(device), prot_seq.to(device)
            preds, _ = model(mol_fp, prot_seq)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    pearson_corr, _ = pearsonr(preds, targets)
    spearman_corr, _ = spearmanr(preds, targets)

    # CI is O(n^2); subsample for large test sets to keep this tractable
    if len(targets) > max_ci_samples:
        idx = np.random.choice(len(targets), max_ci_samples, replace=False)
        ci = concordance_index(targets[idx], preds[idx])
    else:
        ci = concordance_index(targets, preds)

    metrics = {
        "rmse": rmse, "mse": mse, "mae": mae,
        "pearson": float(pearson_corr), "spearman": float(spearman_corr),
        "concordance_index": ci,
    }
    return metrics, preds, targets


def plot_predictions(preds, targets, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.3, s=10)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual pKd")
    plt.ylabel("Predicted pKd")
    plt.title("Predicted vs Actual Binding Affinity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved scatter plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained DTI model on the test set")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--out_dir", default="results/")
    parser.add_argument("--prot_encoder", choices=VALID_PROT_ENCODERS, default="cnn",
                         help="Must match the architecture the checkpoint was trained with")
    parser.add_argument("--fusion", choices=VALID_FUSIONS, default="cross_attention",
                         help="Must match the architecture the checkpoint was trained with")
    parser.add_argument("--hidden_dim", type=int, default=256,
                         help="Must match the --hidden_dim the checkpoint was trained with")
    parser.add_argument("--embed_dim", type=int, default=128,
                         help="Must match the --embed_dim the checkpoint was trained with")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()

    test_ds = DTIDataset(os.path.join(args.data_dir, "test.pkl"))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = MultimodalDTIModel(prot_encoder_type=args.prot_encoder, fusion_type=args.fusion,
                                hidden_dim=args.hidden_dim, embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    metrics, preds, targets = evaluate(model, test_loader, device)

    print("\n=== Test Set Results ===")
    for k, v in metrics.items():
        print(f"  {k:>18}: {v:.4f}")

    with open(os.path.join(args.out_dir, "test_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    plot_predictions(preds, targets, os.path.join(args.out_dir, "pred_vs_actual.png"))

    np.savez(os.path.join(args.out_dir, "predictions.npz"), preds=preds, targets=targets)
    print(f"\nSaved metrics and predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
