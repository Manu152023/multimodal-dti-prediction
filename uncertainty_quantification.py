"""
uncertainty_quantification.py
------------------------------
Adds per-prediction UNCERTAINTY estimates using Monte Carlo Dropout --
something almost no student DTI project includes, and a genuine extension
of the "explainable insights" framing of this project: instead of just a
point prediction, the model can say "I'm confident about this one" vs.
"I'm not sure about this one."

How it works: your model already has dropout layers (used during training
to prevent overfitting). Normally dropout is turned OFF at inference time.
MC Dropout keeps it ON and runs the same input through the model N times --
each pass randomly drops different neurons, so you get N slightly different
predictions. Their spread (standard deviation) is a genuine, well-established
uncertainty estimate (Gal & Ghahramani, 2016).

Critically, this script also VALIDATES the uncertainty: if the model is
well-calibrated, examples where it reports high uncertainty should, on
average, have higher actual error. This correlation is itself a reportable
result -- it's not guaranteed to work, so showing that it does is evidence
your model's confidence is meaningful, not just noise.

Usage:
    python uncertainty_quantification.py --data_dir data/ \
        --checkpoint checkpoints/bilstm_cross_attention_large/best_model.pt \
        --prot_encoder bilstm --fusion cross_attention --hidden_dim 512 --embed_dim 256
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from model import MultimodalDTIModel, VALID_PROT_ENCODERS, VALID_FUSIONS
from train import get_device


def enable_dropout(model):
    """Sets ONLY dropout layers to train mode, keeping BatchNorm etc. (if any)
    in eval mode. This is the standard MC Dropout trick: the model has
    already been trained, but we want dropout's randomness active at
    inference time to sample from the model's implied predictive distribution.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(model, mol_fp, prot_seq, n_samples=30):
    """Runs n_samples stochastic forward passes and returns (mean, std) per
    example. std is the uncertainty estimate: higher = model is less sure.
    """
    model.eval()          # base eval mode (no batchnorm surprises etc.)
    enable_dropout(model)  # ...but force dropout layers back on

    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds, _ = model(mol_fp, prot_seq)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.stack(all_preds, axis=0)  # (n_samples, batch)
    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)
    return mean_pred, std_pred


def main():
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty quantification")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prot_encoder", choices=VALID_PROT_ENCODERS, required=True)
    parser.add_argument("--fusion", choices=VALID_FUSIONS, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=30,
                         help="Number of stochastic forward passes per example (more = smoother "
                              "uncertainty estimate, but slower). 30 is a reasonable default.")
    parser.add_argument("--out_dir", default="results/uncertainty/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    model = MultimodalDTIModel(prot_encoder_type=args.prot_encoder, fusion_type=args.fusion,
                                hidden_dim=args.hidden_dim, embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    with open(os.path.join(args.data_dir, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    mol_feats = torch.tensor(test_data["mol_feats"], dtype=torch.float32)
    prot_feats = torch.tensor(test_data["prot_feats"], dtype=torch.long)
    targets = np.asarray(test_data["labels"])

    print(f"Running MC Dropout ({args.n_samples} samples/example) on {len(targets)} test examples ...")
    print("This is slower than a normal forward pass since each example runs N times -- "
          "be patient on CPU/MPS.")

    all_means, all_stds = [], []
    batch_size = 128
    for i in range(0, len(targets), batch_size):
        fp_batch = mol_feats[i:i + batch_size].to(device)
        seq_batch = prot_feats[i:i + batch_size].to(device)
        mean_pred, std_pred = mc_dropout_predict(model, fp_batch, seq_batch, n_samples=args.n_samples)
        all_means.append(mean_pred)
        all_stds.append(std_pred)
        if (i // batch_size) % 20 == 0:
            print(f"  ... {i}/{len(targets)}")

    means = np.concatenate(all_means)
    stds = np.concatenate(all_stds)
    errors = np.abs(means - targets)

    # ---- Calibration check: does higher uncertainty predict higher error? ----
    calib_corr, calib_p = pearsonr(stds, errors)
    print(f"\nCalibration check: correlation between predicted uncertainty (std) and "
          f"actual error = {calib_corr:.3f} (p={calib_p:.2e})")
    if calib_corr > 0.15:
        print("  -> The model's uncertainty is meaningfully calibrated: when it says "
              "it's unsure, it tends to actually be more wrong. This is a genuine, "
              "reportable finding.")
    else:
        print("  -> Weak calibration signal. Still worth reporting honestly -- not every "
              "model produces well-calibrated uncertainty, and this itself is informative.")

    # ---- Plot 1: uncertainty vs error scatter ----
    plt.figure(figsize=(7, 6))
    plt.scatter(stds, errors, alpha=0.2, s=8)
    plt.xlabel("Predicted uncertainty (std across MC Dropout samples)")
    plt.ylabel("Actual absolute error")
    plt.title(f"Uncertainty Calibration (Pearson r={calib_corr:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "uncertainty_calibration.png"), dpi=150)
    plt.close()

    # ---- Plot 2: predictions with error bars for a random sample of examples ----
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(targets), size=min(40, len(targets)), replace=False)
    sample_idx = sample_idx[np.argsort(targets[sample_idx])]  # sort by true value for readability

    plt.figure(figsize=(10, 6))
    x = np.arange(len(sample_idx))
    plt.errorbar(x, means[sample_idx], yerr=stds[sample_idx], fmt="o", markersize=4,
                 alpha=0.7, label="Prediction ± uncertainty", capsize=3)
    plt.scatter(x, targets[sample_idx], color="red", marker="x", s=40, label="Actual pKd", zorder=5)
    plt.xlabel("Test example (sorted by true pKd)")
    plt.ylabel("pKd")
    plt.title("Predictions with Uncertainty Bars vs. Actual Values (40 random test examples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "predictions_with_uncertainty.png"), dpi=150)
    plt.close()

    # ---- Save raw numbers ----
    np.savez(os.path.join(args.out_dir, "mc_dropout_results.npz"),
              means=means, stds=stds, targets=targets)

    print(f"\nSaved:")
    print(f"  {args.out_dir}uncertainty_calibration.png")
    print(f"  {args.out_dir}predictions_with_uncertainty.png")
    print(f"  {args.out_dir}mc_dropout_results.npz")
    print(f"\nMean uncertainty (std): {stds.mean():.4f}")
    print(f"Uncertainty range: [{stds.min():.4f}, {stds.max():.4f}]")


if __name__ == "__main__":
    main()
