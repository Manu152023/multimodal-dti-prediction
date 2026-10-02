"""
run_experiments.py
-------------------
Systematically trains and evaluates every meaningful combination of
protein encoder (CNN / BiLSTM) and fusion mechanism (cross-attention /
concat / bilinear / mol-only / protein-only), then produces a ranked
comparison table -- exactly the ablation/model-comparison table you'd
want for the "Experiments" or "Ablation Studies" section of the paper.

By default this runs a REDUCED number of epochs per config (fast
comparison pass) so you can see which architecture wins before committing
full training time to just one. Once you know the winner, run train.py
directly with more epochs for the final reported result.

Usage:
    python run_experiments.py --data_dir data/ --epochs 15 --batch_size 64

    # Run only a subset (comma-separated "prot_encoder:fusion" pairs):
    python run_experiments.py --data_dir data/ \
        --configs cnn:cross_attention,cnn:concat,bilstm:cross_attention

Output:
    results/experiment_comparison.csv  -- one row per config, all metrics
    Printed ranked table (best RMSE first)
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from model import VALID_PROT_ENCODERS, VALID_FUSIONS
from train import DTIDataset, get_device, train_model, default_out_dir
from evaluate import evaluate as evaluate_on_loader

# Meaningful default combinations. We skip redundant pairings (e.g.
# mol_only / protein_only don't depend on the *other* modality's encoder
# choice in a way that matters for mol_only, so we only run each once).
DEFAULT_CONFIGS = [
    ("cnn", "cross_attention"),     # full proposed model
    ("cnn", "concat"),              # ablation: fusion mechanism
    ("cnn", "bilinear"),            # ablation: fusion mechanism
    ("bilstm", "cross_attention"),  # ablation: protein encoder choice
    ("bilstm", "concat"),
    ("cnn", "mol_only"),            # ablation: value of protein modality
    ("cnn", "protein_only"),        # ablation: value of molecule modality
    ("bilstm", "protein_only"),
]


def parse_configs(configs_str):
    if not configs_str:
        return DEFAULT_CONFIGS
    configs = []
    for pair in configs_str.split(","):
        prot_enc, fusion = pair.split(":")
        prot_enc, fusion = prot_enc.strip(), fusion.strip()
        if prot_enc not in VALID_PROT_ENCODERS:
            raise ValueError(f"Invalid prot_encoder '{prot_enc}' in --configs. "
                              f"Expected one of {VALID_PROT_ENCODERS}.")
        if fusion not in VALID_FUSIONS:
            raise ValueError(f"Invalid fusion '{fusion}' in --configs. "
                              f"Expected one of {VALID_FUSIONS}.")
        configs.append((prot_enc, fusion))
    return configs


def main():
    parser = argparse.ArgumentParser(description="Compare multiple DTI model architectures")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--epochs", type=int, default=15,
                         help="Epochs per config for this comparison pass. For a FAIR final "
                              "comparison (not just a quick screen), use the same epoch/patience "
                              "budget you used for your best single model, e.g. --epochs 50 "
                              "--patience 7, so no config is unfairly cut short.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4,
                         help="Early stopping patience per config (kept tight for a fast screen; "
                              "raise this for a fair final comparison)")
    parser.add_argument("--out_dir", default="results/")
    parser.add_argument("--configs", default=None,
                         help="Comma-separated prot_encoder:fusion pairs to run. "
                              "Default: a curated set of 8 combinations covering "
                              "the main ablations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted_loss", action="store_true",
                         help="Apply inverse-frequency loss weighting to every config, to counter "
                              "affinity-range imbalance (see train.py's compute_inverse_frequency_weights)")
    parser.add_argument("--n_weight_bins", type=int, default=10)
    parser.add_argument("--weight_power", type=float, default=0.3,
                         help="Softens the weighting (see train.py --weight_power for details). "
                              "Default 0.3 is a gentle setting; 1.0 (raw inverse-frequency) is "
                              "known to destabilize training on this dataset.")
    parser.add_argument("--max_weight", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = parse_configs(args.configs)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Running {len(configs)} configuration(s): {configs}")
    print(f"Epochs per config: {args.epochs}, patience: {args.patience}, "
          f"weighted_loss: {args.weighted_loss}\n")

    if args.weighted_loss:
        with open(os.path.join(args.data_dir, "train.pkl"), "rb") as f:
            train_labels_for_weights = np.asarray(pickle.load(f)["labels"])
        from train import compute_inverse_frequency_weights
        weights = compute_inverse_frequency_weights(train_labels_for_weights, n_bins=args.n_weight_bins,
                                                     power=args.weight_power, max_weight=args.max_weight)
        train_ds = DTIDataset(os.path.join(args.data_dir, "train.pkl"), sample_weights=weights)
    else:
        train_ds = DTIDataset(os.path.join(args.data_dir, "train.pkl"))

    val_ds = DTIDataset(os.path.join(args.data_dir, "val.pkl"))
    test_ds = DTIDataset(os.path.join(args.data_dir, "test.pkl"))
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = []

    for i, (prot_enc, fusion) in enumerate(configs, 1):
        config_name = f"{prot_enc}_{fusion}"
        print("=" * 70)
        print(f"[{i}/{len(configs)}] Training config: prot_encoder={prot_enc}, fusion={fusion}")
        print("=" * 70)

        out_dir = default_out_dir(prot_enc, fusion, weighted_loss=args.weighted_loss)
        start = time.time()

        best_val_metrics, _, checkpoint_path = train_model(
            train_loader, val_loader, device,
            prot_encoder_type=prot_enc, fusion_type=fusion,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            out_dir=out_dir, seed=args.seed, verbose=True,
            use_weighted_loss=args.weighted_loss,
        )

        # Re-load the best checkpoint for this config and evaluate on the test set
        import torch
        from model import MultimodalDTIModel
        model = MultimodalDTIModel(prot_encoder_type=prot_enc, fusion_type=fusion).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_metrics, _, _ = evaluate_on_loader(model, test_loader, device)

        elapsed = time.time() - start
        n_params = sum(p.numel() for p in model.parameters())

        row = {
            "config": config_name,
            "prot_encoder": prot_enc,
            "fusion": fusion,
            "params": n_params,
            "val_rmse": best_val_metrics["rmse"],
            "val_pearson": best_val_metrics["pearson"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_pearson": test_metrics["pearson"],
            "test_spearman": test_metrics["spearman"],
            "test_ci": test_metrics["concordance_index"],
            "train_time_sec": round(elapsed, 1),
            "checkpoint": checkpoint_path,
        }
        results.append(row)
        print(f"\n[{config_name}] test_rmse={test_metrics['rmse']:.4f} "
              f"test_pearson={test_metrics['pearson']:.4f} test_ci={test_metrics['concordance_index']:.4f} "
              f"({elapsed:.1f}s)\n")

    df = pd.DataFrame(results).sort_values("test_rmse")
    csv_path = os.path.join(args.out_dir, "experiment_comparison.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON (ranked by test RMSE, lower is better)")
    print("=" * 70)
    print(df[["config", "params", "test_rmse", "test_mae", "test_pearson",
              "test_spearman", "test_ci", "train_time_sec"]].to_string(index=False))
    print(f"\nFull results saved to: {csv_path}")

    best = df.iloc[0]
    print(f"\nBest config: {best['config']} "
          f"(test RMSE={best['test_rmse']:.4f}, Pearson={best['test_pearson']:.4f}, CI={best['test_ci']:.4f})")
    print(f"Checkpoint: {best['checkpoint']}")
    print("\nNext step: run this config's full training with more epochs for your reported result, e.g.:")
    print(f"  python train.py --data_dir {args.data_dir} --prot_encoder {best['prot_encoder']} "
          f"--fusion {best['fusion']} --epochs 50")


if __name__ == "__main__":
    main()
