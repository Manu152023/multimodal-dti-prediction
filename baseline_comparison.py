"""
baseline_comparison.py
------------------------
Trains a classical machine learning baseline (Random Forest) on the SAME
underlying features your deep model uses (Morgan fingerprint + a simple
protein sequence summary), and compares it against your multimodal deep
model.

Why this matters for your paper: right now, nothing in your results proves
that a deep multimodal architecture was actually NECESSARY -- for all a
reader knows, a simple Random Forest might do just as well on this data.
Running this baseline and showing your deep model beats it (or, if it
doesn't, honestly reporting that) is exactly the kind of comparison a
professor or reviewer will ask for if you don't include it yourself.

The protein feature here is intentionally simple (amino acid composition --
the frequency of each of the 20 amino acids in the sequence) since Random
Forest can't consume a raw variable-length sequence or a learned embedding
the way a neural network can. This is a fair, standard way to give a
classical model a shot at the same problem.

Usage:
    python baseline_comparison.py --data_dir data/
"""

import argparse
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard CI metric used in DTI literature (Gonen & Heller, 2005).
    Duplicated here (rather than imported from evaluate.py) so this script
    has zero deep-learning dependencies -- a classical ML baseline should
    be able to run in a plain sklearn environment without torch installed.
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


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def sequence_to_composition(int_seq: np.ndarray, vocab_offset: int = 1) -> np.ndarray:
    """Converts an integer-encoded protein sequence (as produced by
    data_preprocessing.py) back into a 20-dim amino acid composition vector
    (the fraction of the sequence made up of each amino acid). This is the
    standard, simplest protein feature classical ML models use.
    """
    # data_preprocessing.py's AA_TO_INT maps "ACDEFGHIKLMNPQRSTVWYXBZUO" to 1..25,
    # with 0 = padding. We only care about the 20 standard amino acids (1..20).
    counts = np.zeros(20, dtype=np.float32)
    real_residues = int_seq[int_seq != 0]
    if len(real_residues) == 0:
        return counts
    for aa_idx in range(1, 21):  # standard 20 amino acids occupy codes 1-20
        counts[aa_idx - 1] = np.sum(real_residues == aa_idx)
    counts = counts / max(len(real_residues), 1)
    return counts


def compute_metrics(preds, targets, max_ci_samples=3000):
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
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
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": float(pearson_corr),
            "spearman": float(spearman_corr), "concordance_index": ci}


def load_split(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    mol_feats = data["mol_feats"]  # already Morgan fingerprints, usable as-is
    prot_comp = np.stack([sequence_to_composition(seq) for seq in data["prot_feats"]])
    X = np.concatenate([mol_feats, prot_comp], axis=1)
    y = np.asarray(data["labels"])
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Classical ML baseline for comparison")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=-1, help="-1 uses all CPU cores")
    parser.add_argument("--out_dir", default="results/baseline/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading and featurizing data (Morgan fingerprint + amino acid composition) ...")
    X_train, y_train = load_split(os.path.join(args.data_dir, "train.pkl"))
    X_test, y_test = load_split(os.path.join(args.data_dir, "test.pkl"))
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print(f"\nTraining Random Forest (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}) ...")
    start = time.time()
    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                   n_jobs=args.n_jobs, random_state=42, verbose=1)
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training took {elapsed:.1f}s")

    preds = model.predict(X_test)
    metrics = compute_metrics(preds, y_test)

    print("\n=== Random Forest Baseline -- Test Set Results ===")
    for k, v in metrics.items():
        print(f"  {k:>18}: {v:.4f}")

    with open(os.path.join(args.out_dir, "rf_baseline_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    np.savez(os.path.join(args.out_dir, "rf_baseline_predictions.npz"), preds=preds, targets=y_test)

    print(f"\nSaved to: {args.out_dir}")
    print("\nCompare these numbers directly against your deep model's final_metrics.json.")
    print("If your deep model wins (especially on RMSE/R^2/CI), that's your evidence that ")
    print("the multimodal architecture earns its complexity over a simple classical baseline.")


if __name__ == "__main__":
    main()
