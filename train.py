"""
train.py
--------
Trains the MultimodalDTIModel on the preprocessed BindingDB splits
produced by data_preprocessing.py.

Usage (single run):
    python train.py --data_dir data/ --epochs 30 --batch_size 64 --lr 1e-3 \
        --prot_encoder cnn --fusion cross_attention

To compare multiple architecture configurations automatically, see
run_experiments.py, which imports train_model() from this file.
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

from model import MultimodalDTIModel, VALID_PROT_ENCODERS, VALID_FUSIONS


class DTIDataset(Dataset):
    def __init__(self, pkl_path, sample_weights=None):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.mol_feats = torch.tensor(data["mol_feats"], dtype=torch.float32)
        self.prot_feats = torch.tensor(data["prot_feats"], dtype=torch.long)
        self.labels = torch.tensor(data["labels"], dtype=torch.float32)
        # Optional per-example loss weights (e.g. inverse-frequency weights to
        # counter affinity-range imbalance). Defaults to all-ones so nothing
        # changes for existing code that doesn't pass this in.
        if sample_weights is not None:
            self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        else:
            self.sample_weights = torch.ones(len(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mol_feats[idx], self.prot_feats[idx], self.labels[idx], self.sample_weights[idx]


def compute_inverse_frequency_weights(labels: np.ndarray, n_bins: int = 10,
                                       power: float = 1.0, max_weight: float = 5.0) -> np.ndarray:
    """Computes a per-example loss weight that's inversely proportional to
    how common that example's affinity range is in the training set. This
    directly counters the regression-to-the-mean bias seen in the residual
    plot: without it, MSE loss is dominated by the dense mid-range bins and
    the model is never penalized much for being wrong on the sparse
    high/low-affinity examples.

    power: softens the weighting. power=1.0 is raw inverse-frequency (can be
    very aggressive -- an empty-ish bin with 1/1000th the density of the
    mode gets ~1000x the weight, which in practice destabilizes training and
    makes the model chase a handful of outliers rather than generalizing).
    power=0.5 (square-root of inverse-frequency) is a much gentler default
    that still nudges the model to care about sparse ranges without
    wrecking overall performance. power=0.0 disables weighting entirely
    (equivalent to plain MSE).

    max_weight: hard cap on any single example's weight, applied after the
    power transform, as a second safety net against near-empty bins
    producing runaway weights.

    Weights are normalized to have mean 1.0 (after capping) so the overall
    loss scale -- and therefore the learning rate -- doesn't need retuning.
    """
    bins = np.linspace(labels.min(), labels.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(labels, bins) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)
    counts = np.where(counts == 0, 1, counts)  # avoid div-by-zero for empty bins

    if power <= 0:
        return np.ones(len(labels), dtype=np.float32)

    inv_freq = (1.0 / counts[bin_idx]) ** power
    weights = inv_freq / inv_freq.mean()
    weights = np.clip(weights, None, max_weight)
    weights = weights / weights.mean()  # renormalize to mean 1.0 after capping
    return weights.astype(np.float32)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    if np.std(preds) > 1e-8 and np.std(targets) > 1e-8:
        pearson_corr, _ = pearsonr(preds, targets)
    else:
        pearson_corr = 0.0
    return {"rmse": rmse, "mse": mse, "mae": mae, "pearson": float(pearson_corr)}


def run_epoch(model, loader, optimizer, device, criterion, train=True, use_weights=False):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for mol_fp, prot_seq, labels, weights in loader:
            mol_fp, prot_seq, labels = mol_fp.to(device), prot_seq.to(device), labels.to(device)
            weights = weights.to(device)

            if train:
                optimizer.zero_grad()

            preds, _ = model(mol_fp, prot_seq)
            per_example_loss = criterion(preds, labels)  # criterion always uses reduction="none"
            loss = (per_example_loss * weights).mean() if use_weights else per_example_loss.mean()

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(all_targets)
    return metrics


def train_model(train_loader, val_loader, device, prot_encoder_type="cnn",
                 fusion_type="cross_attention", epochs=30, lr=1e-3,
                 weight_decay=1e-5, patience=7, out_dir="checkpoints/",
                 seed=42, verbose=True, use_weighted_loss=False,
                 hidden_dim=256, embed_dim=128, scheduler_patience=3):
    """Trains one (prot_encoder_type, fusion_type) configuration end-to-end
    with early stopping. Returns (best_val_metrics, history, checkpoint_path).

    This is the function run_experiments.py calls in a loop to compare
    architectures; train.py's CLI (main(), below) is a thin wrapper around it
    for single manual runs.

    use_weighted_loss: if True, the train_loader's dataset must have been
    built with compute_inverse_frequency_weights() sample weights -- this
    counters the regression-to-the-mean bias from affinity-range imbalance
    (see compute_inverse_frequency_weights docstring).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    model = MultimodalDTIModel(prot_encoder_type=prot_encoder_type, fusion_type=fusion_type,
                                hidden_dim=hidden_dim, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                            patience=scheduler_patience)
    criterion = nn.MSELoss(reduction="none")  # reduced manually in run_epoch (weighted or plain mean)

    best_val_metrics = None
    best_val_rmse = float("inf")
    epochs_no_improve = 0
    history = []
    checkpoint_path = os.path.join(out_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, criterion,
                                   train=True, use_weights=use_weighted_loss)
        val_metrics = run_epoch(model, val_loader, optimizer, device, criterion,
                                 train=False, use_weights=False)  # always evaluate unweighted
        scheduler.step(val_metrics["rmse"])
        elapsed = time.time() - start

        if verbose:
            print(f"[Epoch {epoch:03d}/{epochs}] "
                  f"train_loss={train_metrics['loss']:.4f} train_rmse={train_metrics['rmse']:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} val_rmse={val_metrics['rmse']:.4f} "
                  f"val_pearson={val_metrics['pearson']:.4f} ({elapsed:.1f}s)")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_val_metrics = val_metrics
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(f"  -> New best model saved (val_rmse={best_val_rmse:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch} epochs.")
                break

    with open(os.path.join(out_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    return best_val_metrics, history, checkpoint_path


def default_out_dir(prot_encoder_type, fusion_type, weighted_loss=False):
    suffix = "_weighted" if weighted_loss else ""
    return os.path.join("checkpoints", f"{prot_encoder_type}_{fusion_type}{suffix}")


def main():
    parser = argparse.ArgumentParser(description="Train the multimodal DTI model")
    parser.add_argument("--data_dir", default="data/", help="Directory with train/val/test .pkl splits")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--out_dir", default=None,
                         help="Where to save the best model. Defaults to "
                              "checkpoints/<prot_encoder>_<fusion>/ so different "
                              "configs don't overwrite each other.")
    parser.add_argument("--prot_encoder", choices=VALID_PROT_ENCODERS, default="cnn",
                         help="Protein encoder architecture")
    parser.add_argument("--fusion", choices=VALID_FUSIONS, default="cross_attention",
                         help="Fusion mechanism (see model.py docstring for details on each)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted_loss", action="store_true",
                         help="Use inverse-frequency loss weighting to counter affinity-range "
                              "imbalance (helps the regression-to-the-mean bias / improves R^2 "
                              "on sparse high/low-affinity examples, at some cost to mid-range RMSE). "
                              "Saves to a separate checkpoint dir (suffix '_weighted') so it never "
                              "overwrites an unweighted run.")
    parser.add_argument("--n_weight_bins", type=int, default=10,
                         help="Number of affinity bins used for --weighted_loss")
    parser.add_argument("--weight_power", type=float, default=0.3,
                         help="Softens --weighted_loss: 1.0 = raw inverse-frequency (aggressive, "
                              "can destabilize training), 0.5 = sqrt (moderate), 0.3 = gentle "
                              "(default, recommended starting point), 0.0 = disabled.")
    parser.add_argument("--max_weight", type=float, default=5.0,
                         help="Hard cap on any single example's loss weight, as a safety net "
                              "against near-empty bins producing runaway weights.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Model capacity: head hidden size")
    parser.add_argument("--embed_dim", type=int, default=128, help="Model capacity: encoder embedding size")
    args = parser.parse_args()

    out_dir = args.out_dir or default_out_dir(args.prot_encoder, args.fusion, weighted_loss=args.weighted_loss)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Config: prot_encoder={args.prot_encoder}, fusion={args.fusion}, "
          f"weighted_loss={args.weighted_loss} (power={args.weight_power}, max={args.max_weight}), "
          f"hidden_dim={args.hidden_dim}, embed_dim={args.embed_dim}")
    print(f"Checkpoints will be saved to: {out_dir}")

    if args.weighted_loss:
        with open(os.path.join(args.data_dir, "train.pkl"), "rb") as f:
            train_labels_for_weights = pickle.load(f)["labels"]
        weights = compute_inverse_frequency_weights(
            np.asarray(train_labels_for_weights), n_bins=args.n_weight_bins,
            power=args.weight_power, max_weight=args.max_weight)
        train_ds = DTIDataset(os.path.join(args.data_dir, "train.pkl"), sample_weights=weights)
        print(f"Weighted loss enabled ({args.n_weight_bins} bins, power={args.weight_power}). "
              f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    else:
        train_ds = DTIDataset(os.path.join(args.data_dir, "train.pkl"))

    val_ds = DTIDataset(os.path.join(args.data_dir, "val.pkl"))
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    best_val_metrics, _, checkpoint_path = train_model(
        train_loader, val_loader, device,
        prot_encoder_type=args.prot_encoder, fusion_type=args.fusion,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, out_dir=out_dir, seed=args.seed, verbose=True,
        use_weighted_loss=args.weighted_loss, hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
    )

    print(f"\nTraining complete. Best val RMSE: {best_val_metrics['rmse']:.4f}")
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
