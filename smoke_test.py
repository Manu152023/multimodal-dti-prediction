"""
smoke_test.py
-------------
Sanity-checks the entire pipeline using small SYNTHETIC data before you
spend time running it on the real 1.59GB BindingDB file. This catches
environment/shape/import bugs in seconds.

Run this FIRST after `pip install -r requirements.txt`:
    python smoke_test.py
"""

import os
import random
import shutil

import numpy as np
import pandas as pd
import torch

from data_preprocessing import build_dataset, split_and_save, clean_dataframe
from model import MultimodalDTIModel, VALID_PROT_ENCODERS, VALID_FUSIONS
from train import DTIDataset
from torch.utils.data import DataLoader

# A handful of real, valid small-molecule SMILES (so RDKit parsing succeeds)
SAMPLE_SMILES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",          # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",      # caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",     # ibuprofen
    "C1=CC=C(C=C1)C=O",                   # benzaldehyde
    "CCO",                                # ethanol
    "C1CCCCC1",                           # cyclohexane
]

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def make_synthetic_csv(path: str, n_rows: int = 300):
    random.seed(0)
    rows = []
    for _ in range(n_rows):
        smiles = random.choice(SAMPLE_SMILES)
        seq_len = random.randint(50, 400)
        sequence = "".join(random.choice(AMINO_ACIDS) for _ in range(seq_len))
        affinity = round(random.uniform(3.0, 10.0), 2)
        rows.append({
            "compound_iso_smiles": smiles,
            "target_sequence": sequence,
            "affinity": affinity,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Created synthetic dataset: {path} ({n_rows} rows)")


def main():
    tmp_dir = "smoke_test_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    csv_path = os.path.join(tmp_dir, "synthetic.csv")
    data_dir = os.path.join(tmp_dir, "data")

    print("=" * 60)
    print("STEP 1: Generating synthetic data")
    print("=" * 60)
    make_synthetic_csv(csv_path)

    print("\n" + "=" * 60)
    print("STEP 2: Testing data_preprocessing.py")
    print("=" * 60)
    mol_feats, prot_feats, labels, df = build_dataset(csv_path)
    assert mol_feats.shape[1] == 1024, "Fingerprint dim mismatch"
    assert prot_feats.shape[1] == 1000, "Protein encoding length mismatch"
    split_and_save(mol_feats, prot_feats, labels, data_dir)
    print("Preprocessing OK.")

    print("\n" + "=" * 60)
    print("STEP 3: Testing model.py forward pass (full model zoo)")
    print("=" * 60)
    dummy_fp = torch.rand(4, 1024)
    dummy_seq = torch.randint(0, 26, (4, 1000))
    dummy_seq[:, 500:] = 0
    for prot_enc in VALID_PROT_ENCODERS:
        for fusion in VALID_FUSIONS:
            model = MultimodalDTIModel(prot_encoder_type=prot_enc, fusion_type=fusion)
            pred, attn = model(dummy_fp, dummy_seq)
            assert pred.shape == (4,), f"[{prot_enc}/{fusion}] Unexpected prediction shape: {pred.shape}"
            assert attn.shape == (4, 1000), f"[{prot_enc}/{fusion}] Unexpected attention shape: {attn.shape}"
    print(f"All {len(VALID_PROT_ENCODERS) * len(VALID_FUSIONS)} model configs OK.")
    model = MultimodalDTIModel()  # default config, used in step 4 below
    print(f"Default model params: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("STEP 4: Testing DTIDataset + DataLoader + one training step")
    print("=" * 60)
    train_ds = DTIDataset(os.path.join(data_dir, "train.pkl"))
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    mol_fp, prot_seq, y, _weights = next(iter(loader))
    optimizer.zero_grad()
    pred, _ = model(mol_fp, prot_seq)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    print(f"One training step OK. Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Download the real dataset CSV from Kaggle")
    print("  2. python data_preprocessing.py --csv <real_file>.csv --out data/")
    print("  3. python run_experiments.py --data_dir data/ --epochs 15   # compare architectures")
    print("  4. python train.py --data_dir data/ --prot_encoder <best> --fusion <best> --epochs 50")
    print("  5. python evaluate.py --data_dir data/ --checkpoint <path> --prot_encoder <best> --fusion <best>")
    print("  6. python explainability.py --checkpoint <path> --smiles <SMILES> --sequence <SEQUENCE>")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
