"""
data_preprocessing.py
----------------------
Loads the BindingDB high-confidence pKd dataset, cleans it, and converts
raw modalities (SMILES strings, protein sequences) into numeric tensors
ready for the multimodal DTI model.

Expected input CSV columns:
    compound_iso_smiles : str   -- isomeric SMILES of the ligand/drug
    target_sequence     : str   -- amino-acid sequence of the target protein
    affinity            : float -- pKd (negative log dissociation constant)

Usage:
    python data_preprocessing.py --csv bindingdb_highconf_pKd_clean.csv --out data/
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")  # suppress RDKit's per-molecule deprecation/warning spam
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARN] RDKit not installed. Install with `pip install rdkit`. "
          "Falling back is not supported for real training.")

# ---------------------------------------------------------------------------
# Protein sequence vocabulary (20 standard amino acids + a few ambiguity codes)
# ---------------------------------------------------------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYXBZUO"
AA_TO_INT = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 reserved for padding
PROTEIN_MAX_LEN = 1000     # truncate / pad protein sequences to this length
MORGAN_RADIUS = 2
MORGAN_NBITS = 1024        # length of the molecular fingerprint vector

# Instantiate the fingerprint generator once at module load (not per-molecule):
# this is both faster and avoids the deprecated per-call API.
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS
) if RDKIT_AVAILABLE else None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing values, deduplicate, and filter invalid SMILES/sequences."""
    required_cols = {"compound_iso_smiles", "target_sequence", "affinity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.dropna(subset=list(required_cols)).copy()
    df = df.drop_duplicates(subset=["compound_iso_smiles", "target_sequence"])

    # Keep only sensible affinity values (pKd typically ranges ~0-14)
    df = df[(df["affinity"] > 0) & (df["affinity"] < 15)]

    # Validate SMILES parse with RDKit; drop unparsable molecules
    if RDKIT_AVAILABLE:
        valid_mask = df["compound_iso_smiles"].apply(_is_valid_smiles)
        df = df[valid_mask]

    # Drop sequences with non-standard characters beyond our vocabulary,
    # or that are unreasonably short/long
    df = df[df["target_sequence"].str.len().between(20, 3000)]

    return df.reset_index(drop=True)


def _is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def smiles_to_fingerprint(smiles: str, n_bits: int = MORGAN_NBITS) -> np.ndarray:
    """Convert a SMILES string to a Morgan (ECFP) fingerprint bit vector.

    Morgan fingerprints are used instead of a full graph representation to
    keep the pipeline dependency-light (no torch-geometric needed) while
    still capturing local substructure information relevant to binding.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = _MORGAN_GEN.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for i in fp.GetOnBits():
        arr[i] = 1.0
    return arr


def sequence_to_int_array(seq: str, max_len: int = PROTEIN_MAX_LEN) -> np.ndarray:
    """Convert a protein sequence to a fixed-length integer-encoded array."""
    seq = re.sub(r"[^A-Z]", "", seq.upper())
    encoded = [AA_TO_INT.get(ch, 0) for ch in seq[:max_len]]
    if len(encoded) < max_len:
        encoded = encoded + [0] * (max_len - len(encoded))
    return np.array(encoded, dtype=np.int64)


def build_dataset(csv_path: str, sample_frac: float = None, random_state: int = 42):
    """Load, clean, and featurize the dataset. Returns numpy arrays."""
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  raw rows: {len(df)}")

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"  sampled to: {len(df)} rows (frac={sample_frac})")

    df = clean_dataframe(df)
    print(f"  rows after cleaning: {len(df)}")

    print("Featurizing molecules (Morgan fingerprints) ...")
    mol_feats = np.stack(df["compound_iso_smiles"].apply(smiles_to_fingerprint).values)

    print("Featurizing proteins (integer-encoded sequences) ...")
    prot_feats = np.stack(df["target_sequence"].apply(sequence_to_int_array).values)

    labels = df["affinity"].values.astype(np.float32)

    return mol_feats, prot_feats, labels, df


def split_and_save(mol_feats, prot_feats, labels, out_dir, df=None, test_size=0.15, val_size=0.15,
                    random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    # First split off test set, then split remainder into train/val
    idx = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    val_fraction_of_remainder = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_fraction_of_remainder,
                                           random_state=random_state)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for name, indices in splits.items():
        data = {
            "mol_feats": mol_feats[indices],
            "prot_feats": prot_feats[indices],
            "labels": labels[indices],
        }
        # Also carry the raw SMILES/sequence strings (when available) so
        # downstream scripts (generate_report.py) can trace a prediction
        # back to the actual molecule/protein for explainability figures,
        # without needing to re-search the original multi-GB CSV.
        if df is not None:
            data["smiles"] = df["compound_iso_smiles"].values[indices]
            data["sequence"] = df["target_sequence"].values[indices]
        out_path = os.path.join(out_dir, f"{name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {name} split ({len(indices)} rows) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess BindingDB DTI dataset")
    parser.add_argument("--csv", required=True, help="Path to bindingdb_highconf_pKd_clean.csv")
    parser.add_argument("--out", default="data/", help="Output directory for processed splits")
    parser.add_argument("--sample_frac", type=float, default=None,
                         help="Optional fraction of the data to use (e.g. 0.1 for a quick run)")
    args = parser.parse_args()

    mol_feats, prot_feats, labels, df = build_dataset(args.csv, sample_frac=args.sample_frac)
    split_and_save(mol_feats, prot_feats, labels, args.out, df=df)

    print("\nDataset summary:")
    print(f"  total usable pairs : {len(labels)}")
    print(f"  unique compounds   : {df['compound_iso_smiles'].nunique()}")
    print(f"  unique proteins    : {df['target_sequence'].nunique()}")
    print(f"  affinity (pKd) range: {labels.min():.2f} - {labels.max():.2f}, mean {labels.mean():.2f}")


if __name__ == "__main__":
    main()
