"""
explainability.py
------------------
Generates explainable insights from the trained multimodal model by
extracting and visualizing the cross-attention weights the model places
over the protein sequence for a specific drug-target pair. High-attention
residues are a hypothesis about which parts of the binding pocket / protein
region the model considers most relevant to the predicted affinity -- this
is the "explainable AI" component of the paper's contribution.

Usage:
    python explainability.py --checkpoint checkpoints/best_model.pt \
        --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
        --sequence "MENDKV..." \
        --out results/attention_case_study.png

Note: this produces a per-example case study. For the paper, run this on
a handful of representative test-set examples (e.g. best-predicted,
worst-predicted, and a well-known drug-target pair) rather than the whole
test set.
"""

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import MultimodalDTIModel, VALID_PROT_ENCODERS
from train import get_device
from data_preprocessing import smiles_to_fingerprint, sequence_to_int_array, PROTEIN_MAX_LEN


def get_attention_for_pair(model, smiles: str, sequence: str, device):
    """Runs a single forward pass and returns the predicted affinity plus
    the per-residue attention weight vector (trimmed to the real sequence
    length, i.e. padding removed).
    """
    model.eval()
    fp = smiles_to_fingerprint(smiles)
    seq_encoded = sequence_to_int_array(sequence)

    fp_tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device)
    seq_tensor = torch.tensor(seq_encoded, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        pred, attn_weights = model(fp_tensor, seq_tensor)

    attn = attn_weights.squeeze(0).cpu().numpy()
    real_len = min(len(sequence), PROTEIN_MAX_LEN)
    attn = attn[:real_len]

    return float(pred.item()), attn


def plot_attention(sequence: str, attn: np.ndarray, out_path: str, top_k: int = 15):
    real_len = len(attn)
    seq_trimmed = sequence[:real_len]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(attn, linewidth=1)
    ax.fill_between(range(real_len), attn, alpha=0.3)
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Attention weight")
    ax.set_title("Model attention over protein sequence for this drug-target pair")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved attention plot -> {out_path}")

    top_positions = np.argsort(attn)[::-1][:top_k]
    top_positions_sorted = sorted(top_positions.tolist())
    print(f"\nTop-{top_k} attended residues (position: amino acid, weight):")
    for pos in sorted(top_positions.tolist(), key=lambda p: -attn[p]):
        aa = seq_trimmed[pos] if pos < len(seq_trimmed) else "?"
        print(f"  position {pos:>5}: {aa}  (weight={attn[pos]:.4f})")

    return top_positions_sorted


def main():
    parser = argparse.ArgumentParser(description="Explain a single drug-target prediction via attention")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--smiles", required=True, help="Isomeric SMILES of the drug/ligand")
    parser.add_argument("--sequence", required=True, help="Amino-acid sequence of the target protein")
    parser.add_argument("--out", default="results/attention_case_study.png")
    parser.add_argument("--prot_encoder", choices=VALID_PROT_ENCODERS, default="cnn",
                         help="Must match the architecture the checkpoint was trained with")
    parser.add_argument("--hidden_dim", type=int, default=256,
                         help="Must match the --hidden_dim the checkpoint was trained with")
    parser.add_argument("--embed_dim", type=int, default=128,
                         help="Must match the --embed_dim the checkpoint was trained with")
    args = parser.parse_args()

    # Explainability via attention only makes sense for the cross_attention
    # fusion mode -- concat/bilinear/mol_only/protein_only checkpoints would
    # just produce all-zero "attention" here, so we don't expose --fusion.
    device = get_device()
    model = MultimodalDTIModel(prot_encoder_type=args.prot_encoder, fusion_type="cross_attention",
                                hidden_dim=args.hidden_dim, embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    pred, attn = get_attention_for_pair(model, args.smiles, args.sequence, device)
    print(f"Predicted pKd: {pred:.3f}")

    plot_attention(args.sequence, attn, args.out)


if __name__ == "__main__":
    main()
