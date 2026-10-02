"""
embedding_visualization.py
----------------------------
Extracts the model's LEARNED internal representations (the dense embeddings
it computes for each molecule and protein, right before fusion) and
visualizes them in 2D using t-SNE, colored by binding affinity.

Why this matters for your paper: this is a form of "unsupervised" evidence
that the model learned meaningful chemistry, not just numbers. If molecules
with similar binding affinity cluster together in this learned space (even
though the model was never explicitly told to cluster anything -- affinity
was only used as a regression target), that's a genuinely interesting,
visual, defensible result almost no basic DTI project shows.

Usage:
    python embedding_visualization.py --data_dir data/ \
        --checkpoint checkpoints/bilstm_cross_attention_large/best_model.pt \
        --prot_encoder bilstm --fusion cross_attention --hidden_dim 512 --embed_dim 256
"""

import argparse
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from model import MultimodalDTIModel, VALID_PROT_ENCODERS, VALID_FUSIONS
from train import get_device


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of learned embeddings")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prot_encoder", choices=VALID_PROT_ENCODERS, required=True)
    parser.add_argument("--fusion", choices=VALID_FUSIONS, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=3000,
                         help="How many test examples to visualize (t-SNE gets slow/unreadable "
                              "beyond a few thousand points, so we subsample)")
    parser.add_argument("--out_dir", default="results/embeddings/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    model = MultimodalDTIModel(prot_encoder_type=args.prot_encoder, fusion_type=args.fusion,
                                hidden_dim=args.hidden_dim, embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with open(os.path.join(args.data_dir, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)

    n = min(args.n_samples, len(test_data["labels"]))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(test_data["labels"]), size=n, replace=False)

    mol_feats = torch.tensor(test_data["mol_feats"][idx], dtype=torch.float32)
    prot_feats = torch.tensor(test_data["prot_feats"][idx], dtype=torch.long)
    labels = np.asarray(test_data["labels"])[idx]

    print(f"Extracting embeddings for {n} test examples ...")
    mol_embeds, prot_embeds = [], []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, n, batch_size):
            fp_batch = mol_feats[i:i + batch_size].to(device)
            seq_batch = prot_feats[i:i + batch_size].to(device)

            mol_embed = model.mol_encoder(fp_batch)
            _, pooled_protein = model.prot_encoder(seq_batch)

            mol_embeds.append(mol_embed.cpu().numpy())
            prot_embeds.append(pooled_protein.cpu().numpy())

    mol_embeds = np.concatenate(mol_embeds, axis=0)
    prot_embeds = np.concatenate(prot_embeds, axis=0)

    for name, embeds in [("molecule", mol_embeds), ("protein", prot_embeds)]:
        print(f"\nRunning t-SNE on {name} embeddings (this can take a minute) ...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n // 4))
        coords = tsne.fit_transform(embeds)

        plt.figure(figsize=(8, 7))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis",
                               alpha=0.6, s=12)
        plt.colorbar(scatter, label="Actual pKd")
        plt.title(f"t-SNE of Learned {name.capitalize()} Embeddings\n(colored by binding affinity)")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"tsne_{name}_embeddings.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    print(f"\nAll embedding visualizations saved to: {args.out_dir}")
    print("\nWhat to look for: if points of similar color (similar pKd) form loose "
          "clusters or gradients rather than being randomly scattered, that's evidence "
          "the model organized its internal representation meaningfully around binding "
          "strength -- a good figure for your paper's 'what did the model learn' discussion.")


if __name__ == "__main__":
    main()
