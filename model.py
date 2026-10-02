"""
model.py
--------
Multimodal Drug-Target Interaction (DTI) model -- now a small "model zoo"
so you can systematically compare architecture choices for your paper's
ablation studies, rather than committing to a single design up front.

Configurable axes:

    prot_encoder_type ("cnn" | "bilstm"):
        "cnn"    - multi-scale 1D convolutions over the residue embedding
                   sequence (fast, good local-motif detector).
        "bilstm" - bidirectional LSTM over the residue embedding sequence
                   (captures longer-range dependencies, slower to train).

    fusion_type ("cross_attention" | "concat" | "bilinear" |
                 "mol_only" | "protein_only"):
        "cross_attention" - molecule attends over per-residue protein
                             representations. Doubles as the explainability
                             mechanism (attention weights).
        "concat"          - naive baseline: concatenate pooled mol + protein
                             embeddings, no interaction term.
        "bilinear"        - low-rank bilinear pooling between pooled mol and
                             protein embeddings, a classic multimodal fusion
                             technique that captures multiplicative
                             interactions without full cross-attention cost.
        "mol_only"        - ignores the protein branch entirely. Ablation to
                             show what the protein modality contributes.
        "protein_only"    - ignores the molecule branch entirely. Ablation to
                             show what the molecular modality contributes.

Every configuration exposes the same forward(mol_fp, prot_seq) -> (pred, attn)
interface. `attn` is a real per-residue attention vector only when
fusion_type="cross_attention"; otherwise it's zeros, so downstream code
(train.py, evaluate.py) never needs to branch on fusion type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_PROT_ENCODERS = ("cnn", "bilstm")
VALID_FUSIONS = ("cross_attention", "concat", "bilinear", "mol_only", "protein_only")


class MolecularEncoder(nn.Module):
    """Encodes a Morgan fingerprint bit-vector into a dense embedding."""

    def __init__(self, input_dim=1024, hidden_dim=256, embed_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)  # (batch, embed_dim)


class ProteinEncoderCNN(nn.Module):
    """Multi-scale 1D-CNN protein encoder. Returns per-residue representation
    (for attention) plus a mean-pooled global representation.
    """

    def __init__(self, vocab_size=26, embed_dim=128, num_filters=128,
                 kernel_sizes=(3, 5, 7), dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.proj = nn.Linear(num_filters * len(kernel_sizes), embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)                     # (batch, embed_dim, seq_len)
        conv_outs = [F.relu(conv(emb)) for conv in self.convs]
        concat = torch.cat(conv_outs, dim=1).transpose(1, 2)         # (batch, seq_len, num_filters*len(kernels))
        residue_repr = self.dropout(F.relu(self.proj(concat)))       # (batch, seq_len, embed_dim)
        pooled = residue_repr.mean(dim=1)
        return residue_repr, pooled


class ProteinEncoderBiLSTM(nn.Module):
    """Bidirectional LSTM protein encoder. Captures longer-range sequence
    dependencies than the CNN at the cost of slower training. Returns
    per-residue hidden states plus a pooled (mean over time) representation.
    """

    def __init__(self, vocab_size=26, embed_dim=128, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                             bidirectional=True, num_layers=1)
        # hidden_dim*2 (bidirectional) -> embed_dim, so output dim matches
        # the CNN encoder and downstream fusion code doesn't need to know
        # which protein encoder produced it.
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x)                       # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(emb)                   # (batch, seq_len, hidden_dim*2)
        residue_repr = self.dropout(F.relu(self.proj(lstm_out)))  # (batch, seq_len, embed_dim)
        pooled = residue_repr.mean(dim=1)
        return residue_repr, pooled


class CrossAttentionFusion(nn.Module):
    """Molecule attends over per-residue protein representations. Attention
    weights double as the explainability signal.
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, mol_embed, residue_repr, mask=None):
        query = self.query_proj(mol_embed).unsqueeze(1)
        keys = self.key_proj(residue_repr)
        values = self.value_proj(residue_repr)

        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, values).squeeze(1)
        return context, attn_weights.squeeze(1)


class BilinearFusion(nn.Module):
    """Low-rank bilinear pooling between pooled molecule and protein
    embeddings. Captures multiplicative (rather than purely additive)
    cross-modal interactions -- a well-established multimodal fusion
    technique (see e.g. Multimodal Compact Bilinear Pooling) that sits
    architecturally between naive concatenation and full cross-attention.
    """

    def __init__(self, embed_dim=128, rank=64):
        super().__init__()
        self.mol_proj = nn.Linear(embed_dim, rank)
        self.prot_proj = nn.Linear(embed_dim, rank)
        self.out_proj = nn.Linear(rank, embed_dim)

    def forward(self, mol_pooled, prot_pooled):
        m = self.mol_proj(mol_pooled)
        p = self.prot_proj(prot_pooled)
        interaction = m * p              # element-wise product = low-rank bilinear term
        return self.out_proj(interaction)


class MultimodalDTIModel(nn.Module):
    """Configurable DTI model. See module docstring for the full set of
    prot_encoder_type / fusion_type options.
    """

    def __init__(self, fp_dim=1024, vocab_size=26, embed_dim=128,
                 hidden_dim=256, dropout=0.3,
                 prot_encoder_type="cnn", fusion_type="cross_attention"):
        super().__init__()
        if prot_encoder_type not in VALID_PROT_ENCODERS:
            raise ValueError(f"Unknown prot_encoder_type: {prot_encoder_type!r}. "
                              f"Expected one of {VALID_PROT_ENCODERS}.")
        if fusion_type not in VALID_FUSIONS:
            raise ValueError(f"Unknown fusion_type: {fusion_type!r}. "
                              f"Expected one of {VALID_FUSIONS}.")

        self.prot_encoder_type = prot_encoder_type
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim

        self.mol_encoder = MolecularEncoder(input_dim=fp_dim, embed_dim=embed_dim, dropout=dropout)

        if prot_encoder_type == "cnn":
            self.prot_encoder = ProteinEncoderCNN(vocab_size=vocab_size, embed_dim=embed_dim, dropout=dropout)
        else:  # "bilstm"
            self.prot_encoder = ProteinEncoderBiLSTM(vocab_size=vocab_size, embed_dim=embed_dim, dropout=dropout)

        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(embed_dim=embed_dim)
            fused_dim = embed_dim * 3   # [mol_embed, attention_context, pooled_protein]
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(embed_dim=embed_dim)
            fused_dim = embed_dim * 3   # [mol_embed, bilinear_interaction, pooled_protein]
        elif fusion_type == "concat":
            self.fusion = None
            fused_dim = embed_dim * 2   # [mol_embed, pooled_protein]
        elif fusion_type == "mol_only":
            self.fusion = None
            fused_dim = embed_dim       # [mol_embed] only -- protein branch unused
        else:  # "protein_only"
            self.fusion = None
            fused_dim = embed_dim       # [pooled_protein] only -- molecule branch unused

        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, mol_fp, prot_seq):
        mol_embed = self.mol_encoder(mol_fp)
        residue_repr, pooled_protein = self.prot_encoder(prot_seq)

        attn_weights = torch.zeros(prot_seq.shape[0], prot_seq.shape[1], device=prot_seq.device)

        if self.fusion_type == "cross_attention":
            mask = (prot_seq != 0).float()
            context, attn_weights = self.fusion(mol_embed, residue_repr, mask=mask)
            fused = torch.cat([mol_embed, context, pooled_protein], dim=-1)
        elif self.fusion_type == "bilinear":
            interaction = self.fusion(mol_embed, pooled_protein)
            fused = torch.cat([mol_embed, interaction, pooled_protein], dim=-1)
        elif self.fusion_type == "concat":
            fused = torch.cat([mol_embed, pooled_protein], dim=-1)
        elif self.fusion_type == "mol_only":
            fused = mol_embed
        else:  # "protein_only"
            fused = pooled_protein

        pred = self.head(fused).squeeze(-1)
        return pred, attn_weights


if __name__ == "__main__":
    # Shape + parameter-count sanity check across the whole model zoo
    batch_size = 4
    dummy_fp = torch.rand(batch_size, 1024)
    dummy_seq = torch.randint(0, 26, (batch_size, 1000))
    dummy_seq[:, 500:] = 0  # simulate padding

    for prot_enc in VALID_PROT_ENCODERS:
        for fusion in VALID_FUSIONS:
            model = MultimodalDTIModel(prot_encoder_type=prot_enc, fusion_type=fusion)
            pred, attn = model(dummy_fp, dummy_seq)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[prot={prot_enc:7s} fusion={fusion:14s}] "
                  f"pred={tuple(pred.shape)} attn={tuple(attn.shape)} params={n_params:,}")
