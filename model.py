"""
HSPDL model — deliberately simpler than the old architecture. No e_y, no
mu_y, no frequency gate, no p_A/p_P prototype tables. Only p_S (semantic
type prototypes) is learned; parent/grandparent influence the model only
through L_SSL's pair construction and L_dist's distribution weighting,
never as trainable embedding tables.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

ENCODER_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
EMBED_DIM = 768


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, h):
        return F.normalize(self.net(h), dim=-1)


class SemanticTypePrototypes(nn.Module):
    """The ONLY trainable prototype table in this design."""

    def __init__(self, num_semantic_types: int, dim: int):
        super().__init__()
        self.p_S = nn.Embedding(num_semantic_types, dim)
        nn.init.normal_(self.p_S.weight, std=0.02)

    def normalized(self):
        return F.normalize(self.p_S.weight, dim=-1)


class HSPDLModel(nn.Module):
    def __init__(self, num_labels: int, num_semantic_types: int, proj_dim: int = 256,
                 use_semantic_compatibility: bool = False, beta: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER_NAME)
        self.proj = ProjectionHead(EMBED_DIM, proj_dim)
        self.semantic_protos = SemanticTypePrototypes(num_semantic_types, proj_dim)

        self.classifier = nn.Linear(proj_dim, num_labels)
        self.use_semantic_compatibility = use_semantic_compatibility
        self.beta = beta

        # label_to_semantic_idx: which semantic-type row(s) each label maps to,
        # needed only if use_semantic_compatibility=True. Set via set_label_semantic_map().
        self.register_buffer("label_semantic_idx", torch.full((num_labels,), -1, dtype=torch.long))

    def set_label_semantic_map(self, label_semantic_idx: torch.Tensor):
        """label_semantic_idx: [num_labels] LongTensor, semantic-type row index
        per label (mean-pooled externally if a label has multiple semantic
        types -- see build_label_semantic_compat() in losses.py)."""
        self.label_semantic_idx.copy_(label_semantic_idx)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_i = out.last_hidden_state[:, 0]  # [CLS]
        return self.proj(h_i)

    def classify(self, z_i):
        """
        Default: standard classifier, y_hat = sigma(W z_i + b).
        If use_semantic_compatibility: adds beta * z_i . p_bar_S(y) to each
        label's logit (Step 9's optional enhancement) -- the classifier
        remains label-specific; semantic compatibility is an additive prior,
        not a replacement scoring mechanism.
        """
        logits = self.classifier(z_i)
        if self.use_semantic_compatibility:
            protos = self.semantic_protos.normalized()  # [num_semantic_types, dim]
            # mean-pooled semantic compatibility per label, precomputed as a
            # [num_labels, dim] buffer for efficiency -- see build_label_semantic_compat_matrix()
            logits = logits + self.beta * (z_i @ self._label_semantic_compat_matrix.T)
        return logits

    def set_label_semantic_compat_matrix(self, matrix: torch.Tensor):
        """matrix: [num_labels, proj_dim], precomputed mean-pooled p_bar_S(y)
        per label (recomputed each epoch since p_S is still training)."""
        self._label_semantic_compat_matrix = matrix
