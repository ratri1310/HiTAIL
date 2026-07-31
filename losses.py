"""


Deliberately does NOT include: cl_ancestor, mu_y_consistency, prototype
alignment against e_y -- all removed, per the redesign that resolved the
representation-collapse crisis.
"""

import torch
import torch.nn.functional as F


# ============================================================

# ============================================================

def semantic_prototype_loss(z, semantic_idx_batch, semantic_protos, temperature: float = 0.1):
    """
    semantic_idx_batch: list of lists -- each document's set of semantic-type
    row indices (from its positive labels' S(y), unioned). Multi-positive
    softmax cross-entropy against the FULL semantic-type prototype table.
    """
    protos = semantic_protos.normalized()  # [num_ST, dim]
    sim = (z @ protos.T) / temperature      # [batch, num_ST]
    log_prob = F.log_softmax(sim, dim=1)

    losses = []
    for i, s_set in enumerate(semantic_idx_batch):
        if not s_set:
            continue
        losses.append(-log_prob[i, list(s_set)].mean())
    if not losses:
        return torch.tensor(0.0, device=z.device)
    return torch.stack(losses).mean()


# ============================================================
# Step 4/5 -- L_SSL: ontology-guided surrogate contrastive learning
# ============================================================

def categorize_pair(S_i, A1_i, A2_i, S_j, A1_j, A2_j):
    """The exact 4-way rule from Decision 1. Sets of hashable IDs (strings/ints)."""
    s_overlap = len(S_i & S_j) > 0
    a1_overlap = len(A1_i & A1_j) > 0
    a2_overlap = len(A2_i & A2_j) > 0
    if s_overlap and (a1_overlap or a2_overlap):
        return "positive"
    elif s_overlap and not a1_overlap and not a2_overlap:
        return "hard_negative"
    elif not s_overlap and not a1_overlap and not a2_overlap:
        return "easy_negative"
    else:
        return "ignore"


def build_document_ontology_sets(label_ids_batch, label_to_semantic, label_to_parents, label_to_gp):
    """
    For each document, union its true labels' S(y)/A1(y)/A2(y) into per-document
    sets S_i/A_i^1/A_i^2. label_to_* are dicts: label_idx -> set of IDs.
    """
    S, A1, A2 = [], [], []
    for labels in label_ids_batch:
        s_set, a1_set, a2_set = set(), set(), set()
        for lid in labels:
            s_set |= label_to_semantic.get(lid, set())
            a1_set |= label_to_parents.get(lid, set())
            a2_set |= label_to_gp.get(lid, set())
        S.append(s_set)
        A1.append(a1_set)
        A2.append(a2_set)
    return S, A1, A2


def surrogate_contrastive_loss(z, label_ids_batch, label_freq_lookup,
                                 label_to_semantic, label_to_parents, label_to_gp,
                                 temperature: float = 0.1, c: float = 2.0):
    """
   
    """
    import math
    batch = z.size(0)
    device = z.device
    sim = (z @ z.T) / temperature

    S, A1, A2 = build_document_ontology_sets(label_ids_batch, label_to_semantic,
                                               label_to_parents, label_to_gp)

    # rho_i: document rarity weight
    rho = []
    for labels in label_ids_batch:
        if not labels:
            rho.append(0.0)
            continue
        vals = [1.0 / math.log(label_freq_lookup.get(lid, 1) + c) for lid in labels]
        rho.append(sum(vals) / len(vals))

    total_loss = torch.tensor(0.0, device=device)
    n_active = 0

    for i in range(batch):
        positives, hard_negs, easy_negs = [], [], []
        for j in range(batch):
            if j == i:
                continue
            cat = categorize_pair(S[i], A1[i], A2[i], S[j], A1[j], A2[j])
            if cat == "positive":
                positives.append(j)
            elif cat == "hard_negative":
                hard_negs.append(j)
            elif cat == "easy_negative":
                easy_negs.append(j)
            # "ignore" -> excluded entirely, not added to any list

        if not positives:
            continue  # L_SSL^(i) = 0 if P_i is empty

        valid = positives + hard_negs + easy_negs
        valid_idx = torch.tensor(valid, device=device, dtype=torch.long)
        denom = torch.logsumexp(sim[i, valid_idx], dim=0)

        pos_idx = torch.tensor(positives, device=device, dtype=torch.long)
        per_pos_log_prob = sim[i, pos_idx] - denom
        anchor_loss = -per_pos_log_prob.mean()

        total_loss = total_loss + rho[i] * anchor_loss
        n_active += 1

    if n_active == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / n_active


# ============================================================
# Step 6/7 -- L_dist: hierarchy-aware distribution alignment
# ============================================================

def build_distribution_targets_weighted(label_ids_batch, num_labels, omega_y_lookup,
                                          parent_idx_all, grandparent_idx_all,
                                          num_a1, num_a2, device):
    """
    q_L, q_A1, q_A2 -- weighted by omega_y (NOT uniform, per the finalized
    design). parent_idx_all/grandparent_idx_all here are the SINGLE-VALUE
    prototype-row mappings used only for POOLING predicted mass (a
    label->row index lookup, distinct from the full polyhierarchy sets used
    in L_SSL's pair construction) -- see note in train.py about this
    resolving to one canonical row per level for the purpose of building a
    fixed-size predicted distribution to compare against.
    """
    batch = len(label_ids_batch)
    q_L = torch.zeros(batch, num_labels, device=device)
    q_A1 = torch.zeros(batch, num_a1, device=device)
    q_A2 = torch.zeros(batch, num_a2, device=device)

    for i, labels in enumerate(label_ids_batch):
        if not labels:
            continue
        weights = {lid: omega_y_lookup.get(lid, 0.0) for lid in labels}
        total_w = sum(weights.values())
        if total_w <= 0:
            continue
        for lid in labels:
            q_L[i, lid] = weights[lid] / total_w

        a1_weight = torch.zeros(num_a1, device=device)
        a2_weight = torch.zeros(num_a2, device=device)
        for lid in labels:
            p = parent_idx_all[lid].item()
            g = grandparent_idx_all[lid].item()
            if p >= 0:
                a1_weight[p] += weights[lid]
            if g >= 0:
                a2_weight[g] += weights[lid]
        if a1_weight.sum() > 0:
            q_A1[i] = a1_weight / a1_weight.sum()
        if a2_weight.sum() > 0:
            q_A2[i] = a2_weight / a2_weight.sum()

    return q_L, q_A1, q_A2


def pool_predictions_to_ancestor(pred_probs, ancestor_idx_for_label, num_ancestors):
    batch = pred_probs.size(0)
    device = pred_probs.device
    pooled = torch.zeros(batch, num_ancestors, device=device)
    valid = ancestor_idx_for_label >= 0
    if valid.sum() == 0:
        return pooled
    idx = ancestor_idx_for_label[valid].unsqueeze(0).expand(batch, -1)
    vals = pred_probs[:, valid]
    pooled.scatter_add_(1, idx, vals)
    return pooled / pooled.sum(dim=1, keepdim=True).clamp(min=1e-8)


def distribution_alignment_loss(logits, label_ids_batch, omega_y_lookup,
                                  parent_idx_all, grandparent_idx_all, num_a1, num_a2,
                                  alpha_L: float = 1.0, alpha_1: float = 1.0, alpha_2: float = 1.0):
    """
    FIXED (was: raw kl_L + kl_A1 + kl_A2). Confirmed via cross-domain preflight
    checks (immunology + embryology) that the raw sum lets the leaf-level term
    dominate by ~15-25x purely because it's computed over the largest vocabulary
    (thousands of labels vs. hundreds of ancestor nodes) -- nothing to do with
    which signal is actually more informative. Fix: divide each level's KL by
    log(|V_level|+1) before summing, so all three are comparable in scale
    regardless of vocabulary size. Verified: this brings all three components
    into a tight, nearly domain-invariant band (~0.63-0.76) on both immunology
    and embryology, instead of a raw ~5-7x spread.
    """
    device = logits.device
    num_labels = logits.size(1)
    pred_probs = torch.sigmoid(logits)

    q_L, q_A1, q_A2 = build_distribution_targets_weighted(
        label_ids_batch, num_labels, omega_y_lookup, parent_idx_all, grandparent_idx_all,
        num_a1, num_a2, device
    )

    pred_L = pred_probs / pred_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    pred_A1 = pool_predictions_to_ancestor(pred_probs, parent_idx_all, num_a1)
    pred_A2 = pool_predictions_to_ancestor(pred_probs, grandparent_idx_all, num_a2)

    kl_L = F.kl_div(torch.log(pred_L.clamp(min=1e-8)), q_L, reduction="batchmean")
    kl_A1 = F.kl_div(torch.log(pred_A1.clamp(min=1e-8)), q_A1, reduction="batchmean")
    kl_A2 = F.kl_div(torch.log(pred_A2.clamp(min=1e-8)), q_A2, reduction="batchmean")

    norm_L = torch.log(torch.tensor(float(num_labels) + 1, device=device))
    norm_A1 = torch.log(torch.tensor(float(num_a1) + 1, device=device))
    norm_A2 = torch.log(torch.tensor(float(num_a2) + 1, device=device))

    return (alpha_L * kl_L / norm_L) + (alpha_1 * kl_A1 / norm_A1) + (alpha_2 * kl_A2 / norm_A2)


# ============================================================
# Step 8 -- 
# ============================================================

def semantic_prototype_separation_loss(semantic_protos, gamma: float = 0.5):
    """
    FIXED, v2 (v1 replaced the inactive hinge margin with an orthogonality
    regularizer, but left it UNNORMALIZED -- caught via preflight_check on
    real data: raw value was 55.7-57.0 in both domains, confirmed to be
    ~n(n-1)/dim (its expected value under random init: measured/theoretical
    ratio = 0.999), meaning even with the existing lambda_R=0.1 applied it
    would still be the LARGEST weighted term in the full objective, ~4x
    L_ST's weighted contribution -- reproducing the exact same single-term-
    domination failure just diagnosed and fixed for L_dist.

    v2: normalize by n(n-1)/dim, its theoretical expected value at random
    init, so the loss starts at ~1.0 regardless of prototype count or
    embedding dimension, and decreases as training actually pushes toward
    orthogonality -- same principle as L_dist's log(|V|+1) normalization,
    applied correctly this time.

    `gamma` kept as an argument for interface compatibility but unused;
    left in place rather than silently dropped from the call signature.
    """
    protos = semantic_protos.normalized()  # [num_ST, dim]
    n = protos.size(0)
    dim = protos.size(1)
    if n < 2:
        return torch.tensor(0.0, device=protos.device)
    gram = protos @ protos.T  # [num_ST, num_ST]
    identity = torch.eye(n, device=protos.device)
    raw = ((gram - identity) ** 2).sum()
    expected_at_random_init = (n * (n - 1)) / dim
    return raw / expected_at_random_init
