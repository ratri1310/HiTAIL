"""
HSPDL evaluation. Extends the stratified-by-rarity-bucket framework with
macro-F1, micro-F1, and hit@k on top of P@k/nDCG/PSP@k/PSnDCG@k.
"""

import torch
import torch.nn.functional as F
import numpy as np


def precision_at_k(scores, labels, k):
    topk = scores.topk(k, dim=1).indices
    hits = torch.gather(labels, 1, topk).sum(dim=1)
    return (hits / k).mean().item()


def hit_at_k(scores, labels, k):
    """Fraction of documents where AT LEAST ONE true label is in the top-k."""
    topk = scores.topk(k, dim=1).indices
    hits = torch.gather(labels, 1, topk).sum(dim=1)
    return (hits > 0).float().mean().item()


def _dcg_at_k(gains, k):
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=gains.device).float())
    return (gains * discounts).sum(dim=1)


def ndcg_at_k(scores, labels, k):
    topk_idx = scores.topk(k, dim=1).indices
    gains = torch.gather(labels, 1, topk_idx).float()
    dcg = _dcg_at_k(gains, k)
    num_true = labels.sum(dim=1)
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=scores.device).float())
    idcg = torch.zeros_like(dcg)
    for i in range(k):
        idcg = idcg + discounts[i] * (num_true > i).float()
    ndcg = torch.where(idcg > 0, dcg / idcg.clamp(min=1e-8), torch.zeros_like(dcg))
    return ndcg.mean().item()


def inverse_propensity(label_freq, A=0.55, B=1.5, num_train=None):
    N = num_train or int(label_freq.sum().item())
    C = (np.log(N) - 1) * ((B + 1) ** A)
    inv_prop = 1 + C * (label_freq.cpu().numpy() + 1) ** (-A)
    return torch.tensor(inv_prop, dtype=torch.float32, device=label_freq.device)


def psp_at_k(scores, labels, label_freq, k, A=0.55, B=1.5, num_train=None):
    inv_prop = inverse_propensity(label_freq, A, B, num_train)
    topk_idx = scores.topk(k, dim=1).indices
    topk_true = torch.gather(labels, 1, topk_idx)
    topk_inv_prop = torch.gather(inv_prop.unsqueeze(0).expand(labels.size(0), -1), 1, topk_idx)
    weighted_hits = (topk_true * topk_inv_prop).sum(dim=1)
    return (weighted_hits / k).mean().item()


def psndcg_at_k(scores, labels, label_freq, k, A=0.55, B=1.5, num_train=None):
    inv_prop = inverse_propensity(label_freq, A, B, num_train)
    topk_idx = scores.topk(k, dim=1).indices
    topk_true = torch.gather(labels, 1, topk_idx)
    topk_inv_prop = torch.gather(inv_prop.unsqueeze(0).expand(labels.size(0), -1), 1, topk_idx)
    gains = topk_true * topk_inv_prop
    dcg = _dcg_at_k(gains, k)
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=scores.device).float())
    true_inv_prop = labels * inv_prop.unsqueeze(0)
    sorted_true_inv_prop, _ = true_inv_prop.sort(dim=1, descending=True)
    ideal_topk = sorted_true_inv_prop[:, :k]
    idcg = (ideal_topk * discounts).sum(dim=1)
    psndcg = torch.where(idcg > 0, dcg / idcg.clamp(min=1e-8), torch.zeros_like(dcg))
    return psndcg.mean().item()


def macro_micro_f1(scores, labels, threshold=0.5):
    """
    NEW. macro-F1: per-label F1, averaged equally across labels (rare labels
    count the same as common ones). micro-F1: aggregate TP/FP/FN across ALL
    labels first, then one F1 -- dominated by common labels, same imbalance
    caveat as raw aggregate P@k.
    """
    preds = (torch.sigmoid(scores) >= threshold).float()
    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)

    precision = tp / (tp + fp).clamp(min=1e-8)
    recall = tp / (tp + fn).clamp(min=1e-8)
    f1_per_label = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    has_support = (tp + fn) > 0  # only score labels that actually appear in this eval set
    macro_f1 = f1_per_label[has_support].mean().item() if has_support.sum() > 0 else 0.0

    tp_sum, fp_sum, fn_sum = tp.sum(), fp.sum(), fn.sum()
    micro_p = tp_sum / (tp_sum + fp_sum).clamp(min=1e-8)
    micro_r = tp_sum / (tp_sum + fn_sum).clamp(min=1e-8)
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r).clamp(min=1e-8)).item()

    return macro_f1, micro_f1


def evaluate_all_metrics(scores, labels, label_freq, k_values=(1, 3, 5)):
    """Single flat dict of every metric -- simple, matches 'keep it simple'."""
    out = {}
    for k in k_values:
        if scores.size(1) >= k:
            out[f"P@{k}"] = precision_at_k(scores, labels, k)
            out[f"hit@{k}"] = hit_at_k(scores, labels, k)
            out[f"nDCG@{k}"] = ndcg_at_k(scores, labels, k)
            out[f"PSP@{k}"] = psp_at_k(scores, labels, label_freq, k)
            out[f"PSnDCG@{k}"] = psndcg_at_k(scores, labels, label_freq, k)
    macro_f1, micro_f1 = macro_micro_f1(scores, labels)
    out["macro_F1"] = macro_f1
    out["micro_F1"] = micro_f1
    return out
