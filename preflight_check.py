"""
Phase 2 diagnostic -- the check that SHOULD run before every real training
job, not after. One forward pass on a FRESH (untrained) model, one real
batch, every loss term computed and printed side by side. This is exactly
the kind of check that would have caught embryology's L_dist magnitude
blowup in under a minute instead of after a 10-hour run.

Run this for EVERY domain before committing to a full training job.

Usage:
    python -m hspdl.preflight_check --domain embryology
    python -m hspdl.preflight_check --domain immunology
    python -m hspdl.preflight_check --domain neurology
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .config import DomainConfig
from .data import MeshDataset, collate_fn, build_tokenizer, load_label_map, load_label_freq, build_freq_tensor
from .hierarchy import load_hierarchy, build_label_index_maps
from .model import HSPDLModel
from .losses import (
    semantic_prototype_loss, surrogate_contrastive_loss,
    semantic_prototype_separation_loss,
    build_distribution_targets_weighted, pool_predictions_to_ancestor,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg = DomainConfig.build(args.domain)
    label_map = load_label_map(dcfg.label_map_path)
    label_freq = load_label_freq(dcfg.label_freq_path)

    from .config import GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT
    hierarchy = load_hierarchy(
        GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT, dcfg.semantic_type_path,
        dcfg.legacy_parent_map_path, dcfg.legacy_grandparent_map_path, label_map,
    )
    label_to_semantic, label_to_parents, label_to_gp = build_label_index_maps(label_map, hierarchy)

    with open(dcfg.omega_y_path) as f:
        omega_y_raw = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    omega_y_lookup = {idx: omega_y_raw.get(uid, 0.0) for idx, uid in inv_label_map.items()}
    label_freq_lookup = {idx: label_freq.get(uid, 1) for idx, uid in inv_label_map.items()}

    tokenizer = build_tokenizer()
    train_ds = MeshDataset(os.path.join(dcfg.data_dir, "train.jsonl"), label_map, tokenizer)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    freq_all = build_freq_tensor(label_map, label_freq, device)
    parent_idx_all = torch.tensor([hierarchy.parent_idx[i] for i in range(len(label_map))],
                                    dtype=torch.long, device=device)
    grandparent_idx_all = torch.tensor([hierarchy.grandparent_idx[i] for i in range(len(label_map))],
                                         dtype=torch.long, device=device)

    model = HSPDLModel(
        num_labels=len(label_map), num_semantic_types=hierarchy.num_semantic_types,
        proj_dim=args.proj_dim,
    ).to(device)
    model.eval()

    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    label_ids_batch = batch["label_ids"]

    with torch.no_grad():
        z_i = model.encode(input_ids, attention_mask)
        logits = model.classify(z_i)

        l_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()

        semantic_idx_batch = [
            set().union(*[label_to_semantic.get(lid, set()) for lid in labels_i]) if labels_i else set()
            for labels_i in label_ids_batch
        ]
        l_st = semantic_prototype_loss(z_i, semantic_idx_batch, model.semantic_protos, 0.1).item()

        l_ssl = surrogate_contrastive_loss(
            z_i, label_ids_batch, label_freq_lookup,
            label_to_semantic, label_to_parents, label_to_gp, 0.1,
        ).item()

        pred_probs = torch.sigmoid(logits)
        q_L, q_A1, q_A2 = build_distribution_targets_weighted(
            label_ids_batch, len(label_map), omega_y_lookup, parent_idx_all, grandparent_idx_all,
            hierarchy.num_parent_rows, hierarchy.num_grandparent_rows, device,
        )
        pred_L = pred_probs / pred_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pred_A1 = pool_predictions_to_ancestor(pred_probs, parent_idx_all, hierarchy.num_parent_rows)
        pred_A2 = pool_predictions_to_ancestor(pred_probs, grandparent_idx_all, hierarchy.num_grandparent_rows)
        kl_L = torch.nn.functional.kl_div(torch.log(pred_L.clamp(min=1e-8)), q_L, reduction="batchmean").item()
        kl_A1 = torch.nn.functional.kl_div(torch.log(pred_A1.clamp(min=1e-8)), q_A1, reduction="batchmean").item()
        kl_A2 = torch.nn.functional.kl_div(torch.log(pred_A2.clamp(min=1e-8)), q_A2, reduction="batchmean").item()

        l_sep = semantic_prototype_separation_loss(model.semantic_protos, 0.5).item()

    print(f"\n=== Pre-flight loss magnitude check: {args.domain} ({len(label_map)} labels), FRESH model ===\n")
    print(f"  L_BCE          {l_bce:12.4f}")
    print(f"  L_ST           {l_st:12.4f}")
    print(f"  L_SSL          {l_ssl:12.4f}")
    norm_L = torch.log(torch.tensor(float(len(label_map)) + 1))
    norm_A1 = torch.log(torch.tensor(float(hierarchy.num_parent_rows) + 1))
    norm_A2 = torch.log(torch.tensor(float(hierarchy.num_grandparent_rows) + 1))
    kl_L_norm = kl_L / norm_L.item()
    kl_A1_norm = kl_A1 / norm_A1.item()
    kl_A2_norm = kl_A2 / norm_A2.item()

    print(f"  L_dist (q_L)   raw={kl_L:8.4f}  normalized={kl_L_norm:8.4f}   <-- leaf-level")
    print(f"  L_dist (q_A1)  raw={kl_A1:8.4f}  normalized={kl_A1_norm:8.4f}   <-- parent-level")
    print(f"  L_dist (q_A2)  raw={kl_A2:8.4f}  normalized={kl_A2_norm:8.4f}   <-- grandparent-level")
    print(f"  L_dist (total) raw={kl_L + kl_A1 + kl_A2:8.4f}  normalized={kl_L_norm + kl_A1_norm + kl_A2_norm:8.4f}")
    print(f"  L_sep^S        {l_sep:12.4f}")

    values = {"L_BCE": l_bce, "L_ST": l_st, "L_SSL": l_ssl,
              "L_dist_normalized_total": kl_L_norm + kl_A1_norm + kl_A2_norm, "L_sep": l_sep}
    max_term, max_val = max(values.items(), key=lambda kv: abs(kv[1]))
    min_term, min_val = min(values.items(), key=lambda kv: abs(kv[1]))
    ratio = abs(max_val) / max(abs(min_val), 1e-8)
    print(f"\n  [post-fix] largest/smallest term ratio: {max_term}={max_val:.2f} vs {min_term}={min_val:.2f} "
          f"({ratio:.1f}x)")
    if ratio > 20:
        print(f"  FLAG: >20x spread between loss terms on a FRESH model -- check before training, "
              f"this term will likely dominate the shared encoder's gradient.")


if __name__ == "__main__":
    main()