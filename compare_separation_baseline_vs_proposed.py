"""
Computes per-semantic-type silhouette score for BOTH baseline (Stage 0) and
proposed model, on the SAME documents -- direct answer to the question the
single-model ranking couldn't answer: is the (weak) organization we found
actually better than baseline, or is baseline just as good/bad?

Usage:
    python -m hspdl.compare_separation_baseline_vs_proposed --domain immunology --proposed-stage 5
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import silhouette_samples
from torch.utils.data import DataLoader

from .config import DomainConfig, PROJECT_ROOT
from .data import MeshDataset, collate_fn, build_tokenizer, load_label_map
from .hierarchy import load_hierarchy
from .model import HSPDLModel


def compute_per_type_silhouette(domain, stage, kept_types, type_list, label_map,
                                  hierarchy, loader, device, proj_dim):
    use_semantic_compat = (stage == 5)
    model = HSPDLModel(
        num_labels=len(label_map), num_semantic_types=hierarchy.num_semantic_types,
        proj_dim=proj_dim, use_semantic_compatibility=use_semantic_compat,
    ).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, "runs", f"{domain}_stage{stage}", "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            z = model.encode(input_ids, attention_mask)
            embeddings.append(z.cpu())
    X = torch.cat(embeddings).numpy()

    type_to_int = {t: i for i, t in enumerate(type_list)}
    cluster_labels = np.array([type_to_int[t] for t in kept_types])
    sample_scores = silhouette_samples(X, cluster_labels)

    per_type_score = {}
    for t in type_list:
        mask = cluster_labels == type_to_int[t]
        per_type_score[t] = sample_scores[mask].mean()
    overall = sample_scores.mean()
    return per_type_score, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--proposed-stage", type=int, default=5)
    p.add_argument("--min-doc-count", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--proj-dim", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg = DomainConfig.build(args.domain)
    label_map = load_label_map(dcfg.label_map_path)
    from .config import GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT
    hierarchy = load_hierarchy(
        GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT, dcfg.semantic_type_path,
        dcfg.legacy_parent_map_path, dcfg.legacy_grandparent_map_path, label_map,
    )

    tokenizer = build_tokenizer()
    test_ds = MeshDataset(os.path.join(dcfg.data_dir, "test.jsonl"), label_map, tokenizer)

    with open(dcfg.semantic_type_path) as f:
        raw_semantic = json.load(f)

    def all_semantic_types(uid):
        v = raw_semantic.get(uid)
        if isinstance(v, list):
            return v
        elif v:
            return [v]
        return []

    doc_type = []
    for rec in test_ds.records:
        st_counts = Counter()
        for u in rec["labels"]:
            for st in all_semantic_types(u):
                st_counts[st] += 1
        doc_type.append(st_counts.most_common(1)[0][0] if st_counts else None)

    type_counts = Counter(t for t in doc_type if t is not None)
    valid_types = {t for t, c in type_counts.items() if c >= args.min_doc_count}
    type_list = sorted(valid_types)

    keep_idx = [i for i, t in enumerate(doc_type) if t in valid_types]
    kept_types = [doc_type[i] for i in keep_idx]
    sub_ds = torch.utils.data.Subset(test_ds, keep_idx)
    loader = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Encoding {len(keep_idx)} documents with Stage 0 (baseline)...")
    baseline_scores, baseline_overall = compute_per_type_silhouette(
        args.domain, 0, kept_types, type_list, label_map, hierarchy, loader, device, args.proj_dim
    )
    print(f"Encoding {len(keep_idx)} documents with Stage {args.proposed_stage} (proposed)...")
    proposed_scores, proposed_overall = compute_per_type_silhouette(
        args.domain, args.proposed_stage, kept_types, type_list, label_map, hierarchy,
        loader, device, args.proj_dim
    )

    print(f"\n=== Baseline vs. Proposed: per-type silhouette score ===\n")
    print(f"{'Semantic Type':45s} {'Baseline':>10s} {'Proposed':>10s} {'Change':>10s}")
    rows = sorted(type_list, key=lambda t: -(proposed_scores[t] - baseline_scores[t]))
    for t in rows:
        b, pr = baseline_scores[t], proposed_scores[t]
        print(f"{t:45s} {b:10.4f} {pr:10.4f} {pr-b:+10.4f}")

    print(f"\n{'OVERALL':45s} {baseline_overall:10.4f} {proposed_overall:10.4f} "
          f"{proposed_overall-baseline_overall:+10.4f}")

    n_improved = sum(1 for t in type_list if proposed_scores[t] > baseline_scores[t])
    print(f"\n{n_improved}/{len(type_list)} types show improved separation under the proposed model.")


if __name__ == "__main__":
    main()
