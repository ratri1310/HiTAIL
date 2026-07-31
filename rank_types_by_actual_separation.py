"""
Ranks semantic types by how well their REAL documents actually separate in
embedding space -- using silhouette score (a standard clustering-quality
metric), computed directly on document embeddings, not the prototype-space
farthest-point proxy from select_distinct_semantic_types.py. That proxy
proved insufficient: "Age Group" was selected for maximal PROTOTYPE
separation but still looked poorly separated in the real rendered plot.

This measures the thing we actually care about directly: for each
document, is it closer to other same-type documents than to the nearest
different-type documents, in real embedding space.

Usage:
    python -m hspdl.rank_types_by_actual_separation --domain immunology --stage 5 --top-n 8
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--stage", type=int, default=5)
    p.add_argument("--min-doc-count", type=int, default=20)
    p.add_argument("--top-n", type=int, default=8, help="how many best-separated types to report")
    p.add_argument("--candidates", default="",
                    help="comma-separated exact type names to restrict the comparison to -- "
                         "use this to fairly compare specific types together in ONE pass, "
                         "since silhouette score is relative to whatever else is in the comparison")
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
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
    if args.candidates:
        requested = {t.strip() for t in args.candidates.split(",")}
        valid_types = {t for t in requested if type_counts.get(t, 0) >= args.min_doc_count}
        missing = requested - valid_types
        if missing:
            print(f"WARNING: excluded (below min-doc-count or not found): {missing}")
    else:
        valid_types = {t for t, c in type_counts.items() if c >= args.min_doc_count}
    print(f"{len(valid_types)} semantic types meet the >= {args.min_doc_count} document threshold")

    keep_idx = [i for i, t in enumerate(doc_type) if t in valid_types]
    kept_types = [doc_type[i] for i in keep_idx]
    sub_ds = torch.utils.data.Subset(test_ds, keep_idx)
    sub_loader = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Encoding {len(keep_idx)} documents with Stage {args.stage}...")
    use_semantic_compat = (args.stage == 5)
    model = HSPDLModel(
        num_labels=len(label_map), num_semantic_types=hierarchy.num_semantic_types,
        proj_dim=args.proj_dim, use_semantic_compatibility=use_semantic_compat,
    ).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, "runs", f"{args.domain}_stage{args.stage}", "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    embeddings = []
    with torch.no_grad():
        for batch in sub_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            z = model.encode(input_ids, attention_mask)
            embeddings.append(z.cpu())
    X = torch.cat(embeddings).numpy()

    type_list = sorted(valid_types)
    type_to_int = {t: i for i, t in enumerate(type_list)}
    cluster_labels = np.array([type_to_int[t] for t in kept_types])

    print("Computing silhouette scores (this measures ACTUAL embedding separation, not a proxy)...")
    sample_scores = silhouette_samples(X, cluster_labels)

    per_type_score = {}
    for t in type_list:
        mask = cluster_labels == type_to_int[t]
        per_type_score[t] = sample_scores[mask].mean()

    ranked = sorted(per_type_score.items(), key=lambda kv: -kv[1])

    print(f"\n=== Semantic types ranked by ACTUAL document-embedding separation "
          f"(silhouette score, higher = cleaner real-world visual separation) ===\n")
    for t, score in ranked:
        flag = "  <-- best candidates" if t in [r[0] for r in ranked[:args.top_n]] else ""
        print(f"  {score:+.4f}  {t}  (n={type_counts[t]}){flag}")

    best_n = [t for t, _ in ranked[:args.top_n]]
    print(f"\nTop {args.top_n} by REAL separation (pick any 5 from here for the plot):")
    print(f"  --types \"{','.join(best_n[:5])}\"")


if __name__ == "__main__":
    main()