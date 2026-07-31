"""
Picks semantic types for the t-SNE comparison plot using the model's own
trained prototypes, not intuition. Farthest-point sampling: greedily selects
the N semantic types whose prototype vectors are maximally mutually
separated in embedding space -- directly checks whether L_sep^S succeeded,
since that's the component whose job is keeping these prototypes apart.

Filtered by a minimum real-document count too -- a type can be maximally
separated in PROTOTYPE space but still be a bad visualization choice if
almost no test documents actually have it as their majority type (this is
exactly what happened with "Chemical" in the last plot).

Usage:
    python -m hspdl.select_distinct_semantic_types --domain immunology --stage 5 --n-types 5
"""

import argparse
import json
import os
from collections import Counter

import torch

from .config import DomainConfig, PROJECT_ROOT
from .data import load_label_map, load_jsonl
from .hierarchy import load_hierarchy
from .model import HSPDLModel


def farthest_point_sampling(dist_matrix, candidate_idx, n_select):
    """
    Greedy max-min diversity selection: start with the globally farthest
    pair among candidates, then repeatedly add whichever remaining
    candidate maximizes its MINIMUM distance to everything already
    selected. Standard technique for picking a maximally-separated subset.
    """
    candidate_idx = list(candidate_idx)
    if len(candidate_idx) <= n_select:
        return candidate_idx

    best_pair, best_dist = None, -1
    for a_i, a in enumerate(candidate_idx):
        for b in candidate_idx[a_i + 1:]:
            d = dist_matrix[a, b]
            if d > best_dist:
                best_dist, best_pair = d, (a, b)
    selected = list(best_pair)

    while len(selected) < n_select:
        remaining = [c for c in candidate_idx if c not in selected]
        best_c, best_min_dist = None, -1
        for c in remaining:
            min_dist = min(dist_matrix[c, s] for s in selected)
            if min_dist > best_min_dist:
                best_min_dist, best_c = min_dist, c
        selected.append(best_c)

    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--stage", type=int, default=5)
    p.add_argument("--n-types", type=int, default=5)
    p.add_argument("--min-doc-count", type=int, default=20,
                    help="exclude semantic types with fewer than this many test documents")
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

    model = HSPDLModel(
        num_labels=len(label_map), num_semantic_types=hierarchy.num_semantic_types,
        proj_dim=args.proj_dim,
    ).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, "runs", f"{args.domain}_stage{args.stage}", "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        protos = model.semantic_protos.normalized().cpu()  # [num_ST, dim]

    dist_matrix = torch.cdist(protos, protos, p=2).numpy()

    # real document counts per type, same majority-vote logic as the plotting script
    with open(dcfg.semantic_type_path) as f:
        raw_semantic = json.load(f)

    def all_semantic_types(uid):
        v = raw_semantic.get(uid)
        if isinstance(v, list):
            return v
        elif v:
            return [v]
        return []

    test_records = load_jsonl(os.path.join(dcfg.data_dir, "test.jsonl"))
    doc_type_counts = Counter()
    for rec in test_records:
        st_counts = Counter()
        for u in rec["labels"]:
            for st in all_semantic_types(u):
                st_counts[st] += 1
        if st_counts:
            doc_type_counts[st_counts.most_common(1)[0][0]] += 1

    # need the row index in `protos` for each named semantic type
    st_to_row = hierarchy.semantic_type_index  # name -> row index

    candidate_rows = [row for name, row in st_to_row.items()
                       if doc_type_counts.get(name, 0) >= args.min_doc_count]
    row_to_name = {row: name for name, row in st_to_row.items()}

    print(f"{len(candidate_rows)}/{len(st_to_row)} semantic types have >= {args.min_doc_count} "
          f"test documents")

    if len(candidate_rows) < args.n_types:
        print(f"WARNING: only {len(candidate_rows)} candidates meet the document-count threshold, "
              f"fewer than the requested {args.n_types} -- lower --min-doc-count or accept fewer types.")

    selected_rows = farthest_point_sampling(dist_matrix, candidate_rows, args.n_types)
    selected_names = [row_to_name[r] for r in selected_rows]

    print(f"\nSelected {len(selected_names)} maximally-separated types "
          f"(prototype space, filtered by real document support):")
    for name in selected_names:
        print(f"  {name}  (doc count: {doc_type_counts.get(name, 0)}, "
              f"min prototype distance to others selected: "
              f"{min(dist_matrix[st_to_row[name], st_to_row[o]] for o in selected_names if o != name):.4f})")

    print(f"\nReady to use:")
    print(f"  --types \"{','.join(selected_names)}\"")


if __name__ == "__main__":
    main()
