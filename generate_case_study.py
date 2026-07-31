"""
Generates a real qualitative case-study table: baseline (Stage 0) vs proposed
(Stage 5), same format as the reference figure -- PMID, truncated abstract,
ground-truth labels, top-5 predictions from each model, with correct/incorrect
marked. Pulls REAL documents/predictions, nothing fabricated.

Two case types selected automatically, matching the reference figure:
  A) "Rare label recovery" -- a true label missing from Stage 0's top-5 but
     present in Stage 5's top-5, and that label is in a rare-frequency bucket.
  B) "Improved ranking" -- a true label present in BOTH top-5s, but ranked
     meaningfully higher (lower rank number) by Stage 5.

Usage:
    python -m hspdl.generate_case_study --domain immunology --n-cases 5
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .config import DomainConfig, PROJECT_ROOT
from .data import MeshDataset, collate_fn, build_tokenizer, load_label_map, load_label_freq, build_freq_tensor
from .hierarchy import load_hierarchy
from .model import HSPDLModel
from .train import build_label_semantic_compat_matrix
from .hierarchy import build_label_index_maps

HEAD_BUCKET_SIZE = 50
RARITY_PERCENTILE_TIERS = [(0, 5, "rarest0to5"), (5, 10, "rarest5to10"), (10, 20, "rarest10to20")]


def assign_percentile_buckets(label_freq):
    n = label_freq.size(0)
    order_desc = torch.argsort(label_freq, descending=True)
    head_idx = set(order_desc[:HEAD_BUCKET_SIZE].tolist())
    remaining = [i for i in order_desc.tolist() if i not in head_idx]
    remaining_sorted = sorted(remaining, key=lambda i: label_freq[i].item())
    n_rem = len(remaining_sorted)
    buckets = [None] * n
    for i in head_idx:
        buckets[i] = "head50"
    for rank, idx in enumerate(remaining_sorted):
        pct = 100 * rank / max(1, n_rem)
        assigned = "middle20plus"
        for lo, hi, name in RARITY_PERCENTILE_TIERS:
            if lo <= pct < hi:
                assigned = name
                break
        buckets[idx] = assigned
    return buckets


def load_model_and_score(domain, stage, label_map, hierarchy, freq_all, test_loader, device, proj_dim=256):
    dcfg = DomainConfig.build(domain)
    use_semantic_compat = (stage == 5)
    model = HSPDLModel(
        num_labels=len(label_map), num_semantic_types=hierarchy.num_semantic_types,
        proj_dim=proj_dim, use_semantic_compatibility=use_semantic_compat,
    ).to(device)

    ckpt_path = os.path.join(PROJECT_ROOT, "runs", f"{domain}_stage{stage}", "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if use_semantic_compat:
        label_to_semantic, _, _ = build_label_index_maps(label_map, hierarchy)
        compat_matrix = build_label_semantic_compat_matrix(label_to_semantic, len(label_map), model, device)
        model.set_label_semantic_compat_matrix(compat_matrix)

    all_scores = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            z_i = model.encode(input_ids, attention_mask)
            logits = model.classify(z_i)
            all_scores.append(logits.cpu())
    return torch.cat(all_scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--n-cases", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--abstract-chars", type=int, default=200)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg = DomainConfig.build(args.domain)
    label_map = load_label_map(dcfg.label_map_path)
    label_freq = load_label_freq(dcfg.label_freq_path)
    inv_label_map = {v: k for k, v in label_map.items()}

    # Human-readable label names -- checking this file exists before assuming
    names_path = "/localscratch/Users/ratri/Bert/datasets/global_mesh_names.json"
    if not os.path.exists(names_path):
        print(f"WARNING: {names_path} not found -- falling back to raw UIDs as label names. "
              f"If this file is at a different path, pass it via --names-path.")
        label_names = {}
    else:
        with open(names_path) as f:
            label_names = json.load(f)

    def name_of(uid):
        return label_names.get(uid, uid)

    from .config import GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT
    hierarchy = load_hierarchy(
        GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT, dcfg.semantic_type_path,
        dcfg.legacy_parent_map_path, dcfg.legacy_grandparent_map_path, label_map,
    )

    tokenizer = build_tokenizer()
    test_ds = MeshDataset(os.path.join(dcfg.data_dir, "test.jsonl"), label_map, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Checking whether raw records carry a PMID field before assuming -- flag
    # clearly if not, rather than silently substituting something else.
    has_pmid = len(test_ds.records) > 0 and "pmid" in test_ds.records[0]
    if not has_pmid:
        print("WARNING: 'pmid' field not found in test.jsonl records -- using document "
              "index as the identifier instead. Check your raw data field name if PMIDs "
              "should be available (may be under a different key).")

    freq_all = build_freq_tensor(label_map, label_freq, device)
    bucket_names = assign_percentile_buckets(freq_all.cpu())
    rare_buckets = {"rarest0to5", "rarest5to10", "rarest10to20"}

    print("Scoring with Stage 0 (baseline)...")
    scores_s0 = load_model_and_score(args.domain, 0, label_map, hierarchy, freq_all, test_loader, device)
    print("Scoring with Stage 5 (proposed)...")
    scores_s5 = load_model_and_score(args.domain, 5, label_map, hierarchy, freq_all, test_loader, device)

    topk_s0 = scores_s0.topk(args.top_k, dim=1).indices
    topk_s5 = scores_s5.topk(args.top_k, dim=1).indices
    rank_s0 = torch.argsort(torch.argsort(scores_s0, dim=1, descending=True), dim=1)
    rank_s5 = torch.argsort(torch.argsort(scores_s5, dim=1, descending=True), dim=1)

    case_a, case_b = [], []  # rare recovery, improved ranking

    for i, rec in enumerate(test_ds.records):
        true_labels = [label_map[u] for u in rec["labels"] if u in label_map]
        s0_top = set(topk_s0[i].tolist())
        s5_top = set(topk_s5[i].tolist())

        for lid in true_labels:
            if lid not in s0_top and lid in s5_top and bucket_names[lid] in rare_buckets:
                case_a.append((i, lid))
            elif lid in s0_top and lid in s5_top:
                r0, r5 = rank_s0[i, lid].item(), rank_s5[i, lid].item()
                if r0 - r5 >= 2:  # meaningfully higher-ranked in stage 5
                    case_b.append((i, lid, r0, r5))

    print(f"\nFound {len(case_a)} candidate 'rare label recovery' cases, "
          f"{len(case_b)} candidate 'improved ranking' cases.\n")

    selected = case_a[: (args.n_cases + 1) // 2] + [c[:2] for c in case_b[: args.n_cases // 2]]
    selected = selected[: args.n_cases]

    print("=" * 100)
    saved_cases = []
    for doc_idx, highlight_lid in selected:
        rec = test_ds.records[doc_idx]
        pmid = rec.get("pmid", f"doc_index_{doc_idx}")
        abstract = rec["text"][: args.abstract_chars] + "..."
        true_names = [name_of(u) for u in rec["labels"] if u in label_map]

        s0_names = [(name_of(inv_label_map[l]), l in [label_map[u] for u in rec["labels"] if u in label_map])
                    for l in topk_s0[doc_idx].tolist()]
        s5_names = [(name_of(inv_label_map[l]), l in [label_map[u] for u in rec["labels"] if u in label_map])
                    for l in topk_s5[doc_idx].tolist()]

        print(f"PMID: {pmid}")
        print(f"Abstract: {abstract}")
        print(f"Ground Labels: {'; '.join(true_names)}")
        print(f"Baseline (Stage 0) Top-{args.top_k}: " +
              "; ".join(f"{n} ({'CORRECT' if c else 'wrong'})" for n, c in s0_names))
        print(f"Proposed (Stage 5) Top-{args.top_k}: " +
              "; ".join(f"{n} ({'CORRECT' if c else 'wrong'})" for n, c in s5_names))
        print(f"Highlighted label: {name_of(inv_label_map[highlight_lid])}")
        print("-" * 100)

        saved_cases.append({
            "pmid": pmid, "abstract": abstract, "ground_labels": true_names,
            "baseline_top5": [{"label": n, "correct": c} for n, c in s0_names],
            "proposed_top5": [{"label": n, "correct": c} for n, c in s5_names],
            "highlighted_label": name_of(inv_label_map[highlight_lid]),
        })

    out_dir = os.path.join(PROJECT_ROOT, "runs", "case_studies")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{args.domain}_case_studies.json")
    with open(json_path, "w") as f:
        json.dump(saved_cases, f, indent=2)

    md_path = os.path.join(out_dir, f"{args.domain}_case_studies.md")
    with open(md_path, "w") as f:
        f.write(f"| PMID | Abstract | Ground Labels | Baseline Top-5 | Proposed Top-5 |\n")
        f.write(f"|---|---|---|---|---|\n")
        for c in saved_cases:
            base_str = "; ".join(f"{n}{'✓' if ok else '✗'}" for n, ok in
                                   [(x['label'], x['correct']) for x in c['baseline_top5']])
            prop_str = "; ".join(f"{n}{'✓' if ok else '✗'}" for n, ok in
                                   [(x['label'], x['correct']) for x in c['proposed_top5']])
            f.write(f"| {c['pmid']} | {c['abstract'][:100]}... | {'; '.join(c['ground_labels'])} | "
                    f"{base_str} | {prop_str} |\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()