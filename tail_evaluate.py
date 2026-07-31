"""
Adds frequency-bucket stratification on top of what train.py already
computes -- NO retraining, just re-scores the test set from an existing
checkpoint.pt and breaks metrics down by label rarity, same percentile-rank
scheme already verified against real neurology/immunology data before
(HEAD_BUCKET_SIZE=50 fixed count, then percentile tiers among the rest).

Usage:
    python -m hspdl.tail_evaluate --domain immunology --stage 1
    python -m hspdl.tail_evaluate --domain immunology --stage 2
    python -m hspdl.tail_evaluate --domain immunology --stage 4
    python -m hspdl.tail_evaluate --domain immunology --stage 5
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
from .evaluate import evaluate_all_metrics

HEAD_BUCKET_SIZE = 50
RARITY_PERCENTILE_TIERS = [(0, 5, "rarest0to5"), (5, 10, "rarest5to10"), (10, 20, "rarest10to20")]
MIDDLE_BUCKET_NAME = "middle20plus"
HEAD_BUCKET_NAME = "head50"


def assign_percentile_buckets(label_freq: torch.Tensor) -> list:
    """Verified logic, ported unchanged from the earlier pipeline's evaluate.py."""
    n = label_freq.size(0)
    order_desc = torch.argsort(label_freq, descending=True)
    head_idx = set(order_desc[:HEAD_BUCKET_SIZE].tolist())

    remaining = [i for i in order_desc.tolist() if i not in head_idx]
    remaining_sorted_ascending = sorted(remaining, key=lambda i: label_freq[i].item())
    n_remaining = len(remaining_sorted_ascending)

    bucket_names = [None] * n
    for i in head_idx:
        bucket_names[i] = HEAD_BUCKET_NAME

    for rank, idx in enumerate(remaining_sorted_ascending):
        pct = 100 * rank / max(1, n_remaining)
        assigned = MIDDLE_BUCKET_NAME
        for lo, hi, name in RARITY_PERCENTILE_TIERS:
            if lo <= pct < hi:
                assigned = name
                break
        bucket_names[idx] = assigned

    return bucket_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--run-dir", default="")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--proj-dim", type=int, default=256)
    args = p.parse_args()

    run_dir = args.run_dir or os.path.join(PROJECT_ROOT, "runs", f"{args.domain}_stage{args.stage}")
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg = DomainConfig.build(args.domain)
    label_map = load_label_map(dcfg.label_map_path)
    label_freq = load_label_freq(dcfg.label_freq_path)

    from .config import GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT
    hierarchy = load_hierarchy(
        GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT, dcfg.semantic_type_path,
        dcfg.legacy_parent_map_path, dcfg.legacy_grandparent_map_path, label_map,
    )

    tokenizer = build_tokenizer()
    test_ds = MeshDataset(os.path.join(dcfg.data_dir, "test.jsonl"), label_map, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    freq_all = build_freq_tensor(label_map, label_freq, device)

    use_semantic_compat = (args.stage == 5)
    model = HSPDLModel(
        num_labels=len(label_map), num_semantic_types=hierarchy.num_semantic_types,
        proj_dim=args.proj_dim, use_semantic_compatibility=use_semantic_compat,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if use_semantic_compat:
        # rebuild the same compat matrix train.py used at final eval time
        from .hierarchy import build_label_index_maps
        label_to_semantic, _, _ = build_label_index_maps(label_map, hierarchy)
        from .train import build_label_semantic_compat_matrix
        compat_matrix = build_label_semantic_compat_matrix(label_to_semantic, len(label_map), model, device)
        model.set_label_semantic_compat_matrix(compat_matrix)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            z_i = model.encode(input_ids, attention_mask)
            logits = model.classify(z_i)
            all_scores.append(logits.cpu())
            all_labels.append(batch["labels"])

    scores = torch.cat(all_scores)
    labels_all = torch.cat(all_labels)
    freq_cpu = freq_all.cpu()

    bucket_names = assign_percentile_buckets(freq_cpu)
    print(f"\n=== {args.domain} stage {args.stage} — tail-stratified results ===\n")

    results = {}
    for bucket_name in ["rarest0to5", "rarest5to10", "rarest10to20", "middle20plus", "head50"]:
        idx = [i for i, b in enumerate(bucket_names) if b == bucket_name]
        if not idx:
            continue
        idx_t = torch.tensor(idx, dtype=torch.long)
        sub_scores = scores[:, idx_t]
        sub_labels = labels_all[:, idx_t]
        sub_freq = freq_cpu[idx_t]
        has_any = sub_labels.sum(dim=1) > 0
        if has_any.sum() == 0:
            print(f"  {bucket_name}: no test documents touch this bucket, skipping")
            continue
        m = evaluate_all_metrics(sub_scores[has_any], sub_labels[has_any], sub_freq, k_values=(1, 3, 5))
        m["num_labels_in_bucket"] = len(idx)
        results[bucket_name] = m
        print(f"  {bucket_name} (n={len(idx)} labels): P@3={m['P@3']:.4f} P@5={m['P@5']:.4f} "
              f"nDCG@5={m['nDCG@5']:.4f} macro_F1={m['macro_F1']:.6f} micro_F1={m['micro_F1']:.4f}")

    out_path = os.path.join(run_dir, "tail_stratified_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
