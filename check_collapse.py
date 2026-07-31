"""
Phase 1 diagnostic: did the HSPDL rebuild actually fix representation
collapse, or did we just assume it did because we removed the suspected
cause (mu_y_consistency)? Never re-verified until now. Same check as the
one that found TOTAL collapse (std=0.0000) in the old architecture.

Usage:
    python -m hspdl.check_collapse --domain immunology --stage 5
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .config import DomainConfig, PROJECT_ROOT
from .data import MeshDataset, collate_fn, build_tokenizer, load_label_map
from .hierarchy import load_hierarchy
from .model import HSPDLModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--run-dir", default="")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--n-batches", type=int, default=5)
    args = p.parse_args()

    run_dir = args.run_dir or os.path.join(PROJECT_ROOT, "runs", f"{args.domain}_stage{args.stage}")
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg = DomainConfig.build(args.domain)
    label_map = load_label_map(dcfg.label_map_path)
    from .config import GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT
    hierarchy = load_hierarchy(
        GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT, dcfg.semantic_type_path,
        dcfg.legacy_parent_map_path, dcfg.legacy_grandparent_map_path, label_map,
    )

    tokenizer = build_tokenizer()
    train_ds = MeshDataset(os.path.join(dcfg.data_dir, "train.jsonl"), label_map, tokenizer)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = HSPDLModel(
        num_labels=len(label_map), num_semantic_types=hierarchy.num_semantic_types,
        proj_dim=args.proj_dim,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"=== Collapse check: {ckpt_path} ===\n")
    all_sims = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            z = model.encode(input_ids, attention_mask)
            sim = z @ z.T
            n = sim.size(0)
            off_diag = sim[~torch.eye(n, dtype=torch.bool, device=device)]
            all_sims.append(off_diag.cpu())
            print(f"batch {i}: mean={off_diag.mean().item():.4f} std={off_diag.std().item():.4f} "
                  f"min={off_diag.min().item():.4f} max={off_diag.max().item():.4f}")

    all_sims = torch.cat(all_sims)
    print(f"\nOverall: mean={all_sims.mean().item():.4f} std={all_sims.std().item():.4f}")
    print("std < 0.02 with mean near 1.0 = COLLAPSED. Healthy = real spread (std > 0.05-0.1).")


if __name__ == "__main__":
    main()
