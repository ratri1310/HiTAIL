"""
Side-by-side t-SNE: baseline (Stage 0) vs proposed (Stage N), document
embeddings colored by semantic type. Direct visual test of what L_ST
actually did to the representation space -- matches the reference figure
style (discrete, well-separated clusters = success; loose/overlapping =
the resolution-ceiling theory holding).

Picks the TOP-N most frequent semantic types in the domain (for a readable
plot -- all ~120 would be unreadable), and only plots documents whose
labels are dominated by one of those N types, so each point has one clear
color.

Usage:
    python -m hspdl.generate_tsne_comparison --domain immunology --proposed-stage 5
"""

import argparse
import json
import os
from collections import Counter

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from .config import DomainConfig, PROJECT_ROOT
from .data import MeshDataset, collate_fn, build_tokenizer, load_label_map
from .hierarchy import load_hierarchy
from .model import HSPDLModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--proposed-stage", type=int, default=5)
    p.add_argument("--n-types", type=int, default=5, help="how many top semantic types to color/plot")
    p.add_argument("--types", default="", help="comma-separated exact semantic type names to use "
                                                  "instead of auto-picking the top-N most frequent "
                                                  "(useful for testing conceptually distinct types)")
    p.add_argument("--max-docs-per-type", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--out", default="tsne_comparison.png")
    p.add_argument("--marker-size", type=float, default=100,
                    help="lower this if points overlap too much and look 'blobby'")
    p.add_argument("--dpi", type=int, default=400,
                    help="raise this if the figure looks blurry once shrunk to fit a paper column")
    p.add_argument("--fig-width", type=float, default=10,
                    help="smaller canvas needs less downscaling to fit a narrow column")
    p.add_argument("--fig-height", type=float, default=4.5)
    p.add_argument("--alpha", type=float, default=0.85,
                    help="lower this to make overlapping points more distinguishable")
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

    # Assign each document a single DOMINANT semantic type (first semantic type
    # of the MAJORITY semantic type across ALL its true labels, not just the
    # first one -- "first label only" was noisier than needed for a
    # multi-label document, adding ambiguity to the ground-truth coloring
    # itself, independent of anything the model did.
    with open(dcfg.semantic_type_path) as f:
        raw_semantic = json.load(f)

    def all_semantic_types(uid):
        v = raw_semantic.get(uid)
        if isinstance(v, list):
            return v
        elif v:
            return [v]
        return []

    doc_semantic_type = []
    for rec in test_ds.records:
        st_counts = Counter()
        for u in rec["labels"]:
            for st in all_semantic_types(u):
                st_counts[st] += 1
        doc_semantic_type.append(st_counts.most_common(1)[0][0] if st_counts else None)

    type_counts = Counter(st for st in doc_semantic_type if st is not None)
    if args.types:
        top_types = [t.strip() for t in args.types.split(",")]
        missing = [t for t in top_types if t not in type_counts]
        if missing:
            print(f"WARNING: these requested types have zero matching documents: {missing}")
    else:
        top_types = [t for t, _ in type_counts.most_common(args.n_types)]
    print(f"Top {args.n_types} semantic types by document count: {top_types}")

    selected_idx = []
    per_type_count = Counter()
    for i, st in enumerate(doc_semantic_type):
        if st in top_types and per_type_count[st] < args.max_docs_per_type:
            selected_idx.append(i)
            per_type_count[st] += 1
    print(f"Selected {len(selected_idx)} documents across {len(top_types)} types: {dict(per_type_count)}")

    sub_ds = torch.utils.data.Subset(test_ds, selected_idx)
    loader = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    colors_for_selected = [doc_semantic_type[i] for i in selected_idx]

    def encode_all(stage):
        use_semantic_compat = (stage == 5)
        model = HSPDLModel(
            num_labels=len(label_map), num_semantic_types=hierarchy.num_semantic_types,
            proj_dim=args.proj_dim, use_semantic_compatibility=use_semantic_compat,
        ).to(device)
        ckpt_path = os.path.join(PROJECT_ROOT, "runs", f"{args.domain}_stage{stage}", "checkpoint.pt")
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
        return torch.cat(embeddings).numpy()

    print("Encoding with Stage 0 (baseline)...")
    z_baseline = encode_all(0)
    print(f"Encoding with Stage {args.proposed_stage} (proposed)...")
    z_proposed = encode_all(args.proposed_stage)

    print("Running t-SNE (baseline)...")
    tsne_baseline = TSNE(n_components=2, random_state=42, init="pca").fit_transform(z_baseline)
    print("Running t-SNE (proposed)...")
    tsne_proposed = TSNE(n_components=2, random_state=42, init="pca").fit_transform(z_proposed)

    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height))
    # Bright, high-saturation palette instead of matplotlib's default tab10
    # (which has some muted tones) -- explicit hex values for consistent,
    # vivid colors regardless of how many types are plotted.
    # ColorBrewer "Set1" -- a purpose-built qualitative palette (Cynthia Brewer),
    # specifically designed so categorical colors stay maximally distinguishable
    # from each other, even as small scattered points. Standard choice in
    # scientific plotting for exactly this reason.
    # Maximally saturated primary/secondary hues, as requested -- brightest
    # possible red, then alternating warm/cool for maximum contrast, tuned
    # for visibility when embedded/overlaid in another document (not just
    # print-safe distinctness, which was the prior ColorBrewer choice).
    BRIGHT_COLORS = ["#FF0000", "#FF8000", "#0080FF", "#FFD700", "#00B300",
                      "#FF00FF", "#00FFFF", "#8000FF", "#FF0080"]
    type_to_color = {t: BRIGHT_COLORS[i % len(BRIGHT_COLORS)] for i, t in enumerate(top_types)}

    for ax, tsne_coords, title in [
        (axes[0], tsne_baseline, "Pretrained Encoder"),
        (axes[1], tsne_proposed, "Proposed Approach"),
    ]:
        for t in top_types:
            mask = [c == t for c in colors_for_selected]
            pts = tsne_coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1], color=type_to_color[t], s=args.marker_size, alpha=args.alpha,
                       edgecolors="none")
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_xlabel("t-SNE dim1", fontsize=20)
        ax.set_ylabel("t-SNE dim2", fontsize=20)
        ax.tick_params(axis="both", labelsize=20)

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.out}")
    print("\nColor key (for the figure caption/description, since the legend was removed):")
    for t in top_types:
        print(f"  {type_to_color[t]}  =  {t}")


if __name__ == "__main__":
    main()