"""
HSPDL training. Run one stage at a time, see metrics after each.

Training strategy (per today's discussion): for any stage with a contrastive
term active (stage >= 1), the first `warmup_epochs` epochs optimize ONLY the
contrastive terms (L_ST [+ L_SSL]) -- NO BCE. This gives the encoder a clean
period to build real discriminative structure before BCE's gradient (and any
other terms) start competing with it, directly targeting the collapse
mechanism found today (multiple competing terms from batch 1, no chance for
any single term to establish structure first). After warmup, the FULL
stage objective (BCE + everything active at that stage) trains for the
remaining epochs.

Early stopping: if the epoch loss changes by less than
`early_stop_rel_threshold` (relative) for `early_stop_patience` consecutive
epochs, stop early -- avoids wasting epochs once training has plateaued
(exactly what we saw happen, unnoticed, in the old pipeline).

Usage:
    python -m hspdl.train --domain immunology --stage 0
    python -m hspdl.train --domain immunology --stage 2 --warmup-epochs 3
"""

import argparse
import json
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import DomainConfig, TrainConfig, DOMAINS, active_terms_for_stage, \
    LAMBDA_ST, LAMBDA_SSL, LAMBDA_D, LAMBDA_R, GAMMA_SEP, TEMPERATURE, BETA_DEFAULT
from .data import MeshDataset, collate_fn, build_tokenizer, load_label_map, load_label_freq, build_freq_tensor
from .hierarchy import load_hierarchy, build_label_index_maps
from .model import HSPDLModel
from .losses import (
    semantic_prototype_loss, surrogate_contrastive_loss,
    distribution_alignment_loss, semantic_prototype_separation_loss,
)
from .evaluate import evaluate_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=DOMAINS)
    p.add_argument("--stage", type=int, required=True, choices=range(0, 6))
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="")
    p.add_argument("--lambda-st", type=float, default=None)
    p.add_argument("--lambda-ssl", type=float, default=None)
    p.add_argument("--lambda-d", type=float, default=None)
    p.add_argument("--lambda-r", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    return p.parse_args()


def build_label_semantic_compat_matrix(label_to_semantic, num_labels, model, device):
    """
    Mean-pooled p_bar_S(y) per label, per Decision 3 (mean pooling). Recomputed
    fresh each time this is called, since p_S is still training -- must be
    called at least once per epoch when use_semantic_compatibility=True, not
    just once at model construction.
    """
    protos = model.semantic_protos.normalized()  # [num_ST, dim]
    dim = protos.size(1)
    matrix = torch.zeros(num_labels, dim, device=device)
    for lid in range(num_labels):
        st_idx = label_to_semantic.get(lid, set())
        if st_idx:
            idx_t = torch.tensor(list(st_idx), device=device, dtype=torch.long)
            matrix[lid] = protos[idx_t].mean(dim=0)
    # DETACHED: this matrix gets reused across every batch in the epoch, but
    # each batch calls .backward() separately -- if left attached to the
    # graph, the first batch's backward() frees the intermediate nodes behind
    # this matrix's own computation (it depends on p_S), and the next batch
    # crashes trying to backprop through them again. p_S still gets a real,
    # direct gradient from L_ST regardless -- this just isn't a second path.
    return matrix.detach()


def run(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    def log(msg):
        print(f"[{cfg.domain} stage {cfg.stage}] {msg}", flush=True)

    dcfg = DomainConfig.build(cfg.domain)
    label_map = load_label_map(dcfg.label_map_path)
    label_freq = load_label_freq(dcfg.label_freq_path)

    from .config import GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT
    hierarchy = load_hierarchy(
        GLOBAL_FULL_PARENT, GLOBAL_FULL_GRANDPARENT, dcfg.semantic_type_path,
        dcfg.legacy_parent_map_path, dcfg.legacy_grandparent_map_path, label_map,
    )
    label_to_semantic, label_to_parents, label_to_gp = build_label_index_maps(label_map, hierarchy)

    with open(dcfg.omega_y_path) as f:
        omega_y_raw = json.load(f)  # UID -> omega_y
    inv_label_map = {v: k for k, v in label_map.items()}
    omega_y_lookup = {idx: omega_y_raw.get(uid, 0.0) for idx, uid in inv_label_map.items()}
    label_freq_lookup = {idx: label_freq.get(uid, 1) for idx, uid in inv_label_map.items()}

    tokenizer = build_tokenizer()
    train_ds = MeshDataset(os.path.join(dcfg.data_dir, "train.jsonl"), label_map, tokenizer)
    test_ds = MeshDataset(os.path.join(dcfg.data_dir, "test.jsonl"), label_map, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    freq_all = build_freq_tensor(label_map, label_freq, device)

    parent_idx_all = torch.tensor([hierarchy.parent_idx[i] for i in range(len(label_map))],
                                    dtype=torch.long, device=device)
    grandparent_idx_all = torch.tensor([hierarchy.grandparent_idx[i] for i in range(len(label_map))],
                                         dtype=torch.long, device=device)

    use_semantic_compat = (cfg.stage == 5)
    model = HSPDLModel(
        num_labels=len(label_map), num_semantic_types=hierarchy.num_semantic_types,
        proj_dim=cfg.proj_dim, use_semantic_compatibility=use_semantic_compat, beta=cfg.beta,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    active = active_terms_for_stage(cfg.stage)
    log(f"active terms: {active}, warmup_epochs={cfg.warmup_epochs}, lr={cfg.lr}, "
        f"lambda_st={cfg.lambda_st}, lambda_ssl={cfg.lambda_ssl}, lambda_d={cfg.lambda_d}, lambda_r={cfg.lambda_r}")

    epoch_losses = []
    for epoch in range(cfg.epochs):
        model.train()
        in_warmup = epoch < cfg.warmup_epochs and ("st" in active or "ssl" in active)
        running = 0.0
        n_batches = 0

        if use_semantic_compat and not in_warmup:
            compat_matrix = build_label_semantic_compat_matrix(
                label_to_semantic, len(label_map), model, device
            )
            model.set_label_semantic_compat_matrix(compat_matrix)

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_ids_batch = batch["label_ids"]

            z_i = model.encode(input_ids, attention_mask)

            loss = torch.tensor(0.0, device=device)

            if "st" in active:
                semantic_idx_batch = [
                    set().union(*[label_to_semantic.get(lid, set()) for lid in labels_i]) if labels_i else set()
                    for labels_i in label_ids_batch
                ]
                l_st = semantic_prototype_loss(z_i, semantic_idx_batch, model.semantic_protos, TEMPERATURE)
                loss = loss + cfg.lambda_st * l_st

            if "ssl" in active:
                l_ssl = surrogate_contrastive_loss(
                    z_i, label_ids_batch, label_freq_lookup,
                    label_to_semantic, label_to_parents, label_to_gp, TEMPERATURE,
                )
                loss = loss + cfg.lambda_ssl * l_ssl

            if not in_warmup:
                logits = model.classify(z_i)
                l_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                loss = loss + l_bce

                if "dist" in active:
                    l_dist = distribution_alignment_loss(
                        logits, label_ids_batch, omega_y_lookup,
                        parent_idx_all, grandparent_idx_all,
                        hierarchy.num_parent_rows, hierarchy.num_grandparent_rows,
                    )
                    loss = loss + cfg.lambda_d * l_dist

                if "sep" in active:
                    l_sep = semantic_prototype_separation_loss(model.semantic_protos, GAMMA_SEP)
                    loss = loss + cfg.lambda_r * l_sep

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1

        avg_loss = running / max(1, n_batches)
        epoch_losses.append(avg_loss)
        phase = "WARMUP" if in_warmup else "full"
        log(f"epoch {epoch} [{phase}] avg loss {avg_loss:.4f}")

        if len(epoch_losses) >= cfg.early_stop_patience + 1:
            recent = epoch_losses[-(cfg.early_stop_patience + 1):]
            rel_changes = [abs(recent[i + 1] - recent[i]) / max(abs(recent[i]), 1e-8)
                            for i in range(len(recent) - 1)]
            if all(rc < cfg.early_stop_rel_threshold for rc in rel_changes):
                log(f"early stopping: loss plateaued for {cfg.early_stop_patience} epochs "
                    f"(rel changes: {[f'{rc:.4f}' for rc in rel_changes]})")
                break

    model.eval()
    if use_semantic_compat:
        compat_matrix = build_label_semantic_compat_matrix(
            label_to_semantic, len(label_map), model, device
        )
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
    metrics = evaluate_all_metrics(scores, labels_all, freq_all.cpu())
    log(f"final metrics: {metrics}")

    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump({"domain": cfg.domain, "stage": cfg.stage, "metrics": metrics,
                    "epoch_losses": epoch_losses}, f, indent=2)

    torch.save({"model_state": model.state_dict(), "epoch_losses": epoch_losses},
               os.path.join(cfg.output_dir, "checkpoint.pt"))
    log(f"done. wrote metrics.json + checkpoint.pt to {cfg.output_dir}")


def main():
    args = parse_args()
    kwargs = {"domain": args.domain, "stage": args.stage, "seed": args.seed}
    if args.lr is not None:
        kwargs["lr"] = args.lr
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.warmup_epochs is not None:
        kwargs["warmup_epochs"] = args.warmup_epochs
    if args.output_dir:
        kwargs["output_dir"] = args.output_dir
    if args.lambda_st is not None:
        kwargs["lambda_st"] = args.lambda_st
    if args.lambda_ssl is not None:
        kwargs["lambda_ssl"] = args.lambda_ssl
    if args.lambda_d is not None:
        kwargs["lambda_d"] = args.lambda_d
    if args.lambda_r is not None:
        kwargs["lambda_r"] = args.lambda_r
    if args.beta is not None:
        kwargs["beta"] = args.beta
    cfg = TrainConfig(**kwargs)
    run(cfg)


if __name__ == "__main__":
    main()