# HSPDL — clean rebuild

A full rewrite, not layered on the old `hier_proto_mesh` package. No `e_y`,
`mu_y`, frequency gate, or `p_A`/`p_P` prototype tables — those were removed
as the root of the representation-collapse failure found in the old design.
Only semantic-type prototypes (`p_S`) are trainable; parent/grandparent
influence the model only through pair construction (`L_SSL`) and
distribution weighting (`L_dist`), never as their own prototype tables.

## What's verified vs. what isn't

Every piece of NEW logic (pair categorization, IC/omega_y computation,
macro/micro-F1, hit@k, early-stopping) was checked against hand-computed
values in a pure-Python reimplementation before being written into the real
(torch-dependent) code — see the conversation for the exact test cases.
**None of this has been run against a real GPU/model** — `torch` isn't
available in the sandbox this was built in. Watch the first real run closely
rather than trust it blind, same as everything built this session.

## Run order

**1. One-time, global (not per-domain):**
```bash
python build_full_parent_sets.py
```
Builds `full_parent_sets.json` / `full_grandparent_sets.json` — the FULL
polyhierarchy sets, not the old single-resolved value. Needs
`global_mesh_tree.json` + `global_tree_number_to_uid.json` to already exist
(from the earlier `build_ancestor_file.py` run).

**2. Once per domain:**
```bash
python precompute_ic_and_weights.py --domain immunology
```
Writes `ic_scores.json` + `omega_y.json` into that domain's data folder.
Needs `label_freq.json` (already have it) and step 1's output.

**3. Train, one stage at a time:**
```bash
python -m hspdl.train --domain immunology --stage 0
python -m hspdl.train --domain immunology --stage 1
python -m hspdl.train --domain immunology --stage 2
python -m hspdl.train --domain immunology --stage 3
python -m hspdl.train --domain immunology --stage 4
python -m hspdl.train --domain immunology --stage 5   # optional semantic-compat enhancement
```

Each stage writes `metrics.json` (every metric: P@k, hit@k, nDCG@k, PSP@k,
PSnDCG@k, macro-F1, micro-F1) and `checkpoint.pt` to
`runs/{domain}_stage{stage}/`.

## What's proposed vs. explicitly locked — flagged, not hidden

- `lr=2e-5` — proposed, not explicitly discussed. Was `1e-3` before, which
  contributed to collapse; this is a standard fine-tuning value. Override
  with `--lr`.
- `warmup_epochs=2` — the contrastive-only-before-BCE strategy was agreed on
  in principle; the exact epoch count is my proposal. Override with
  `--warmup-epochs 0` to disable entirely and compare.
- `gamma=0.5` (L_sep^S margin) and `early_stop_rel_threshold=0.01` — both
  proposed defaults, not explicitly locked during design. Easy to change in
  `config.py` if they're wrong.
- Everything else (`c=2`, `eta1/eta2=0.7/0.3`, `beta=0.1`,
  `lambda_ST/SSL/D/R = 0.3/0.5/0.3/0.1`, the 4-way pair rule, the `L_SSL`
  formula) is exactly what was locked turn by turn — nothing silently
  changed.

## Known simplification, stated plainly

`L_dist`'s predicted-distribution pooling (`q̂_A1`/`q̂_A2`) still needs ONE
canonical parent/grandparent row per label to build a fixed-size target —
that reuses the OLD single-resolved `parent_map.json`/`grandparent_map.json`
for that specific purpose only. `L_SSL`'s pair construction and the
IC/omega_y computation use the NEW full polyhierarchy sets. Both are real,
intentional, and explained in `hierarchy.py`'s docstring — not an
inconsistency.
