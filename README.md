# HSPDL — Hierarchy-guided Semantic Prototype Distribution Learning

A method for extreme multi-label biomedical text classification (MeSH
indexing), aimed at improving prediction for rare/long-tail labels using
UMLS semantic-type prototypes, ontology-guided surrogate contrastive
learning, and hierarchy-aware distribution alignment — without learning a
separate prototype per label or per hierarchy node.

## Project status

Active research project. The architecture and every loss term are
implemented, tested, and numerically verified (see `preflight_check.py`
and `check_collapse.py` for the diagnostic tools used to confirm this).
Current experiments are still evaluating whether the full pipeline
improves over a plain fine-tuned PubMedBERT baseline — see the paper /
internal notes for the latest results, not this README.

## What's in this repo

**Core pipeline:**
- `model.py` — encoder + semantic-type prototypes + classifier
- `losses.py` — L_ST, L_SSL, L_dist, L_sep^S
- `train.py` — staged training (Stage 0–5), with contrastive warm-up
  before BCE joins
- `evaluate.py` — P@k, nDCG@k, PSP@k, PSnDCG@k, macro/micro-F1
- `hierarchy.py`, `data.py`, `config.py` — data loading and hierarchy
  resolution (full polyhierarchy sets + legacy single-resolved index)

**Data preparation (run once, before training):**
- `build_full_parent_sets.py` — full MeSH parent/grandparent sets per label
- `precompute_ic_and_weights.py` — information-content and frequency
  weights per label, per domain

**Diagnostics (no training required — inspect an existing checkpoint or a
fresh model):**
- `preflight_check.py` — loss-magnitude sanity check on a fresh model,
  before committing to a full training run
- `check_collapse.py` — representation-collapse check on a trained model
- `tail_evaluate.py` — frequency-stratified evaluation
- `compare_separation_baseline_vs_proposed.py`,
  `rank_types_by_actual_separation.py`,
  `select_distinct_semantic_types.py` — semantic-type separation analysis

**Qualitative / figures:**
- `generate_case_study.py` — real prediction examples (baseline vs.
  proposed), saved to JSON + markdown
- `generate_tsne_comparison.py` — t-SNE comparison plot
- `find_pmids_from_raw.py` — recovers PMIDs dropped during preprocessing

## Setup

```bash
pip install -r requirements.txt
```

## Data

This repo does not include the underlying datasets. You'll need:
- Raw domain data (PubMed abstracts + MeSH labels) in the format expected
  by `data.py`'s `MeshDataset`
- Global MeSH tree files (`global_mesh_tree.json`,
  `global_tree_number_to_uid.json`) for hierarchy resolution
- UMLS semantic-type mappings (MRCONSO/MRSTY-derived)

Set `HSPDL_DATA_ROOT` and `HSPDL_PROJECT_ROOT` environment variables to
point at your own data/output locations (see `config.py`).

## Usage

```bash
# 1. One-time, global
python build_full_parent_sets.py

# 2. Once per domain
python precompute_ic_and_weights.py --domain <domain>

# 3. Before training a new domain/config — sanity-check loss magnitudes
python -m hspdl.preflight_check --domain <domain>

# 4. Train, one stage at a time
python -m hspdl.train --domain <domain> --stage 0   # baseline
python -m hspdl.train --domain <domain> --stage 1   # + L_ST
python -m hspdl.train --domain <domain> --stage 2   # + L_SSL
python -m hspdl.train --domain <domain> --stage 3   # + L_dist
python -m hspdl.train --domain <domain> --stage 4   # + L_sep^S
python -m hspdl.train --domain <domain> --stage 5   # + semantic compatibility

# 5. Evaluate
python -m hspdl.tail_evaluate --domain <domain> --stage <N>
```

## Citation

TBD.
