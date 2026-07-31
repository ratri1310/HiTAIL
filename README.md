# HiTAIL — Hierarchy-guided Semantic Prototype Distribution Learning

A method for extreme multi-label biomedical text classification (MeSH
indexing), aimed at improving prediction for rare/long-tail labels using
UMLS semantic-type prototypes, ontology-guided surrogate contrastive
learning, and hierarchy-aware distribution alignment — without learning a
separate prototype per label or per hierarchy node.


**Core pipeline:**
- `model.py` — encoder + semantic-type prototypes + classifier
- `losses.py` 
- `train.py` — staged training (Stage 0–5), with contrastive warm-up
  before BCE joins
- `evaluate.py` — P@k, nDCG@k, PSP@k, PSnDCG@k, macro/micro-F1
- `hierarchy.py`, `data.py`, `config.py` — data loading and hierarchy
  resolution (full polyhierarchy sets + legacy single-resolved index)

**Data preparation (run once, before training):**
- `build_full_parent_sets.py` — full MeSH parent/grandparent sets per label
- `precompute_ic_and_weights.py` — information-content and frequency
  weights per label, per domain


## Setup

```bash
pip install -r requirements.txt
```

## Data

Please check /Datasets 
Set `HSPDL_DATA_ROOT` and `HSPDL_PROJECT_ROOT` environment variables to
point at your own data/output locations (see `config.py`).

## Usage

```bash
# 1. One-time, global
python build_full_parent_sets.py

# 2. Once per domain
python precompute_ic_and_weights.py --domain <domain>

# 4. Train

python -m hspdl.train --domain <domain> --stage 5   

# 5. Evaluate
python -m hspdl.tail_evaluate --domain <domain> --stage <N>
```
