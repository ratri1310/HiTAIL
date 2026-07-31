# Dataset README

## Overview

This repository uses biomedical literature datasets derived from the BioASQ Task A corpus and three domain-specific subsets (Neurology, Immunology, Embryology). All datasets use MeSH (Medical Subject Headings) labels for multi-label text classification.

---

## Data Sources

### BioASQ
- **Source:** [BioASQ Task A](http://www.bioasq.org/)
- **Raw file:** `allMeSH_2022.json` — full PubMed corpus with MeSH annotations (16.2M articles)
- **We use:** Articles from years 2019–2021 (1,521,250 articles)
- **Download:** Register and download from the BioASQ website: http://www.bioasq.org/

### Domain Datasets (Neurology, Immunology, Embryology)
- Derived from the same BioASQ corpus by filtering articles whose MeSH labels belong to domain-specific concept sets
- Concept sets are built from UMLS (Unified Medical Language System): https://www.nlm.nih.gov/research/umls/

---

## Dataset Statistics

| Dataset     | Train      | Val       | Test    | Labels |
|-------------|------------|-----------|---------|--------|
| BioASQ      | 1,064,845  | 152,122   | 304,244 | 28,658 |
| Embryology  | 41,399     | 6,786     | 3,097   | 14,116 |
| Immunology  | 13,809     | 4,900     | 2,390   | 9,313  |
| Neurology   | 11,052     | 3,447     | 1,550   | 2,567  |

Split ratio: 70% train / 10% val / 20% test (random split, seed=42)

---

## How to Reproduce the BioASQ Dataset

### Step 1 — Download the raw data
Register at http://www.bioasq.org/ and download:
- `allMeSH_2022.json` — the full PubMed corpus with MeSH annotations

### Step 2 — Extract PMIDs for 2019–2021
```bash
python extract_bioasq_pmids.py \
    --input  /path/to/allMeSH_2022.json \
    --output pmids_2019_2021.txt \
    --years  2019 2020 2021
```
This produces `pmids_2019_2021.txt` with 1,521,250 PMIDs (one per line).

### Step 3 — Build the GZSL splits
```bash
python prepare_gzsl_splits_bioasq.py \
    --json_path  /path/to/allMeSH_2022_clean.json \
    --pmid_filter pmids_2019_2021.txt \
    --output_dir  splits/bioasq \
    --label_mode  unique_id \
    --seed        42 \
    --domain      bioasq \
    --overwrite
```

This produces:
```
splits/bioasq/
    train.json              # 1,064,845 documents
    val.json                # 152,122 documents
    test_supervised.json    # 304,244 documents
    seen_codes.json         # 25,793 seen MeSH codes
    unseen_codes.json       # 2,865 unseen MeSH codes
    split_stats.json        # split statistics
```

---

## How to Generate UMLS Semantic Type Files

The pipeline requires a per-label UMLS semantic-type mapping
(`umls_semantic_type.json`) for each domain — this is a required input,
not optional metadata: it defines the semantic-type prototype table and
gates the surrogate contrastive pair rule.

### Requirements
- UMLS `MRCONSO.RRF` (concept names) and `MRSTY.RRF` (semantic types) —
  see the UMLS Requirement section below
- The domain's MeSH descriptor list (from the split files above, or a
  standalone list of MeSH Unique IDs for the domain)

### Generate

```bash
python build_semantic_types.py \
    --domain        <bioasq|neurology|immunology|embryology> \
    --mrconso-path  /path/to/UMLS/META/MRCONSO.RRF \
    --mrsty-path    /path/to/UMLS/META/MRSTY.RRF \
    --mesh-codes    /path/to/meshcodes.xlsx \
    --output-dir    domain_files/<domain>/
```

**Verify the exact flag names against `python build_semantic_types.py
--help` before running** — the above reflects the script's general
inputs/outputs, not a confirmed-current CLI signature.

This produces:
```
domain_files/<domain>/
    mesh_to_semantic_type.csv   # dataset, mesh_descriptor, cui, tui, sty, stn
    umls_semantic_type.json     # per-label semantic type(s), used directly by the pipeline
```

Run once per domain (including `bioasq`).

---

## Notes on NaN Handling

The `allMeSH_2022.json` file contains `NaN` values in some fields. Before running the pipeline, clean the file:

```bash
sed 's/: NaN,/: null,/g; s/: NaN}/: null}/g; s/\bNaN\b/null/g' \
    allMeSH_2022.json > allMeSH_2022_clean.json
```

Use `allMeSH_2022_clean.json` for all downstream scripts.

---

## Label Format

All datasets use MeSH **Unique IDs** as labels (e.g., `D006801`, `D005260`).
The full MeSH code list with headings and tree numbers is available at:
https://www.nlm.nih.gov/mesh/

---

## UMLS Requirement

Building the semantic-type files requires UMLS access. Apply for a license at:
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

Required UMLS files:
- `MRCONSO.RRF` — concept names
- `MRSTY.RRF` — semantic types
