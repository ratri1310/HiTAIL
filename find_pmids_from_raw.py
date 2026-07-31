"""
test.jsonl only kept "text" and "labels" during processing -- PMIDs were
dropped. This recovers them by matching each document's FULL text (from
test.jsonl directly, not the truncated snippet in the case-study file)
against the original raw source file, which almost certainly still has them.

Field names in the raw file aren't assumed -- this checks several common
candidates and reports which ones actually matched, rather than guessing
silently.

Usage:
    python -m hspdl.find_pmids_from_raw --domain immunology
"""

import argparse
import json
import os

from .config import DomainConfig
from .data import load_jsonl

RAW_SOURCE_CANDIDATES = {
    "immunology": "/localscratch/Users/ratri/Bert/datasets/immunology_database_v2.json",
    "neurology": "/localscratch/Users/ratri/Bert/datasets/neurology_database_v2.json",
    "embryology": "/localscratch/Users/ratri/Bert/datasets/embryology_database_v2.json",
}

TEXT_FIELD_CANDIDATES = ["text", "abstract", "AbstractText", "Abstract"]
PMID_FIELD_CANDIDATES = ["pmid", "PMID", "PMID_x", "pubmed_id", "PubmedID"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--raw-source", default="")
    p.add_argument("--case-study-json", default="")
    args = p.parse_args()

    raw_path = args.raw_source or RAW_SOURCE_CANDIDATES.get(args.domain, "")
    if not raw_path or not os.path.exists(raw_path):
        print(f"ERROR: raw source file not found at '{raw_path}'. "
              f"Pass the correct path via --raw-source.")
        return

    with open(raw_path) as f:
        raw_data = json.load(f)
    if isinstance(raw_data, dict):
        raw_records = list(raw_data.values()) if all(isinstance(v, dict) for v in raw_data.values()) else [raw_data]
    else:
        raw_records = raw_data

    print(f"Loaded {len(raw_records)} raw records from {raw_path}")

    # detect which field names this file actually uses
    sample = raw_records[0] if raw_records else {}
    text_field = next((f for f in TEXT_FIELD_CANDIDATES if f in sample), None)
    pmid_field = next((f for f in PMID_FIELD_CANDIDATES if f in sample), None)

    if not text_field or not pmid_field:
        print(f"WARNING: could not confidently detect field names. Sample record keys: {list(sample.keys())}")
        print(f"detected text_field={text_field}, pmid_field={pmid_field} -- "
              f"check these are right before trusting the output.")
        if not text_field or not pmid_field:
            return

    print(f"Using text_field='{text_field}', pmid_field='{pmid_field}'")

    # build a lookup: exact full abstract text -> PMID
    text_to_pmid = {}
    for rec in raw_records:
        t = rec.get(text_field)
        pmid = rec.get(pmid_field)
        if t and pmid:
            text_to_pmid[t.strip()] = pmid

    print(f"Built lookup with {len(text_to_pmid)} entries")

    dcfg = DomainConfig.build(args.domain)
    test_records = load_jsonl(os.path.join(dcfg.data_dir, "test.jsonl"))

    case_study_path = args.case_study_json or f"hspdl/runs/case_studies/{args.domain}_case_studies.json"
    if not os.path.exists(case_study_path):
        print(f"ERROR: case study file not found at '{case_study_path}'")
        return
    with open(case_study_path) as f:
        cases = json.load(f)

    updated = 0
    for case in cases:
        pmid_str = case["pmid"]
        if not pmid_str.startswith("doc_index_"):
            continue  # already had a real pmid
        doc_idx = int(pmid_str.replace("doc_index_", ""))
        full_text = test_records[doc_idx]["text"].strip()
        matched_pmid = text_to_pmid.get(full_text)
        if matched_pmid:
            case["pmid"] = str(matched_pmid)
            updated += 1
        else:
            print(f"  no match found for doc_index_{doc_idx} (abstract starts: '{full_text[:80]}...')")

    print(f"\nRecovered PMIDs for {updated}/{len(cases)} cases")

    out_path = case_study_path.replace(".json", "_with_pmids.json")
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
