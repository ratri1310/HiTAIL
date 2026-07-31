"""
Loads two DIFFERENT representations of the same hierarchy, for two different
purposes -- this distinction matters, so it's kept explicit rather than
silently collapsed into one:

  1. FULL polyhierarchy sets (from full_parent_sets.json/full_grandparent_
     sets.json) -- used by L_SSL's pair construction and the IC/omega_y
     precomputation. A label can have multiple parents/grandparents here.

  2. A SINGLE canonical index per label (reusing the old fast-path-resolved
     parent_map.json/grandparent_map.json) -- used ONLY to build the
     fixed-size q_A1/q_A2 distribution tables and pool predicted mass in
     L_dist, which need one consistent row per ancestor node, not a
     variable-size set. This reuse is fine for this narrower purpose --
     the fast-path resolution was never "wrong", it was just insufficient
     ALONE for the parts of the design that need the full set.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class HierarchyMaps:
    # full sets, keyed by label UID -- for L_SSL / IC
    full_parent_sets: Dict[str, Set[str]]
    full_grandparent_sets: Dict[str, Set[str]]
    semantic_type_sets: Dict[str, Set[str]]  # a label can have >1 semantic type too

    # single canonical index, keyed by label INDEX (0..num_labels-1) -- for L_dist pooling
    parent_idx: Dict[int, int]
    grandparent_idx: Dict[int, int]
    num_parent_rows: int
    num_grandparent_rows: int

    # semantic-type row index (SINGLE, for prototype table row count / L_ST), keyed by UID
    semantic_type_index: Dict[str, int]
    num_semantic_types: int


def load_hierarchy(full_parent_path, full_grandparent_path, semantic_type_path,
                     legacy_parent_map_path, legacy_grandparent_map_path, label_map: dict):
    with open(full_parent_path) as f:
        raw_parents = json.load(f)
    with open(full_grandparent_path) as f:
        raw_grandparents = json.load(f)
    with open(semantic_type_path) as f:
        raw_semantic = json.load(f)
    with open(legacy_parent_map_path) as f:
        legacy_parent = json.load(f)
    with open(legacy_grandparent_map_path) as f:
        legacy_grandparent = json.load(f)

    full_parent_sets = {uid: set(v) for uid, v in raw_parents.items()}
    full_grandparent_sets = {uid: set(v) for uid, v in raw_grandparents.items()}

    semantic_type_sets = {}
    for uid, v in raw_semantic.items():
        if isinstance(v, list):
            semantic_type_sets[uid] = set(v) if v else {"UNKNOWN_ST"}
        elif v:
            semantic_type_sets[uid] = {v}
        else:
            semantic_type_sets[uid] = {"UNKNOWN_ST"}

    all_st = sorted(set().union(*semantic_type_sets.values())) if semantic_type_sets else []
    semantic_type_index = {st: i for i, st in enumerate(all_st)}

    all_legacy_parents = sorted(set(v for v in legacy_parent.values() if v is not None))
    all_legacy_gp = sorted(set(v for v in legacy_grandparent.values() if v is not None))
    parent_row_index = {uid: i for i, uid in enumerate(all_legacy_parents)}
    gp_row_index = {uid: i for i, uid in enumerate(all_legacy_gp)}

    parent_idx, grandparent_idx = {}, {}
    for uid, idx in label_map.items():
        p = legacy_parent.get(uid)
        g = legacy_grandparent.get(uid)
        parent_idx[idx] = parent_row_index.get(p, -1)
        grandparent_idx[idx] = gp_row_index.get(g, -1)

    return HierarchyMaps(
        full_parent_sets=full_parent_sets,
        full_grandparent_sets=full_grandparent_sets,
        semantic_type_sets=semantic_type_sets,
        parent_idx=parent_idx,
        grandparent_idx=grandparent_idx,
        num_parent_rows=len(all_legacy_parents),
        num_grandparent_rows=len(all_legacy_gp),
        semantic_type_index=semantic_type_index,
        num_semantic_types=len(all_st),
    )


def build_label_index_maps(label_map: dict, hierarchy: HierarchyMaps):
    """
    Converts UID-keyed sets into LABEL-INDEX-keyed sets (of semantic-type/
    parent/grandparent ROW indices), for use in losses.py's build_document_
    ontology_sets(). Returns three dicts: label_idx -> set of row indices.
    """
    label_to_semantic, label_to_parents, label_to_gp = {}, {}, {}
    inv_label_map = {v: k for k, v in label_map.items()}

    all_legacy_parents_seen = set()
    all_legacy_gp_seen = set()
    # build fresh row indices over the FULL sets actually used in this domain
    # (separate numbering from the legacy single-value index above -- this
    # one just needs to be internally consistent for pair-overlap checks,
    # not aligned with any fixed-size table)
    for idx, uid in inv_label_map.items():
        for p in hierarchy.full_parent_sets.get(uid, set()):
            all_legacy_parents_seen.add(p)
        for g in hierarchy.full_grandparent_sets.get(uid, set()):
            all_legacy_gp_seen.add(g)

    for idx, uid in inv_label_map.items():
        label_to_semantic[idx] = {hierarchy.semantic_type_index[s]
                                   for s in hierarchy.semantic_type_sets.get(uid, set())
                                   if s in hierarchy.semantic_type_index}
        label_to_parents[idx] = set(hierarchy.full_parent_sets.get(uid, set()))
        label_to_gp[idx] = set(hierarchy.full_grandparent_sets.get(uid, set()))

    return label_to_semantic, label_to_parents, label_to_gp
