"""
Standalone script. No dependency on any other file in this project.

Builds the FULL polyhierarchy parent/grandparent SETS per label — not the
single fast-path-resolved value used by the old architecture. This design's
math (set-intersection pair rule in Step 4, IC averaging in Step 6) needs
every valid parent/grandparent, not one tie-broken winner. No tie-breaking
logic is needed here at all — that's the whole point of keeping the set.

Input (same global files as before):
    global_mesh_tree.json           {uid: [tree_number, ...]}
    global_tree_number_to_uid.json  {tree_number: uid}

Output:
    full_parent_sets.json        {uid: [parent_uid, ...]}       (A1(y))
    full_grandparent_sets.json   {uid: [grandparent_uid, ...]}  (A2(y))

Usage:
    python build_full_parent_sets.py
"""

import argparse
import json

TREE_JSON = "/path_to/global_mesh_tree.json"
INDEX_JSON = "/path_to/global_tree_number_to_uid.json"
OUT_DIR = "/path_to/datasets"


def ancestor_tree_number(tn: str, levels_up: int):
    parts = tn.split(".")
    if len(parts) <= levels_up:
        return None
    return ".".join(parts[: len(parts) - levels_up])


def build_full_sets(uid_to_tree, tree_to_uid, levels_up):
    """Returns {uid: sorted list of ALL distinct ancestor UIDs at this level}."""
    result = {}
    for uid, tns in uid_to_tree.items():
        ancestors = set()
        for tn in tns:
            atn = ancestor_tree_number(tn, levels_up)
            if atn is None:
                continue
            auid = tree_to_uid.get(atn)
            if auid is not None:
                ancestors.add(auid)
        result[uid] = sorted(ancestors)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree-json", default=TREE_JSON)
    p.add_argument("--index-json", default=INDEX_JSON)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()

    with open(args.tree_json) as f:
        uid_to_tree = json.load(f)
    with open(args.index_json) as f:
        tree_to_uid = json.load(f)

    print(f"Loaded {len(uid_to_tree)} descriptors")

    parent_sets = build_full_sets(uid_to_tree, tree_to_uid, 1)
    grandparent_sets = build_full_sets(uid_to_tree, tree_to_uid, 2)

    n_multi_parent = sum(1 for v in parent_sets.values() if len(v) > 1)
    n_multi_gp = sum(1 for v in grandparent_sets.values() if len(v) > 1)
    n_empty_parent = sum(1 for v in parent_sets.values() if len(v) == 0)
    n_empty_gp = sum(1 for v in grandparent_sets.values() if len(v) == 0)

    print(f"parent sets:      {n_multi_parent} labels have >1 distinct parent, "
          f"{n_empty_parent} have none")
    print(f"grandparent sets: {n_multi_gp} labels have >1 distinct grandparent, "
          f"{n_empty_gp} have none")

    with open(f"{args.out_dir}/full_parent_sets.json", "w") as f:
        json.dump(parent_sets, f)
    with open(f"{args.out_dir}/full_grandparent_sets.json", "w") as f:
        json.dump(grandparent_sets, f)

    print(f"wrote full_parent_sets.json, full_grandparent_sets.json to {args.out_dir}")


if __name__ == "__main__":
    main()
