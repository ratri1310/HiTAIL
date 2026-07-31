"""
Standalone script. No dependency on any other file in this project.

Precomputes, once (these don't change during training):
  IC(a)      for every hierarchy node ever referenced as a parent/grandparent
  omega_y    for every leaf label

IC(a) = -log( (mass(a) + eps) / (total_mass + eps) )
mass(a) = f_a (if a itself is an observed label) + sum of f_y over every
          label y whose FULL parent-set or grandparent-set contains a.
omega_y = omega_y_freq * omega_y_hier
omega_y_freq = 1 / log(f_y + c),  c = 2
omega_y_hier = eta1 * mean(IC(a) for a in A1(y)) + eta2 * mean(IC(a) for a in A2(y))
             eta1 = 0.7, eta2 = 0.3

Usage:
    python precompute_ic_and_weights.py --domain immunology
"""

import argparse
import json
import math
from collections import defaultdict

DATA_ROOT = "/localscratch/Users/ratri/Bert/bertldl/datasets/domain_files/"
GLOBAL_FULL_PARENT = "/localscratch/Users/ratri/Bert/datasets/full_parent_sets.json"
GLOBAL_FULL_GRANDPARENT = "/localscratch/Users/ratri/Bert/datasets/full_grandparent_sets.json"

C_FREQ = 2.0
ETA1, ETA2 = 0.7, 0.3
EPS = 1e-8


def compute_ic_scores(label_freq: dict, parent_sets: dict, grandparent_sets: dict):
    """Returns {node_uid: IC value} for every node referenced as a parent or grandparent."""
    descendants = defaultdict(set)  # node -> set of labels that descend from it
    for y, parents in parent_sets.items():
        for a in parents:
            descendants[a].add(y)
    for y, gps in grandparent_sets.items():
        for a in gps:
            descendants[a].add(y)

    total_mass = sum(label_freq.values())
    ic_scores = {}
    for a, desc_labels in descendants.items():
        mass = label_freq.get(a, 0)  # a's own frequency, only if a is itself an observed label
        for y in desc_labels:
            if y == a:
                continue
            mass += label_freq.get(y, 0)
        ic_scores[a] = -math.log((mass + EPS) / (total_mass + EPS))
    return ic_scores


def compute_omega_y(label_freq: dict, parent_sets: dict, grandparent_sets: dict, ic_scores: dict):
    omega = {}
    for y, freq in label_freq.items():
        omega_freq = 1.0 / math.log(freq + C_FREQ)

        a1 = parent_sets.get(y, [])
        a2 = grandparent_sets.get(y, [])
        ic_a1 = sum(ic_scores.get(a, 0.0) for a in a1) / len(a1) if a1 else 0.0
        ic_a2 = sum(ic_scores.get(a, 0.0) for a in a2) / len(a2) if a2 else 0.0
        omega_hier = ETA1 * ic_a1 + ETA2 * ic_a2

        omega[y] = omega_freq * omega_hier
    return omega


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--full-parent-sets", default=GLOBAL_FULL_PARENT)
    p.add_argument("--full-grandparent-sets", default=GLOBAL_FULL_GRANDPARENT)
    args = p.parse_args()

    domain_dir = f"{args.data_root}/{args.domain}"
    with open(f"{domain_dir}/label_freq.json") as f:
        label_freq = json.load(f)
    with open(args.full_parent_sets) as f:
        global_parent_sets = json.load(f)
    with open(args.full_grandparent_sets) as f:
        global_grandparent_sets = json.load(f)

    # restrict to this domain's label vocabulary
    domain_labels = set(label_freq.keys())
    parent_sets = {y: [a for a in global_parent_sets.get(y, [])] for y in domain_labels}
    grandparent_sets = {y: [a for a in global_grandparent_sets.get(y, [])] for y in domain_labels}

    print(f"Domain {args.domain}: {len(domain_labels)} labels")

    ic_scores = compute_ic_scores(label_freq, parent_sets, grandparent_sets)
    omega_y = compute_omega_y(label_freq, parent_sets, grandparent_sets, ic_scores)

    print(f"  computed IC for {len(ic_scores)} hierarchy nodes")
    print(f"  computed omega_y for {len(omega_y)} labels, "
          f"range [{min(omega_y.values()):.4f}, {max(omega_y.values()):.4f}]")

    with open(f"{domain_dir}/ic_scores.json", "w") as f:
        json.dump(ic_scores, f)
    with open(f"{domain_dir}/omega_y.json", "w") as f:
        json.dump(omega_y, f)

    print(f"  wrote ic_scores.json, omega_y.json to {domain_dir}")


if __name__ == "__main__":
    main()
