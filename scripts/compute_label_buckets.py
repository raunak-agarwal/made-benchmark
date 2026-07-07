#!/usr/bin/env python3
"""Recompute the head/medium/tail/extreme-tail label buckets from the training split.

Buckets are defined by training-set label frequency (fraction of training samples
containing the label): head >= 1%, medium 0.1-1%, tail 0.01-0.1%, extreme tail < 0.01%.
The shipped head_medium_tail.json was produced this way from train.jsonl
(144 / 481 / 348 / 181 labels).

Usage:
  python compute_label_buckets.py --train train.jsonl --labels labels.json \
      --output head_medium_tail.json
"""

import argparse
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", required=True, help="train.jsonl")
    parser.add_argument("--labels", required=True, help="labels.json (label vocabulary)")
    parser.add_argument("--output", default="head_medium_tail.json")
    parser.add_argument("--cutoff-head", type=float, default=0.01)
    parser.add_argument("--cutoff-tail", type=float, default=0.001)
    parser.add_argument("--cutoff-extreme-tail", type=float, default=0.0001)
    args = parser.parse_args()

    label_list = json.load(open(args.labels))
    counts = Counter()
    n_samples = 0
    with open(args.train) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counts.update(json.loads(line)["labels"])
            n_samples += 1

    buckets = {"head": [], "medium": [], "tail": [], "extreme_tail": []}
    for label in label_list:
        freq = counts.get(label, 0) / n_samples
        if freq < args.cutoff_extreme_tail:
            buckets["extreme_tail"].append(label)
        elif freq >= args.cutoff_head:
            buckets["head"].append(label)
        elif freq < args.cutoff_tail:
            buckets["tail"].append(label)
        else:
            buckets["medium"].append(label)

    for name, labels in buckets.items():
        labels.sort()
        print(f"{name}: {len(labels)} labels")
    with open(args.output, "w") as f:
        json.dump(buckets, f, indent=2)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
