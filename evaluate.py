#!/usr/bin/env python3
"""Official evaluation script for the MADE benchmark.

Computes the metrics reported in Table 2 of the paper:

  - Macro F1 (overall + head / medium / tail / extreme-tail label buckets)
  - J     : example-based Jaccard score
  - PRR   : Prediction Rejection Rate (sample-level, Jaccard-based)
  - rho   : Spearman correlation between per-label uncertainty and correctness
  - ECE+  : positive-class expected calibration error

Predictions file: JSONL, one object per test sample:

  {
    "idx": 20031503,                                  # sample id (matches test.jsonl)
    "predictions": ["A04", "A0401", ...],             # predicted label codes
    "scores": {"A04": 0.93, "A0401": 0.71, ...},      # optional: per-label confidence in [0, 1]
    "uncertainties": {"A04": 0.11, ...},              # optional: per-label uncertainty (any scale)
    "uncertainty": 0.28                               # optional: per-sample uncertainty (any scale)
  }

Only "idx" and "predictions" are required; they are enough for Macro F1 and J.
The UQ metrics additionally need:

  - ECE+     : "scores" (calibrated probabilities in [0, 1]). Scores may cover any subset
               of the label vocabulary (e.g. the full probability vector for discriminative
               models, or only the generated labels for generative models). A true label
               with no score contributes confidence 0.
  - Spearman : "uncertainties" if present, else 1 - "scores" (over predicted labels).
  - PRR      : "uncertainty" if present, else the mean of "uncertainties" over predicted
               labels, else the mean of 1 - "scores" over predicted labels.

Macro F1 and J are computed over every test sample. The UQ metrics are computed over the
samples that provide the required fields (matching the paper, which excluded samples with
missing/malformed log-probabilities from the UQ evaluation); the sample and pair counts
they were computed on are always reported.

In the paper, per-label uncertainty for generative models is the summed top-5 token entropy
of the label's tokens (the best-performing U_info metric), per-label confidence is the
per-token mean probability exp(mean of token log-probs), and per-sample uncertainty is the
mean per-label uncertainty over the predicted labels. Use scripts/convert_paper_outputs.py
to reproduce these from raw model outputs.

Usage:
  python evaluate.py --predictions preds.jsonl --data-dir /path/to/MADE_Benchmark_Public
  python evaluate.py --predictions preds.jsonl --test-file test.jsonl --labels labels.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Predictive metrics
# ---------------------------------------------------------------------------

def binary_matrices(gold_lists, pred_lists, label_list):
    """Multi-hot matrices over `label_list`; labels outside the vocabulary are ignored."""
    label_to_idx = {label: i for i, label in enumerate(label_list)}
    y_true = np.zeros((len(gold_lists), len(label_list)), dtype=bool)
    y_pred = np.zeros((len(gold_lists), len(label_list)), dtype=bool)
    for i, (gold, pred) in enumerate(zip(gold_lists, pred_lists)):
        for label in gold:
            j = label_to_idx.get(label)
            if j is not None:
                y_true[i, j] = True
        for label in pred:
            j = label_to_idx.get(label)
            if j is not None:
                y_pred[i, j] = True
    return y_true, y_pred


def macro_f1(y_true, y_pred):
    """Unweighted mean of per-label F1; zero-division yields 0 (as in the paper)."""
    tp = (y_pred & y_true).sum(axis=0).astype(float)
    fp = (y_pred & ~y_true).sum(axis=0).astype(float)
    fn = (~y_pred & y_true).sum(axis=0).astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def example_jaccard(gold_lists, pred_lists):
    """Mean per-sample Jaccard between predicted and gold label sets."""
    scores = []
    for gold, pred in zip(gold_lists, pred_lists):
        gold_set, pred_set = set(gold), set(pred)
        union = gold_set | pred_set
        scores.append(len(gold_set & pred_set) / len(union) if union else 0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Uncertainty metrics
# ---------------------------------------------------------------------------

def per_sample_jaccard(y_true, y_pred):
    """Vectorized per-sample Jaccard (empty union counts as 1.0, as in the PRR code)."""
    intersection = (y_true & y_pred).sum(axis=1)
    union = (y_true | y_pred).sum(axis=1)
    return np.where(union > 0, intersection / np.maximum(union, 1), 1.0)


def _rejection_auc(sample_quality, uncertainty, coverage_steps=20):
    """AUC of mean quality vs. coverage when rejecting the most uncertain samples first."""
    n = len(sample_quality)
    order = np.argsort(uncertainty)  # most certain first; the tail gets rejected
    quality_sorted = sample_quality[order]
    coverages = np.linspace(1, 0, coverage_steps)
    curve = np.full(coverage_steps, np.nan)
    for i, c in enumerate(coverages):
        n_keep = int(np.ceil(c * n))
        if n_keep > 0:
            curve[i] = quality_sorted[:n_keep].mean()
    mask = ~np.isnan(curve)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return trapezoid(curve[mask][::-1], coverages[mask][::-1])


def prr(y_true, y_pred, uncertainty, coverage_steps=20, seed=42):
    """Prediction Rejection Rate: (AUC_model - AUC_random) / (AUC_oracle - AUC_random)."""
    quality = per_sample_jaccard(y_true, y_pred)
    auc_model = _rejection_auc(quality, uncertainty, coverage_steps)
    auc_oracle = _rejection_auc(quality, 1.0 - quality, coverage_steps)
    rng = np.random.RandomState(seed)
    auc_random = _rejection_auc(quality, rng.permutation(uncertainty), coverage_steps)
    if auc_oracle - auc_random == 0:
        return float("nan")
    return float((auc_model - auc_random) / (auc_oracle - auc_random))


def spearman_uncertainty(gold_lists, pred_lists, label_uncertainties):
    """Spearman rho between per-label uncertainty and binary correctness.

    Pooled over all (sample, predicted label) pairs that have an uncertainty value,
    where correctness is 1 if the predicted label is in the gold set.
    """
    correct, uncertain = [], []
    for gold, pred, unc in zip(gold_lists, pred_lists, label_uncertainties):
        gold_set = set(gold)
        for label in pred:
            u = unc.get(label)
            if u is None:
                continue
            correct.append(1 if label in gold_set else 0)
            uncertain.append(float(u))
    if len(set(correct)) < 2 or len(set(uncertain)) < 2:
        return float("nan"), 0
    rho, _ = spearmanr(correct, uncertain)
    return float(rho), len(correct)


def ece_plus(gold_lists, score_dicts, label_list):
    """Positive-class ECE: mean over labels of (1 - mean confidence on true positives).

    For each label c, restrict to samples where c is a gold label; a missing score counts
    as confidence 0 (a positive the model did not predict). Labels with no positive test
    instances are skipped.
    """
    positive_conf = {label: [] for label in label_list}
    label_set = set(label_list)
    for gold, scores in zip(gold_lists, score_dicts):
        for label in gold:
            if label in label_set:
                positive_conf[label].append(float(scores.get(label, 0.0)))
    per_label = [1.0 - float(np.mean(confs)) for confs in positive_conf.values() if confs]
    return float(np.mean(per_label)), len(per_label)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def sample_uncertainty(row):
    """Per-sample uncertainty with the documented precedence."""
    if row.get("uncertainty") is not None:
        return float(row["uncertainty"])
    pred = row.get("predictions") or []
    unc = row.get("uncertainties") or {}
    values = [float(unc[label]) for label in pred if label in unc]
    if values:
        return float(np.mean(values))
    scores = row.get("scores") or {}
    values = [1.0 - float(scores[label]) for label in pred if label in scores]
    if values:
        return float(np.mean(values))
    return None


def evaluate(pred_rows, gold_rows, label_list, buckets, skip_missing=False):
    pred_by_idx = {row["idx"]: row for row in pred_rows}
    gold_by_idx = {row["idx"]: row for row in gold_rows}

    extra = set(pred_by_idx) - set(gold_by_idx)
    if extra:
        print(f"WARNING: {len(extra)} prediction idx not in the test set (ignored).", file=sys.stderr)

    missing = [idx for idx in gold_by_idx if idx not in pred_by_idx]
    if missing:
        action = "excluded from evaluation" if skip_missing else "scored as empty predictions"
        print(f"WARNING: {len(missing)}/{len(gold_by_idx)} test samples have no prediction "
              f"({action}).", file=sys.stderr)

    eval_idx = [idx for idx in gold_by_idx
                if not (skip_missing and idx not in pred_by_idx)]
    gold_lists = [gold_by_idx[idx]["labels"] for idx in eval_idx]
    rows = [pred_by_idx.get(idx, {"predictions": []}) for idx in eval_idx]
    pred_lists = [row.get("predictions") or [] for row in rows]

    results = {"n_evaluated": len(eval_idx), "n_missing": len(missing)}

    vocabulary = set(label_list)
    n_unknown = sum(1 for pred in pred_lists for label in pred if label not in vocabulary)
    if n_unknown:
        print(f"WARNING: {n_unknown} predicted labels are not in the label vocabulary "
              f"(ignored for Macro F1; they still count against the Jaccard union).", file=sys.stderr)

    # --- predictive performance ---------------------------------------------
    y_true, y_pred = binary_matrices(gold_lists, pred_lists, label_list)
    label_to_col = {label: i for i, label in enumerate(label_list)}
    results["macro_f1"] = {"overall": macro_f1(y_true, y_pred)}
    for bucket_name in ("head", "medium", "tail", "extreme_tail"):
        cols = [label_to_col[l] for l in buckets[bucket_name] if l in label_to_col]
        results["macro_f1"][bucket_name] = macro_f1(y_true[:, cols], y_pred[:, cols])
    results["jaccard"] = example_jaccard(gold_lists, pred_lists)

    # --- uncertainty quantification ------------------------------------------
    # Each UQ metric is computed over the samples that provide the fields it needs;
    # samples without them (e.g. malformed log-probabilities) are excluded, as in the paper.
    uncertainties = [sample_uncertainty(row) for row in rows]
    with_unc = [i for i, u in enumerate(uncertainties) if u is not None]
    if with_unc:
        unc = np.array([uncertainties[i] for i in with_unc], dtype=float)
        results["prr"] = prr(y_true[with_unc], y_pred[with_unc], unc)
        results["prr_n_samples"] = len(with_unc)
    else:
        print("NOTE: PRR skipped (no per-sample uncertainty or scores).", file=sys.stderr)

    label_unc = [row.get("uncertainties")
                 or {l: 1.0 - float(s) for l, s in (row.get("scores") or {}).items()}
                 for row in rows]
    if any(unc for unc in label_unc):
        rho, n_pairs = spearman_uncertainty(gold_lists, pred_lists, label_unc)
        results["spearman"] = rho
        results["spearman_n_pairs"] = n_pairs
    else:
        print("NOTE: Spearman skipped (no per-label uncertainties or scores).", file=sys.stderr)

    with_scores = [i for i, row in enumerate(rows) if row.get("scores")]
    if with_scores:
        results["ece_plus"], results["ece_plus_n_labels"] = ece_plus(
            [gold_lists[i] for i in with_scores],
            [rows[i]["scores"] for i in with_scores], label_list)
        results["ece_plus_n_samples"] = len(with_scores)
    else:
        print("NOTE: ECE+ skipped (no per-label scores).", file=sys.stderr)

    return results


def print_report(results):
    f1 = results["macro_f1"]
    print()
    print(f"Evaluated samples : {results['n_evaluated']}"
          + (f"  ({results['n_missing']} missing)" if results["n_missing"] else ""))
    print()
    print("Predictive performance")
    print(f"  Macro F1  overall      : {f1['overall']:.4f}")
    print(f"  Macro F1  head         : {f1['head']:.4f}")
    print(f"  Macro F1  medium       : {f1['medium']:.4f}")
    print(f"  Macro F1  tail         : {f1['tail']:.4f}")
    print(f"  Macro F1  extreme tail : {f1['extreme_tail']:.4f}")
    print(f"  Jaccard (J)            : {results['jaccard']:.4f}")
    print()
    print("Uncertainty quantification")
    if "prr" in results:
        print(f"  PRR                    : {results['prr']:.4f}"
              f"   ({results['prr_n_samples']} samples with uncertainty)")
    if "spearman" in results:
        print(f"  Spearman rho           : {results['spearman']:.4f}"
              f"   ({results['spearman_n_pairs']} label predictions)")
    if "ece_plus" in results:
        print(f"  ECE+                   : {results['ece_plus']:.4f}"
              f"   ({results['ece_plus_n_samples']} samples with scores, "
              f"{results['ece_plus_n_labels']} labels with positives)")
    if not any(k in results for k in ("prr", "spearman", "ece_plus")):
        print("  (skipped - provide 'scores'/'uncertainties' to compute PRR, Spearman, ECE+)")
    print()


def main():
    parser = argparse.ArgumentParser(description="MADE benchmark evaluation (Table 2 metrics)")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL (see module docstring)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory with test.jsonl and labels.json (the HF dataset download)")
    parser.add_argument("--test-file", default=None, help="Gold test JSONL (overrides --data-dir)")
    parser.add_argument("--labels", default=None, help="labels.json (overrides --data-dir)")
    parser.add_argument("--buckets", default=str(REPO_DIR / "head_medium_tail.json"),
                        help="Label frequency buckets JSON (default: shipped head_medium_tail.json)")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Evaluate only on samples present in the predictions file "
                             "(default: missing samples are scored as empty predictions)")
    parser.add_argument("--output", default=None, help="Optional path to write results JSON")
    args = parser.parse_args()

    test_file = args.test_file or (Path(args.data_dir) / "test.jsonl" if args.data_dir else None)
    labels_file = args.labels or (Path(args.data_dir) / "labels.json" if args.data_dir else None)
    if not test_file or not labels_file:
        parser.error("provide --data-dir, or both --test-file and --labels")

    label_list = json.load(open(labels_file))
    buckets = json.load(open(args.buckets))
    gold_rows = load_jsonl(test_file)
    pred_rows = load_jsonl(args.predictions)

    results = evaluate(pred_rows, gold_rows, label_list, buckets, skip_missing=args.skip_missing)
    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
