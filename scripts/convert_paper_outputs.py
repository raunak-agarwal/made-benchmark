#!/usr/bin/env python3
"""Convert raw MADE model outputs into the predictions format used by evaluate.py.

Two input formats are supported:

1. Generative / prompting outputs (``--format generative``), one JSON object per line:
     {"idx": ..., "labels": [...], "predictions": [...],
      "logprobs": {label: {token: [[token, logprob], ...top-5...]}}}

   Per label the script computes
     - uncertainty : summed per-token entropy over the top-5 alternatives plus the
                     residual probability mass (the paper's best U_info metric), and
     - score       : the per-token mean probability exp(mean of chosen-token log-probs).
   The per-sample uncertainty is the mean per-label uncertainty over predicted labels.
   Rows with missing or malformed logprobs keep their predictions but get no UQ fields,
   so evaluate.py excludes them from the UQ metrics (as the paper's evaluation did).

2. Discriminative outputs (``--format discriminative``), one JSON object per line:
     {"idx": ..., "labels": [...], "predicted_labels": [...],
      "label_probabilities": {label: prob, ...}}

   Scores are the sigmoid probabilities over the full vocabulary, per-label uncertainty
   is the binary entropy of the probability (for predicted labels), and the per-sample
   uncertainty is the summed binary entropy over all labels (as in the paper).

Usage:
  python convert_paper_outputs.py raw_output.jsonl predictions.jsonl --format generative
"""

import argparse
import json
import math


def token_entropy(top_logprobs):
    """Entropy over the top-k alternatives plus the residual probability mass."""
    probs = [math.exp(lp) for _, lp in top_logprobs]
    residual = max(0.0, 1.0 - sum(probs))
    probs.append(residual)
    return -sum(p * math.log(p) for p in probs if p > 0)


def binary_entropy(p, eps=1e-15):
    p = min(max(p, eps), 1 - eps)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def malformed_logprobs(logprobs):
    """True when logprobs are missing or any label has empty token/alternative lists."""
    if not logprobs:
        return True
    for tokens in logprobs.values():
        if not tokens:
            return True
        for alts in tokens.values():
            if not alts:
                return True
    return False


def convert_generative(row):
    predictions = row.get("predictions") or []
    logprobs = row.get("logprobs") or {}
    if malformed_logprobs(logprobs):
        return {"idx": row["idx"], "predictions": predictions}
    scores, uncertainties = {}, {}
    for label, tokens in logprobs.items():
        chosen = [alts[0][1] for alts in tokens.values()]
        scores[label] = math.exp(sum(chosen) / len(chosen))
        uncertainties[label] = sum(token_entropy(alts) for alts in tokens.values())
    label_unc = [uncertainties[l] for l in predictions if l in uncertainties]
    uncertainty = sum(label_unc) / len(label_unc) if label_unc else 0.0
    return {"idx": row["idx"], "predictions": predictions, "scores": scores,
            "uncertainties": uncertainties, "uncertainty": uncertainty}


def convert_discriminative(row):
    probabilities = row.get("label_probabilities") or {}
    predictions = row.get("predicted_labels") or []
    uncertainties = {l: binary_entropy(probabilities[l]) for l in predictions if l in probabilities}
    uncertainty = sum(binary_entropy(p) for p in probabilities.values())
    return {"idx": row["idx"], "predictions": predictions, "scores": probabilities,
            "uncertainties": uncertainties, "uncertainty": uncertainty}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", help="Raw model output JSONL")
    parser.add_argument("output", help="Predictions JSONL for evaluate.py")
    parser.add_argument("--format", choices=("generative", "discriminative"),
                        default="generative")
    args = parser.parse_args()

    convert = convert_generative if args.format == "generative" else convert_discriminative
    n = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(convert(json.loads(line))) + "\n")
            n += 1
    print(f"Converted {n} rows -> {args.output}")


if __name__ == "__main__":
    main()
