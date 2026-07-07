# MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events

[**📄 Paper (ACL 2026)**](https://aclanthology.org/2026.acl-long.2148/) ·
[**🤗 Dataset**](https://huggingface.co/datasets/ragarwal/MADE-Multilabel-Benchmark) ·
[**🖥️ Interactive Demo**](https://hhi.fraunhofer.de/aml-demonstrator/made-benchmark)

MADE is a living multi-label text classification (MLTC) benchmark built from FDA **m**edical device **ad**verse **e**vent reports. Each report describes a real-world device incident and is annotated with hierarchical [IMDRF](https://www.imdrf.org/) problem codes. The benchmark is continuously updated with newly published reports, so future models can be evaluated on data that post-dates their training — reducing the risk that strong results come from contamination rather than generalization.

Beyond predictive performance, MADE treats **uncertainty quantification (UQ)** as a first-class evaluation target: models are scored not only on *what* they predict, but on whether their confidence is a reliable signal for ranking mistakes and calibrating trust.

**Accepted at ACL 2026 (Main).**

## Benchmark at a glance

| | |
|---|---|
| Reports | 488,273 (FDA, 2015 – mid-2025) |
| Labels | 1,154 hierarchical IMDRF codes (3 levels; `A` = device problems, `E` = health effects) |
| Labels per sample | 8.79 on average |
| Splits | temporal — train 2015–2023 · validation Jan–Jun 2024 · test Jul 2024 – Jun 2025 |
| Evaluation test set | 10,288 reports (representative stratified subset of the 118,177-report test period) |
| Best overall Macro-F1 | 0.54 — substantial headroom remains |

The label distribution is heavily long-tailed. For evaluation, labels are bucketed by training-set frequency: **head** (>1%, 144 labels), **medium** (0.1–1%, 481), **tail** (0.01–0.1%, 348), and **extreme tail** (<0.01%, 181).

## Dataset

Download from [HuggingFace](https://huggingface.co/datasets/ragarwal/MADE-Multilabel-Benchmark). The release contains:

| File | Content |
|---|---|
| `train.jsonl` / `val.jsonl` / `test.jsonl` | the temporal splits (`test.jsonl` is the 10,288-sample evaluation set) |
| `labels.json` | the 1,154-code label vocabulary |
| `imdrf_terminology.json` | code → term + definition for every IMDRF code |
| `code_to_parent_codes.json` | code → ancestor paths (the label hierarchy) |

Each line of the split files is one report:

```json
{
  "idx": 20031503,
  "text": "Device Information: guide, surgical, instrument (2.0/2.7mm double-ended drill guide; smith & nephew, inc.) operated by health professional\nEvent Type: malfunction\nDescription of Event: it was reported that during a clavicle internal fixation, a drill seized in the 2.0/2.7mm double-ended drill guide ...",
  "labels": ["A04", "A0401", "A040104", "A05", "A0501", "A15", "A1503", "A150301", "E24", "E2403"]
}
```

Labels are *upward-propagated*: whenever a child code applies (`A040104`), its parent (`A0401`) and grandparent (`A04`) are included as well.

```python
import json

test = [json.loads(line) for line in open("test.jsonl")]
labels = json.load(open("labels.json"))          # 1,154 codes
terminology = json.load(open("imdrf_terminology.json"))
print(test[0]["text"][:200], test[0]["labels"])
```

## Evaluating your model

`evaluate.py` computes all metrics from Table 2 of the paper. It needs only `numpy` and `scipy`:

```bash
pip install numpy scipy
```

**1. Produce a predictions file** — JSONL, one object per test sample:

```json
{
  "idx": 20031503,
  "predictions": ["A04", "A0401", "A040104"],
  "scores": {"A04": 0.93, "A0401": 0.71, "A040104": 0.55},
  "uncertainties": {"A04": 0.11, "A0401": 0.48, "A040104": 0.74},
  "uncertainty": 0.44
}
```

- `idx`, `predictions` — required; enough for Macro-F1 and Jaccard.
- `scores` — optional per-label confidence in [0, 1]; enables **ECE+** (and is a fallback for the other UQ metrics). May cover any subset of the vocabulary — e.g. the full probability vector for discriminative models, or just the generated labels for generative models.
- `uncertainties` — optional per-label uncertainty on any scale (e.g. token entropy); enables **Spearman ρ**.
- `uncertainty` — optional per-sample uncertainty on any scale; enables **PRR**.

**2. Run the evaluation:**

```bash
python evaluate.py --predictions my_predictions.jsonl --data-dir /path/to/dataset
```

```
Evaluated samples : 10288

Predictive performance
  Macro F1  overall      : 0.5276
  Macro F1  head         : 0.7226
  Macro F1  medium       : 0.6200
  Macro F1  tail         : 0.5114
  Macro F1  extreme tail : 0.1582
  Jaccard (J)            : 0.6080

Uncertainty quantification
  PRR                    : 0.5519   (10189 samples with uncertainty)
  Spearman rho           : -0.2710   (87519 label predictions)
  ECE+                   : 0.4886   (10189 samples with scores, 1133 labels with positives)
```

Add `--output results.json` for machine-readable output. Test samples missing from the predictions file are scored as empty predictions by default (`--skip-missing` restricts evaluation to the provided samples — the counts are always reported).

If your model produces raw per-token log-probabilities in the paper's output format, `scripts/convert_paper_outputs.py` converts them into this predictions format using the paper's UQ aggregations (summed top-5 token entropy per label, per-token mean probability as confidence).

### Metrics

| Metric | What it measures |
|---|---|
| **Macro F1** ↑ | mean per-label F1, overall and per frequency bucket (head/medium/tail/extreme tail) |
| **J** (Jaccard) ↑ | mean per-sample overlap between predicted and gold label sets |
| **PRR** ↑ | how well per-sample uncertainty ranks bad predictions for rejection (1 = oracle, 0 = random) |
| **ρ** (Spearman) ↓ | correlation between per-label uncertainty and correctness — more negative is better |
| **ECE+** ↓ | positive-class calibration error: mean under-confidence on true-positive labels |

All implementations are faithful ports of the code used for the paper; the script reproduces the published Table 2 numbers from the paper's raw model outputs.

## Leaderboard (Table 2)

Macro-F1 per frequency regime, Jaccard, and UQ quality on the 10,288-sample test set. For generative models, PRR corresponds to the best-performing U_info metric. **Bold** = best in paradigm, <ins>underlined</ins> = best overall.

| Paradigm / Model | Overall | Head | Medium | Tail | ET | J ↑ | PRR ↑ | ρ ↓ | ECE+ ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **Discriminative fine-tuning** | | | | | | | | | |
| Llama-3.1-8B-Base | <ins>**0.54**</ins> | <ins>**0.74**</ins> | <ins>**0.64**</ins> | <ins>**0.53**</ins> | 0.12 | <ins>**0.62**</ins> | 0.47 | −0.40 | 0.58 |
| Llama-3.2-3B-Base | 0.51 | 0.72 | 0.62 | 0.49 | 0.11 | 0.59 | 0.46 | −0.41 | 0.59 |
| Llama-3.2-1B-Base | 0.51 | 0.71 | 0.60 | 0.48 | **0.14** | 0.58 | **0.52** | **−0.42** | 0.60 |
| Ettin-1B-Encoder | 0.53 | 0.73 | 0.63 | 0.51 | 0.13 | 0.61 | 0.46 | −0.40 | **0.56** |
| Ettin-400M-Encoder | 0.51 | 0.72 | 0.61 | 0.50 | 0.12 | 0.58 | 0.44 | −0.36 | 0.59 |
| Ettin-150M-Encoder | 0.46 | 0.68 | 0.56 | 0.44 | 0.07 | 0.55 | 0.38 | −0.30 | 0.64 |
| **Generative fine-tuning** | | | | | | | | | |
| Llama-3.1-70B-Base | **0.53** | **0.73** | **0.62** | **0.51** | **0.16** | **0.61** | 0.55 | −0.27 | **0.49** |
| Llama-3.1-8B-Base | 0.50 | 0.70 | 0.59 | 0.48 | 0.12 | 0.59 | <ins>**0.63**</ins> | −0.30 | 0.52 |
| Llama-3.2-3B-Base | 0.48 | 0.67 | 0.57 | 0.46 | 0.12 | 0.58 | 0.60 | <ins>**−0.46**</ins> | 0.57 |
| Llama-3.2-1B-Base | 0.43 | 0.63 | 0.52 | 0.39 | 0.10 | 0.45 | 0.54 | −0.44 | 0.60 |
| Ettin-1B-Decoder | 0.47 | 0.67 | 0.56 | 0.46 | 0.10 | 0.57 | 0.56 | −0.43 | 0.57 |
| Ettin-400M-Decoder | 0.44 | 0.66 | 0.54 | 0.42 | 0.07 | 0.54 | 0.57 | −0.44 | 0.60 |
| **Prompting — instruct** (10-shot kNN) | | | | | | | | | |
| Llama-3.1-70B-Instruct | 0.30 | 0.50 | 0.35 | 0.25 | 0.08 | 0.43 | **0.60** | −0.15 | 0.68 |
| Llama-3.1-8B-Instruct | 0.08 | 0.28 | 0.09 | 0.03 | 0.01 | 0.22 | 0.20 | 0.26 | 0.78 |
| Qwen3-235B-A22B-Instruct | **0.44** | **0.60** | **0.48** | **0.42** | **0.24** | 0.49 | 0.56 | **−0.34** | **0.56** |
| Qwen3-30B-A3B-Instruct | 0.22 | 0.48 | 0.27 | 0.14 | 0.05 | 0.43 | 0.54 | 0.05 | 0.59 |
| Qwen3-4B-Instruct | 0.29 | 0.49 | 0.35 | 0.25 | 0.09 | 0.41 | 0.49 | −0.27 | 0.68 |
| Kimi-K2-Instruct | 0.09 | 0.18 | 0.11 | 0.06 | 0.01 | 0.07 | 0.28 | 0.08 | 0.97 |
| GPT-4.1 | 0.43 | 0.59 | 0.47 | **0.42** | 0.22 | **0.57** | 0.45 | −0.31 | 0.60 |
| **Prompting — thinking** (10-shot kNN) | | | | | | | | | |
| Llama-3.3-Nemotron-49B-v1.5 | 0.42 | 0.57 | 0.46 | 0.38 | 0.19 | 0.46 | 0.21 | −0.03 | 0.59 |
| Qwen3-235B-A22B-Thinking | 0.49 | 0.62 | 0.52 | 0.48 | 0.33 | 0.48 | **0.34** | **−0.09** | <ins>**0.45**</ins> |
| Qwen3-30B-A3B-Thinking | 0.45 | 0.58 | 0.49 | 0.44 | 0.28 | 0.47 | 0.08 | −0.07 | 0.56 |
| Qwen3-4B-Thinking | 0.38 | 0.53 | 0.42 | 0.36 | 0.20 | 0.43 | 0.21 | −0.02 | 0.63 |
| DeepSeek-R1-0528 | 0.48 | 0.62 | 0.51 | 0.47 | 0.30 | 0.50 | 0.24 | **−0.09** | 0.50 |
| GLM-4.5-Air | 0.42 | 0.56 | 0.46 | 0.39 | 0.24 | 0.44 | 0.24 | **−0.09** | 0.62 |
| gpt-oss-120b | 0.40 | 0.57 | 0.45 | 0.38 | 0.15 | 0.45 | 0.05 | 0.00 | 0.63 |
| GPT-5 | <ins>**0.54**</ins> | **0.68** | **0.58** | <ins>**0.53**</ins> | <ins>**0.34**</ins> | **0.57** | — | — | — |

*GPT-5 does not expose log-probabilities, so its UQ metrics cannot be computed.*

Key takeaways: small discriminatively fine-tuned decoders offer the strongest head-to-tail accuracy with competitive UQ; generative fine-tuning gives the most reliable uncertainty; large reasoning models help most on rare labels but have surprisingly weak UQ; self-verbalized confidence is not a reliable uncertainty proxy.

Want your model on the leaderboard? Open an issue or PR with your predictions file and a short description of your setup.

## Prompts

The exact prompts used for all prompting experiments (system prompts for the 10/20/40-shot and self-verbalized-confidence runs, user prompt templates, and the generative fine-tuning format) are in [`prompts/`](prompts/), together with a description of how the kNN few-shot examples and the label taxonomy block are assembled.

## Repository layout

```
├── evaluate.py                        # official evaluation script (Table 2 metrics)
├── head_medium_tail.json              # frozen label frequency buckets used by evaluate.py
├── prompts/                           # exact prompts used in the paper
└── scripts/
    ├── convert_paper_outputs.py       # raw logprob outputs -> predictions format
    └── compute_label_buckets.py       # recompute frequency buckets from train.jsonl
```

## Citation

```bibtex
@inproceedings{agarwal-etal-2026-made,
    title = "{MADE}: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events",
    author = "Agarwal, Raunak  and
      Wenzel, Markus A.  and
      Baur, Simon  and
      Zimmer, Jonas  and
      Harvey, George  and
      Ma, Jackie",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.2148/",
    doi = "10.18653/v1/2026.acl-long.2148",
    pages = "46308--46328",
}
```

## License & data provenance

The code in this repository is released under the MIT License. The dataset is derived from [openFDA](https://open.fda.gov/) Medical Device Adverse Event reports (public domain, [CC0 1.0](https://open.fda.gov/license/)) and the [IMDRF](https://www.imdrf.org/) adverse-event terminology. Reports are annotated by FDA coders; the benchmark inherits the caveats of that source.

## Contact

Department of Artificial Intelligence, [Fraunhofer Heinrich Hertz Institute](https://www.hhi.fraunhofer.de/), Berlin.
Questions and leaderboard submissions: please open a GitHub issue.
