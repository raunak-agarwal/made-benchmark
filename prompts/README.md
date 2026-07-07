# Prompts

The exact prompts used for the prompting experiments in the MADE paper (Table 2, "Prompting – instruct" and "Prompting – thinking" paradigms).

## Files

| File | Purpose |
|---|---|
| `system_prompt_k10.txt` | System prompt for the main 10-shot runs (all Table 2 prompting rows) |
| `system_prompt_k20.txt` / `system_prompt_k40.txt` | 20-/40-shot ablation variants (identical except the stated number of examples) |
| `system_prompt_verbalized_k10.txt` | Self-verbalized confidence variant (model outputs a JSON dict of `label: confidence`) |
| `user_prompt_template.txt` | User message template for the main runs |
| `user_prompt_template_verbalized.txt` | User message template for the self-verbalized confidence runs |
| `generative_finetuning_format.md` | Input/output format used for generative fine-tuning |

## How the user prompt is assembled

The user prompt template has three placeholders:

- **`{LABELS}`** — the full label taxonomy (all 1,154 codes), one line per label, formatted as
  `{code}: {term} - {definition}` using `labels.json` and `imdrf_terminology.json` from the
  [dataset release](https://huggingface.co/datasets/ragarwal/MADE-Multilabel-Benchmark).
  The candidate label set is *never* retrieved or filtered — every prompt contains the complete taxonomy.

- **`{EXAMPLES}`** — `k` few-shot examples (k = 10 for the main runs) retrieved with kNN:
  cosine similarity between the test report embedding and all training-set embeddings
  (embeddings from `bioclinical-modernbert-base-embeddings`). Each example is formatted as

  ```
  EXAMPLE {i}:
  REPORT:
  {text}

  LABELS:
  {label_1}
  {label_2}
  ...
  ```

  Example texts are truncated to 10,000 characters. Retrieval is over the *training set only*.

- **`{CLASSIFICATION_TEXT}`** — the test report to classify, truncated to 10,000 characters.

## Decoding settings

Greedy decoding (`temperature=0`, `top_p=1`, `top_k=-1`), with top-5 token log-probabilities
recorded for uncertainty quantification. The consistency-based UQ ablations (`n=5`) use
5 stochastic samples at `temperature=1`.

For thinking/reasoning models, only the content after the final thinking terminator
(e.g. `</think>`) is parsed. The output is split on newlines; each line is treated as one
predicted label.
