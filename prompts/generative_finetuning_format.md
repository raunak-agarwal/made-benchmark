# Generative fine-tuning format

The generative fine-tuning paradigm ("Generative fine-tuning" rows in Table 2) trains
decoder models to emit the label codes as text. There is no system prompt, no label list,
and no few-shot examples — just a bare completion format.

## Training example

```
INPUT
{report text}

OUTPUT
{labels, sorted alphabetically, joined with "; "}
```

For example:

```
INPUT
Device Information: guide, surgical, instrument ... operated by health professional
Event Type: malfunction
Description of Event: it was reported that during a clavicle internal fixation ...

OUTPUT
A04; A0401; A040104; A05; A0501; A15; A1503; A150301; E24; E2403
```

## Inference

The prompt is `"INPUT\n" + text[:5000] + "\n\nOUTPUT\n"`, decoded greedily
(`temperature=0`, max 100 new tokens) with top-5 token log-probabilities recorded.
Generated text is split on `;`, `,`, and newlines to recover the predicted label list.
