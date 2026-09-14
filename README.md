# RouteKV — Shared-Prefix KV Reuse Across Standard LoRA Adapters

[![paper](https://img.shields.io/badge/paper-PDF-b31b1b)](paper/paper.pdf)
![status](https://img.shields.io/badge/status-preliminary-yellow)
![python](https://img.shields.io/badge/python-3.11-blue)
![transformers](https://img.shields.io/badge/transformers-5.5-orange)
![peft](https://img.shields.io/badge/peft-0.20-orange)
![backbone](https://img.shields.io/badge/backbone-Qwen3--1.7B-brightgreen)

Code, data, and paper for *Shared-Prefix KV Reuse Across Standard LoRA Adapters: Quality and Serving
Tradeoffs* (AltSlate Labs, 2026).

## Figures

**The four takeover boundaries** — blue = reused base-computed KV, orange = specialist-computed, green =
generated answer. Only the boundary moves; the prompt and decoding are identical.

![Takeover boundaries](paper/figures/boundary_schematic.png)

**Non-monotonic boundary result** (GSM8K, math specialist, n=500, paired 95% CIs) — recomputing more of the
prefix is not uniformly better; full-prefix reuse returns closest to native.

![Boundary result](paper/figures/boundary_result.png)

**Specialist-dependence contrasts** — the individual math reuse penalty and QA seam exclude zero, but the
*differences between specialists* include zero: no evidence the penalty is specialist-specific.

![Specialist-dependence forest](paper/figures/forest.png)

## What this studies

A common small-model deployment runs **one shared backbone with several LoRA specialists** that answer over
the same context (retrieved passages, few-shot demonstrations). Serving them naïvely re-prefills that shared
context once per specialist. We ask: for **already-trained standard** LoRA adapters — not adapters retrained
for cache compatibility — how much task quality is preserved if the backbone's prefill KV cache is computed
once and reused across specialists, and what does it buy in serving cost?

Backbone: Qwen3-1.7B. Specialists: extractive QA (HotpotQA, supplied context) and arithmetic reasoning
(GSM8K). Metrics are deterministic (F1, exact-match) with paired bootstrap 95% CIs.

## Headline findings (see the paper for the qualifications)

- **Full-prefix reuse** — reuse the whole base-computed prefix; the specialist processes only the final
  prompt token and generates — had the lowest prefill cost and a small observed quality difference
  (GSM8K exact-match Δ = −1.0 points, 95% CI [−5.0, +3.0]). **Equivalence is not established.**
- **Partial recomputation** (specialist re-encodes only the question over a base-encoded prefix) gave **no
  demonstrated quality advantage** and the worst point estimate. No general boundary-selection rule is
  established.
- A closed-form **ridge KV translator did not beat direct reuse**.
- **Specialist-dependence** contrasts all include zero — no evidence the penalty is specific to the
  reasoning adapter.
- The serving benefit is **M shared-prefix prefills → one**, plus deduplicated prefix storage. Initial
  cache aliasing is verified; **peak memory during generation is not yet measured** — the key open serving
  experiment.

This is a preliminary empirical report. Known gaps: single adapter seed per task, one 1.7B backbone, quality
measured at ≈655 tokens while storage is shown at 8K, and an overlapping (not held-out) follow-up sample.
See the paper's Limitations.

## Layout

```
paper/    paper.typ (Typst source), refs.bib, paper.pdf, figures/, make_figures.py
code/     training + evaluation harnesses
results/  per-example 0/1 outcome arrays (npz) for the n=500 boundary / DiD study
```

## Reproduce

```bash
pip install "torch>=2.11" "transformers==5.5" "peft==0.20" datasets numpy
# train the two specialists
python code/train_lora.py            # QA (HotpotQA) and math (GSM8K) LoRA adapters, r16 a32, q/k/v/o
# quality: boundary sweep + specialist-dependence DiD (two shards over GSM8K test[80:580])
CUDA_VISIBLE_DEVICES=0 python code/confirm_did.py 0 250
CUDA_VISIBLE_DEVICES=1 python code/confirm_did.py 250 250
python code/merge_did.py             # boundary table + direct seam contrast + DiD, with paired CIs
# extractive QA reuse (supplied distractor context, truncated to 700 tokens)
python code/qa_eval_large.py
# serving cost (prefill/cache accounting)
python code/memory_accounting.py ; python code/memory_audit.py
```

`results/did_shard_*.npz` hold the per-example arrays behind the paper's boundary and DiD numbers; load with
`numpy.load` (keys: `math_native, math_early, math_question, math_full, qa_question, qa_full, golds`).

Paper: `paper/paper.pdf` (compile with `typst compile paper/paper.typ`).
