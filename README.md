# RouteKV — Shared-Prefix KV Reuse Across Standard LoRA Adapters

[![paper](https://img.shields.io/badge/paper-PDF-b31b1b)](paper/paper.pdf)
![status](https://img.shields.io/badge/status-preliminary-yellow)
![python](https://img.shields.io/badge/python-3.11-blue)
![transformers](https://img.shields.io/badge/transformers-5.5-orange)
![peft](https://img.shields.io/badge/peft-0.20-orange)
![backbone](https://img.shields.io/badge/backbone-Qwen3--1.7B-brightgreen)

A composable small-model deployment runs **one shared backbone with several LoRA specialists** that answer
over the same context — the same retrieved passages, the same few-shot demonstrations. Served naïvely, that
shared context is re-prefilled once per specialist. This repository studies a narrow, practical question and
measures it carefully:

> **How much task quality does inference-time base-KV cache reuse preserve for _already-trained standard_
> LoRA adapters — without retraining for cache compatibility — and at what serving cost?**

Backbone: Qwen3-1.7B. Specialists: extractive QA (HotpotQA) and arithmetic reasoning (GSM8K). All metrics
are deterministic (F1, exact-match) with paired bootstrap 95% confidence intervals.

## How reuse works

We compute the shared prefix's KV cache once with the adapters disabled, then let a specialist take over at
a chosen boundary and generate. Sweeping that boundary — holding the full prompt and decoding fixed —
separates *how many tokens are reused* from *which content the specialist re-encodes*.

![Takeover boundaries: blue is reused base-computed KV, orange is specialist-computed, green is the generated answer](paper/figures/boundary_schematic.png)

## What we find

**Full-prefix reuse is the cheapest condition and the closest to native**, with a small observed quality
difference — GSM8K exact-match Δ = −1.0 points (95% CI [−5.0, +3.0]). Equivalence is *not* established: the
interval still permits a real loss. **Partial recomputation** (the specialist re-encodes only the question
over a base-encoded prefix) gives *no* demonstrated advantage and the worst point estimate. Recomputing more
of the prefix is not uniformly better.

![Boundary sweep on GSM8K (n=500, paired 95% CIs): the relationship is non-monotonic and full-prefix reuse returns closest to native](paper/figures/boundary_result.png)

**No evidence the penalty is specialist-specific.** The individual math reuse penalty and the QA seam each
exclude zero, but the *differences between specialists* — the penalty difference and the seam
difference-in-differences — both include zero, and the latter's point estimate reverses.

![Specialist-dependence contrasts: individual effects exclude zero, but the between-specialist differences include it](paper/figures/forest.png)

**A ridge KV translator did not beat direct reuse**, and its cost was not justified. On the systems side,
reuse turns *M* shared-prefix prefills into one and deduplicates prefix storage; initial cache aliasing is
verified, but **peak memory during generation is not yet measured** — the central open serving question.

This is a preliminary empirical report. Known limits: a single adapter seed per task, one 1.7B backbone,
quality measured at ≈655-token prompts while storage is shown at 8K, and an overlapping (not held-out)
follow-up sample. See the paper's Limitations.

## Repository

```
paper/     paper.typ (Typst source), refs.bib, paper.pdf, figures/, make_figures.py
code/      training and evaluation harnesses
results/   per-example outcome arrays (npz) behind the n=500 boundary / specialist-dependence study
```

## Reproduce

```bash
pip install "torch>=2.11" "transformers==5.5" "peft==0.20" datasets numpy

# 1. train the two specialists (QA on HotpotQA, math on GSM8K; r16 a32, q/k/v/o)
python code/train_lora.py

# 2. quality: boundary sweep + specialist-dependence DiD over GSM8K test[80:580]
CUDA_VISIBLE_DEVICES=0 python code/confirm_did.py 0 250
CUDA_VISIBLE_DEVICES=1 python code/confirm_did.py 250 250
python code/merge_did.py

# 3. extractive QA reuse (supplied distractor context, truncated to 700 tokens)
python code/qa_eval_large.py

# 4. serving-cost accounting (prefill / cache)
python code/memory_accounting.py && python code/memory_audit.py
```

`results/did_shard_*.npz` hold the per-example arrays behind the paper's boundary and DiD numbers
(`numpy.load`; keys `math_native, math_early, math_question, math_full, qa_question, qa_full, golds`).

## Paper

`paper/paper.pdf` — *Shared-Prefix KV Reuse Across Standard LoRA Adapters: Quality and Serving Tradeoffs*
(AltSlate Labs, 2026). Compile with `typst compile paper/paper.typ`.
