# Results metadata

Frozen configuration, sample identities, and metric definitions for the arrays in this directory.

## Environment
- Backbone `Qwen/Qwen3-1.7B` (bf16), HF revision `main`.
- transformers 5.5, peft 0.20, torch 2.11; one NVIDIA RTX PRO 4500 Blackwell GPU.
- Specialists: LoRA rank 16, α=32, dropout 0, targets `q_proj,k_proj,v_proj,o_proj`; QA on HotpotQA
  (distractor), math on GSM8K (main). Greedy decoding.

## Sample identities (disjoint by role)
| role | task | indices |
|---|---|---|
| dev | GSM8K test | 0–80 |
| boundary / DiD (overlapping follow-up) | GSM8K test | 80–580 |
| matched + budget diagnostic (held-out) | GSM8K test | 580–1080 |
| QA earlier eval | HotpotQA validation | ~60–560 (skip 60, cal 40 + eval 500) |
| QA matched | HotpotQA validation | ≥600 |
| QA distractor pool | HotpotQA validation | 0–60 |

## Generation budgets (frozen)
- Main runs: 160 tokens (GSM8K), 48 tokens (QA).
- Budget diagnostic: 320 tokens (GSM8K), native vs reuse only.

## Metric definitions
- **GSM8K EM**: strip thousands-separators (commas), take the last signed-integer regex match; decimals and
  fractions are not parsed; a completion with no integer scores 0.
- **QA F1**: token-level F1 on the first output line, lowercased, articles/punctuation stripped.
- **Paired bootstrap CI**: 10,000 resamples of per-example (condition − native) differences.

## Files
- `did_shard_{0,250}_250.npz` — boundary + specialist-dependence study (GSM8K test[80:580]).
  Keys: `math_native, math_early, math_question, math_full, qa_question, qa_full, golds`.
- `matched_gsm8k_0.npz`, `matched_qa_2000.npz`, `matched_qa_8000.npz` — matched quality/latency/memory.
  Keys: `{Q,TT,LAT,PK,PKO,CAP}_{base,native,reuse}`, `prefill_once`
  (Q=score, TT=TTFT ms, LAT=completion ms, PK=peak GB, PKO=peak over weights GB, CAP=cap-hit 0/1).
- `budget_320.npz` — generation-budget diagnostic (GSM8K test[580:1080], budget 320).
  Keys: `nat, reu, capn, capr`.
- `matched_gsm8k_0_math_s2b.npz` — second-seed replication (adapter `math_s2b`, `--seed 2`, init only,
  original data order; GSM8K test[580:1080]). Same key schema as the other `matched_*` files.

Code revision: see the repository's git history; the harnesses that produced these arrays are in `code/`
(`confirm_did.py`, `merge_did.py`, `matched.py`, `residency.py`, `budget_diag.py`).
