// RouteKV — shared-prefix KV reuse across standard LoRA adapters. arkheion template.
// Compile: typst compile paper.typ
#import "@preview/arkheion:0.1.0": arkheion

#show: arkheion.with(
  title: "Shared-Prefix KV Reuse Across Standard LoRA Adapters: Quality and Serving Tradeoffs",
  authors: (
    (name: "Dushyant Rajput", email: "dushyant@altslate.com", affiliation: "AltSlate Labs LLP"),
  ),
  abstract: [
    A common small-model deployment runs one shared backbone with several LoRA @lora specialists that
    answer over the same context. Serving them naïvely re-prefills that shared context once per specialist.
    We study a narrow, practical question: for #strong[already-trained standard] LoRA adapters — not adapters
    retrained for cache compatibility — how much task quality is preserved if the backbone's prefill KV cache
    is computed once and reused across specialists, and what does that buy in serving cost? On a Qwen3-1.7B
    @qwen3 backbone with two adapters (extractive QA on HotpotQA @hotpotqa, arithmetic reasoning on GSM8K
    @gsm8k), we sweep the boundary at which the specialist takes over from the reused base cache and measure
    paired quality differences and serving cost. #strong[Full-prefix reuse had the lowest prefill cost and a
    small observed quality difference (GSM8K exact-match $Delta = -1.0$ points, 95% CI $[-5.0, +3.0]$) in
    our tested setup. Partial recomputation provided no demonstrated quality advantage. Neither quality
    equivalence nor a general boundary-selection rule is established.] We also report a closed-form ridge KV
    translator that did not beat direct reuse, and specialist-dependence contrasts whose intervals all
    include zero. The serving benefit is a reduction from $M$ shared-prefix prefills to one (each specialist
    still runs a suffix forward), plus deduplicated prefix storage; the wall-time and peak-memory
    implications depend on the workload and are only partly measured here.
  ],
  keywords: ("KV cache", "LoRA", "efficient serving", "small language models", "prefix reuse"),
  date: "September 2026",
)

= Introduction

Small models are increasingly deployed as a composable system: one backbone plus several lightweight LoRA
@lora specialists, routed per request. These specialists often answer over a #emph[shared] context — the
same retrieved passages, the same conversation history, the same few-shot demonstrations — and differ only
in the adapter applied. Serving such a system the obvious way re-runs the prefill over that shared context
once per specialist that touches it.

Prefix caching in serving stacks (PagedAttention @pagedattention, RadixAttention @sglang) already avoids
recomputation when the #emph[same] model sees an #emph[identical] prefix. The question here is different:
several #emph[different] models (backbone $+ A_i$) see the #emph[same] prefix. For a family of
#emph[different-size] models this needs a learned map between key/value spaces — NVIDIA's closed-form linear
KV transfer @crossmodelkv and the concurrent CacheBridge @cachebridge. Activated LoRA @alora and its serving
engine @aloraserving instead modify the adapter so a base prefix cache is exactly reusable by construction.

We study the case those methods do not target: #strong[standard] LoRA adapters, already trained for their
task with no cache-compatibility objective, on a #strong[shared backbone]. The precise question is:

#block(inset: (left: 8pt), stroke: (left: 1.5pt + gray), [
  #emph[How much task quality does inference-time base-cache reuse preserve for already-trained standard
  LoRA adapters, without retraining for cache compatibility, and at what serving cost?]
])

Our answer is qualified. We report the exact reuse procedure, a quality study with paired confidence
intervals, a reconciled serving-cost accounting, one rejected translator, and specialist-dependence
contrasts that do not resolve. We are explicit about which measurements are exploratory and which we would
trust, and about the gap between the context lengths used for quality and for serving cost.

= Setting and method

*Composable serving.* One backbone $B$ (Qwen3-1.7B @qwen3) hosts LoRA specialists, each rank-16
($alpha=32$) on the attention projections (q/k/v/o). We train two: QA on HotpotQA @hotpotqa and math on
GSM8K @gsm8k (recipe in @app-repro). At request time a router selects a specialist; several specialists may
answer over the same shared context. Because every specialist shares the backbone weights (loaded once) and
the same context, the shared prefix is a candidate for compute-once reuse. Throughout, $M$ denotes the
number of specialists sharing a context; the two trained adapters are used for quality, and $M$ enters only
the (adapter-agnostic) serving-cost accounting of §4.

*Reuse procedure.* The backbone computes the prefill KV cache for the shared prefix #emph[with adapters
disabled] (base representation). A chosen specialist then processes the #emph[non-reused suffix] of the
prompt on top of that cache and generates. No key/value mapping is applied. Crucially, the first answer
token's logits always come from the #emph[specialist], which processes at least one prompt token
(@fig-schematic, @lst-reuse). For a boundary $b$ (number of reused prefix tokens) and prompt of length $T$:

#figure(
  block(inset: (left: 6pt), text(8.5pt, font: "DejaVu Sans Mono")[
    `prefix_kv = forward(B, tokens[0:b], adapters_disabled).kv   # b reused base-computed tokens`\
    `set_adapter(specialist)                                      # base -> specialist`\
    `out = forward(specialist, tokens[b:T], past=prefix_kv)       # suffix; positions continue at b..T-1`\
    `logits_1 = out.logits[-1]                                    # FIRST answer token: specialist's logits`\
    `# then greedy-decode with the specialist and the growing cache`
  ]),
  caption: [
    Reuse procedure. `native`: $b=0$ (specialist computes the whole prompt). `early`: $b=|"instr"|$.
    `question`: $b=|"instr"|+|"demos"|$. `full-prefix`: $b=T-1$, so the specialist processes exactly the
    final prompt token before generating — not "generation only." Position IDs continue from $b$; the reused
    cache is base-computed, the suffix and all generation are specialist-computed.
  ],
) <lst-reuse>

*Takeover boundary.* We formalize the prompt as three segments —
`[instructions | demonstrations | question] -> answer` — and sweep $b$ across the four settings above
(@fig-schematic), holding the full prompt and greedy decoding fixed. This separates "how many tokens are
reused" from "which content the specialist re-encodes."

#figure(image("figures/boundary_schematic.svg", width: 96%), caption: [
  The four takeover boundaries. Blue tokens reuse the base-computed prefill KV; orange tokens are processed
  by the specialist; green is the generated answer. In `full-prefix` the specialist still processes the
  final prompt token (a one-token orange sliver, not drawn to scale). The full prompt and decoding are
  identical across rows; only the boundary $b$ moves.
]) <fig-schematic>

*Evaluation.* Metrics are deterministic: token-level F1 for extractive QA, exact-match (final numeric
answer) for GSM8K; no model-based judging. We report paired bootstrap @bootstrap 95% confidence intervals
over per-example score differences (10,000 resamples), the appropriate interval for the same-example,
cached-vs-native comparisons here. Runs use transformers 5.5 / peft 0.20 / torch 2.11, bf16, greedy
decoding, generation capped at 160 tokens, on NVIDIA Blackwell GPUs. Exact revisions, prompts, sample
ranges, and timing methodology are in @app-repro.

= Serving cost

Reuse changes three distinct quantities, which the draft keeps separate to avoid overstating the benefit.

*Prefill count (structural, exact).* When $M$ specialists answer over one shared context, the shared prefix
is prefilled once instead of $M$ times. This ratio is exactly $M$ by construction, independent of the
adapters.

*Prefix storage (aliasing verified initially; peak memory not measured).* The shared prefix's KV can be a
single allocation rather than $M$ copies. #strong[The implementation demonstrates initial prefix-cache
aliasing. Persistent sharing and its effect on peak memory during generation remain unmeasured] (we did not
instrument peak allocation while branches generate, where concatenation or copy-on-write could break
sharing). Whether this becomes a peak-memory saving depends on the workload: it can help
#emph[simultaneously retained] branches (several specialists answering one context at once) and does nothing
for #emph[sequential] requests that free each cache before the next. Measuring peak memory during actual
branching is the key open serving measurement (@sec-limits).

*Measured prefill time (benchmark-dependent, $approx M times$, not exact).* Reuse does not make per-specialist
work vanish: each specialist still runs a #emph[suffix forward] over its non-reused tokens plus generation
(@lst-reuse). The saving is on the #emph[shared-prefix] portion — one base prefill replaces $M$. For $M$
requests over one shared prefix the shared-prefix wall-time ratio is
$ (M dot t_"prefill,specialist") / (t_"prefill,base" + t_"handoff"), $
which approaches $M times$ only when handoff is small. @tab-cost gives single-prefix cost; note the native
per-specialist prefill (489 ms at 8K) exceeds the base prefill (413 ms) that reuse pays once. @tab-scale's
8K microbenchmark instead repeated the #emph[base] prefill (408 ms), so its "$4 times 408$" is a
repeated-base-prefill figure, not four adapter-enabled prefills; 408 vs 413 ms are separate runs agreeing to
≈1%.

#figure(
  table(
    columns: (auto, auto, auto, auto), inset: 5pt, align: (right, right, right, right), stroke: 0.4pt,
    [*context (tok)*], [*prefix KV cache*], [*base prefill*], [*specialist prefill*],
    [700], [80 MB], [38 ms], [42 ms],
    [2048], [235 MB], [88 ms], [100 ms],
    [8192], [940 MB], [413 ms], [489 ms],
  ),
  caption: [
    Single-prefix cost (Qwen3-1.7B, bf16; one benchmark run). Cache is what reuse deduplicates; prefill is
    what reuse avoids repaying per specialist. Specialist prefill exceeds base by the adapter's
    matrix-multiply overhead.
  ],
) <tab-cost>

#figure(
  table(
    columns: (auto, auto, auto), inset: 5pt, align: (right, right, right), stroke: 0.4pt,
    [*specialists $M$*], [*prefill count*], [*retained prefix caches*],
    [2], [2 → 1], [2 → 1],
    [4], [4 → 1], [4 → 1],
  ),
  caption: [
    Structural ratios (exact, adapter-agnostic). A separate 8K microbenchmark repeated the #emph[base]
    prefill $M=4$ times: $1632 → 408$ ms ($4 times 408$ ms → one) and retained prefix-cache tensor
    footprint $3.76 → 0.94$ GB. This is a repeated-base-prefill figure, not four adapter-enabled prefills,
    and the storage figure is tensor footprint, not measured peak memory during generation (see text).
  ],
) <tab-scale>

*Scope.* Backbone weights (3.5 GB) are shared once regardless of reuse; per-specialist answer-side caches
and working buffers are unchanged. The benefit therefore scales with how much of the workload is shared
context relative to per-specialist generation. The quality study below uses ≈655-token prompts, whereas the
striking storage figures use 8K contexts; these are #emph[separate] measurements and we do not claim the
small quality difference transfers to 8K (@sec-limits).

= Quality of reused-KV inference

*Own-KV control (harness check).* Reusing each specialist's #emph[own] recomputed prefix KV should be
near-identical to native. It was (math EM 49.2 vs. 48.3, $n=120$): 3/120 examples disagreed (two
cached-correct, one native-correct). We did #emph[not] instrument the numerical divergence (e.g. per-layer
KV or first-token logit deltas), so we report the discrepancy as consistent with greedy sensitivity to
cache-reconstruction differences but #emph[not further diagnosed]. It is small relative to the cross-source
effects below.

*Extractive QA with supplied context (small observed difference).* This result is #emph[not] the full
distractor-retrieval task: for each HotpotQA @hotpotqa example we concatenate the distractor context (all 10
paragraphs, 2 gold + 8 distractor) and truncate to 700 tokens, then reuse the base prefix cache of that
context for the QA specialist. Truncation can drop answer-bearing text (not audited), so scores are a lower
bound on the oracle-context setting. On this setting (F1, $n=500$), reuse matched native to within the
interval: $Delta = +0.3$ F1 points, CI $[-1.5, +2.2]$ (@tab-qa). The reasoning task is the harder case.

#figure(
  table(
    columns: (auto, auto, auto), inset: 5pt, align: (left, right, right), stroke: 0.4pt,
    [*QA specialist (HotpotQA, supplied context, $n=500$)*], [*F1*], [*Δ vs native*],
    [Native (specialist prefills context)], [53.0], [—],
    [Full-prefix base-KV reuse], [53.3], [$+0.3$ $[-1.5, +2.2]$],
    [aLoRA adapter, native (preliminary)], [42.6], [—],
  ),
  caption: [
    QA F1 on the supplied-context setting. Native and reuse are the standard-LoRA specialist. The aLoRA row
    is a #emph[preliminary] run with matched rank/targets; activation correctness was not verified (§5), and
    it is not established to be on the identical sample as the standard-LoRA rows. The 53.0 here is QA F1 and
    is unrelated to the numerically-coincident GSM8K native EM of 53.0.
  ],
) <tab-qa>

*Takeover boundary (GSM8K, math specialist, $n=500$).* Moving only $b$ (@tab-boundary, @fig-boundary), the
relationship is non-monotonic: recomputing more of the prefix is not uniformly better.

#figure(
  table(
    columns: (auto, auto, auto, auto), inset: 5pt, align: (left, right, right, right), stroke: 0.4pt,
    [*boundary*], [*specialist processes*], [*EM*], [*Δ vs native (paired)*],
    [Native reference], [entire prompt], [53.0], [—],
    [Early takeover], [demonstrations + question], [55.2], [$+2.2$ $[-0.2, +4.8]$],
    [Question takeover], [question only], [49.2], [$-3.8$ $[-7.6, +0.0]$],
    [Full-prefix reuse], [final prompt token only], [52.0], [$-1.0$ $[-5.0, +3.0]$],
  ),
  caption: [
    Boundary sweep, math specialist, GSM8K, $n=500$ (test[80:580]), paired bootstrap CIs. Full-prefix reuse
    is closest to native and cheapest; the mid-context "question takeover" has the worst point estimate but
    its CI reaches zero.
  ],
) <tab-boundary>

#figure(image("figures/boundary_result.svg", width: 70%), caption: [
  $Delta$ EM vs native as a function of reused-prefix size (paired bootstrap 95% CIs). Non-monotonic:
  reusing more is not uniformly worse. Full-prefix reuse (rightmost) returns closest to native.
]) <fig-boundary>

Two comparisons matter. #strong[Full-prefix reuse] is closest to native ($Delta = -1.0$pp, CI
$[-5.0, +3.0]$) and processes the fewest specialist tokens. #strong[Question takeover] — the intuitive
"share the background, let the specialist encode the question" policy — has the worst point estimate
($Delta = -3.8$pp, CI $[-7.6, +0.0]$); its direct paired contrast against full-prefix reuse is $-2.8$pp
(CI $[-6.6, +1.0]$), which includes zero.

*Interpretation.* In our tested setup full-prefix reuse is cheaper than every other condition and has a
better observed point estimate than question takeover (early takeover has the highest point estimate, but
its interval also includes zero). We do #emph[not] establish quality equivalence (the interval permits a
loss up to ≈5pp) nor a general boundary-selection rule: §4 does not support "always recompute more" or
"never split representations." The
consistent direction (the mid-context boundary is the worst cell in every run we did) is a #emph[tendency]
we cannot yet attribute to a mechanism.

= Comparisons and negative results

*Standard LoRA vs Activated LoRA.* Because aLoRA @alora is designed for exact base-cache reuse (adapter
weights activate only after an invocation sequence) and its serving engine @aloraserving implements this,
it is the natural comparison. We trained an aLoRA adapter on the same QA data (invocation "`Answer:`",
matched rank/target modules; @app-repro). This is a #strong[preliminary aLoRA run; activation correctness
unverified]. Its native F1 was 42.6 versus the standard-LoRA specialist's 53.0, but because we did not
validate the activation gating (below), we do #emph[not] attribute this gap to training quality, capacity,
or architecture — it is not yet interpretable. Our structural cached-vs-uncached parity check for the aLoRA
adapter was #strong[inconclusive]: a plain forward pass does
not exercise the generation-time activation gating that makes aLoRA's reuse exact, so we could not confirm
end-to-end exactness in our harness. A proper comparison (matched standard-LoRA and aLoRA quality, their
training configs, and aLoRA cached-vs-uncached parity under generation) is left as needed work.

*A ridge KV translator does not earn its cost.* Motivated by cross-model transfer @crossmodelkv
@cachebridge, we fit closed-form per-head ridge maps (RoPE-stripped keys, single-layer $l→l$ and top-$k$
multi-layer) to translate base KV into the specialist's space. On our same-backbone setting it did not beat
direct reuse on task quality while adding per-layer matrix-multiply cost per reused token. (For the harder
#emph[cross-size] base→base case it improved fidelity — multi-layer maps reached 78.5% next-token top-1
agreement — but task retention was 33–56% and the accurate map was not economical.) The tested ridge map is
rejected; translation as a class is not.

*Specialist dependence is not established.* Is the reasoning specialist #emph[specifically] fragile under
reuse? Two contrasts test this and both include zero (@app-forest). The base-reuse-vs-native penalty
difference between adapters was $-9.2$pp (CI $[-21.7, +2.5]$, $n=120$). The takeover-seam
difference-in-differences, $("question"-"full-prefix")_"math" - ("question"-"full-prefix")_"QA"$, was
$+2.6$pp (CI $[-3.6, +8.6]$, $n=500$) — and the point estimate #emph[reverses]: the math seam
($-2.8$pp, CI $[-6.6, +1.0]$) was smaller than the QA seam ($-5.4$pp, CI $[-10.6, -0.2]$). Their individual
intervals exclude zero for the math reuse penalty and the QA seam, but the #emph[differences] between
specialists do not. Notably, for the weak-on-task QA specialist, letting the base encode the question was
associated with higher EM (full-prefix 44.2 vs. question-takeover 38.8). The present comparisons do not
establish specialist dependence; we do not inflate the sample to seek significance.

= Limitations and future work <sec-limits>

The central positive claim is that full-prefix reuse is close to native in our setup, and its interval does
#emph[not] establish equivalence — a real loss up to ≈5pp is not excluded. Several gaps bound the claims:

- #strong[Quality and serving cost are measured at different scales.] Quality uses ≈655-token prompts;
  the storage figures use 8K. The single most useful next experiment is one #strong[matched workload]: on a
  shared-context task, measure task quality, total peak allocated memory, and completion latency #emph[together]
  across context lengths, including #strong[base-only] inference (if the reused specialist adds little over
  the base, the adapter's necessity is in question).
- #strong[Two "shared context" workloads differ.] Full-prefix reuse as measured shares an #emph[identical
  full prompt] across specialists. Sharing #emph[background passages] across #emph[different] questions is a
  different setting our full-prefix result does not establish; it corresponds to a mid-prompt boundary, which
  is exactly where we see the (uncertain) penalty.
- #strong[Provenance.] The $n=500$ boundary evaluation (test[80:580]) is a #emph[larger follow-up
  evaluation], not an independent held-out confirmation: it overlaps the boundary-development slice
  (test[80:200]) entirely and the earliest development run used test[0:500]. A clean result needs a frozen
  harness and primary comparison, then a fresh evaluation whose size is set for a prespecified precision or
  noninferiority margin.
- #strong[Generality.] One backbone, one adapter seed per task, small (1.7B) scale, two tasks. A
  #strong[frozen replication] on a second adapter seed and a different backbone would test whether the small
  full-prefix gap and the null specialist-dependence result are stable.

The boundary sweep is parked; the exploratory measurement history is in @app-history.

= Related work

*Cross-model KV transfer.* Heo et al. @crossmodelkv fit a closed-form linear mapping to transfer KV between
#emph[different-size] models in a family, reporting 2.7–25× mapper speedups and 73–98% accuracy retention on
four of six pairs. CacheBridge @cachebridge is a concurrent method for cross-model KV transfer with its own
comparisons and results (we do not restate its numbers). Both target the cross-size case where key/value
spaces differ and a map is required. Our setting is the complementary one — same backbone, different LoRA
head — where, for full-prefix reuse, no map is applied; our ridge-map result is a negative for this regime.

*Adapter-aware caching.* Activated LoRA @alora modifies the adapter so base-prefix KV is exactly reusable,
and a serving engine @aloraserving implements multi-adapter serving on this reuse in vLLM. We instead ask
what already-trained #emph[standard] LoRA adapters preserve under direct reuse, and quantify it with paired
intervals; §5 gives our (partial) aLoRA comparison.

*Prefix caching.* PagedAttention @pagedattention and RadixAttention @sglang reuse KV across requests sharing
an #emph[identical] prefix under the #emph[same] model. Our question is reuse of one prefill across
#emph[different] specialists of a shared backbone.

= Conclusion

For composable serving on a shared backbone, reusing the backbone's prefill KV cache across
already-trained standard LoRA specialists reduces $M$ prefills and $M$ retained prefix caches to one; the
wall-time ($approx M times$ for sequential identical prefills) and peak-memory (for simultaneously-retained
branches) implications depend on the workload. On quality: full-prefix reuse had the lowest prefill cost and
a small observed quality difference in our tested setup (extractive QA within noise; GSM8K
$Delta = -1.0$pp), partial recomputation provided no demonstrated advantage, and neither quality equivalence
nor a general boundary-selection rule is established. A ridge translator did not earn its cost, and
specialist-dependence contrasts did not resolve. The useful next steps are a matched quality–memory–latency
workload at a fixed scale and a frozen replication; the claims above should be confirmed on one's own
adapters before reuse is assumed lossless.

#pagebreak()
= Specialist-dependence contrasts (appendix) <app-forest>

#figure(image("figures/forest.svg", width: 86%), caption: [
  Contrasts bearing on specialist-specific reuse penalty. Red intervals exclude zero; grey include it. The
  individual math reuse penalty and QA seam exclude zero, but the #emph[differences] between specialists —
  the penalty difference ($n=120$) and the seam difference-in-differences ($n=500$) — include zero, and the
  DiD point estimate is positive. Note the differing $n$; these come from separate runs.
]) <fig-forest>

= Exploratory measurement history (appendix) <app-history>

The "question takeover" penalty changed across successive #emph[development] measurements (@tab-drift).
These are not independent replications: sample draw and prompt construction changed together. One difference
we can rule out is `add_special_tokens` (a no-op here — the Qwen3 tokenizer emits no BOS token, verified);
the rest (a shared instruction preamble; different, overlapping GSM8K slices) are confounded. We report the
final row and treat the earlier ones as development history, not evidence of a fixed bug.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto), inset: 5pt, align: (left, right, left, left, right), stroke: 0.4pt,
    [*run (role)*], [*n*], [*prefix*], [*test slice*], [*Δ (question)*],
    [initial (dev)], [500], [demos], [test[0:500]], [$-14.4$],
    [matrix (dev)], [120], [demos], [test[0:120]], [$-9.2$],
    [boundary (dev)], [120], [instr+demos], [test[80:200]], [$-4.2$],
    [confirm (follow-up)], [500], [instr+demos], [test[80:580]], [$-3.8$],
  ),
  caption: [
    Development history of the "question takeover" penalty. The final follow-up overlaps the boundary
    development slice (test[80:200] $subset$ test[80:580]); it is not an independent held-out sample.
  ],
) <tab-drift>

= Reproducibility (appendix) <app-repro>

Code, adapter checkpoints, exact configs, prompts, and per-example score arrays are at
`github.com/AltSlate-Labs/routekv`. Remaining unpinned items (below) are marked; this appendix is an
outline, and the fully-pinned artifact accompanies the repository.

#strong[Models.] Backbone `Qwen/Qwen3-1.7B` (bf16; HF revision `main`, exact commit pinned in the repo).
Specialists are LoRA @lora adapters, rank 16, $alpha=32$, dropout 0, target modules
`q_proj,k_proj,v_proj,o_proj`; QA trained on HotpotQA @hotpotqa (distractor split), math on GSM8K @gsm8k
(main), plus one aLoRA @alora QA adapter (invocation "`Answer:`", matched rank/targets). Per-adapter SFT
recipe, seeds, and checkpoint hashes are in the repo (training seeds were not varied — a single seed per
task, which is why replication across seeds is future work).

#strong[Evaluation.] GSM8K: 4 fixed few-shot demonstrations from the train split; prompt
`[instruction | demos | "Question: {q}\nAnswer:"]`; greedy, generation capped at 160 tokens. Answer
extraction removes thousands-separators (commas), then takes the last signed-integer match; #emph[decimals
and fractions are not parsed] and a completion with no integer scores as wrong. The fraction of completions
that hit the 160-token cap was #emph[not logged] in these runs (it is reported in the planned matched
experiment, §6); cap-induced truncation could bias adapter comparisons and is a known gap. QA: for each
example the distractor context (all 10 paragraphs, 2 gold + 8 distractor) is concatenated and truncated to
700 tokens (truncation can drop answer text; not audited), token-level F1 on the first output line. Sample
ranges: boundary follow-up GSM8K test[80:580] ($n=500$, overlapping earlier development slices, @app-history);
own-KV control and penalty-difference contrast $n=120$; QA $n=500$. Paired bootstrap CIs use 10,000 resamples
of per-example differences @bootstrap; per-example arrays are released.

#strong[Systems measurement.] transformers 5.5 / peft 0.20 / torch 2.11, one NVIDIA RTX PRO 4500 Blackwell
GPU; attention backend and exact timing protocol (warmup, repetitions, CUDA synchronization) are pinned in
the repo — reported timings here are single-run. @tab-cost and @tab-scale are separate runs (the ≈1%
base-prefill difference). Prefix-cache figures are KV tensor footprint; the single-allocation claim is a
pointer-identity check establishing initial aliasing only — peak memory under concurrent generation was not
measured.

#strong[Translator.] Per-head ridge maps calibrated on 40 held-out HotpotQA distractor contexts (`N_CAL=40`)
with RoPE-stripped keys; single-layer $l→l$ and top-$k$ multi-layer variants; ridge regularization
$lambda=10$. Reported cross-size fidelity (next-token top-1 agreement) and task retention are from that
calibration; the numerical comparison against direct reuse (no quality gain, added per-token matrix-multiply
cost) is in the repo. The map was not adopted.

#bibliography("refs.bib")
