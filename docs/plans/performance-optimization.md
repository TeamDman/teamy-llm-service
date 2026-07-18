# Performance optimization loop

**Plan status:** Active
**Primary implementation root:** `G:\Programming\Repos\teamy-llm-service` on `agent/async-model-preparation`
**Last updated:** 2026-07-18

## How to update this plan

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked

Keep at most one work item in progress. Update a work item's status and its completion notes
together. When an iteration completes, retain its evidence below the completed items, then add a
new bounded iteration only after the next profile and decision gate are closed. Do not pre-plan
future optimizations beyond that next iteration.

## Purpose

Improve local CUDA inference performance without changing model output semantics or reintroducing
CPU inference. Each iteration must turn one measured bottleneck into a small, verified code change,
then append comparable release measurements to `PERFORMANCE_BASELINES.md` for later plotting.

## Scope

**In scope for the current iteration:** completed fused linear-attention input projections. The
next code change is intentionally unselected until a fresh caller-attributed profile of this
accepted implementation closes the next decision gate.

**Out of scope for the current iteration:** changing the model, quantization, prompt contract, Vox
API, CUDA-version policy, concurrency model, or planning subsequent optimization ideas. No CPU
fallback may be introduced as a production path; existing diagnostic paths may remain explicitly
non-default correctness oracles.

## Established foundation

### Verified facts

- `PERFORMANCE_BASELINES.md` is append-only and defines the benchmark fields and measurement
  conventions. It records process-cold release runs of `Why is the sky blue?` at a 200-token cap:
  direct is **43.263 s TTFT, 84.837 s completion, 4.811 tok/s**; thinking is **40.846 s TTFT,
  82.476 s completion, 4.804 tok/s**.
- `teamy-llm benchmark` reports TTFT, completion, generated-token count, and post-first-token
  throughput. Its implementation is in `crates/teamy_llm_cli/src/main.rs`.
- `TeamyLlmService` retains a CUDA runtime per model for long-lived processes in
  `crates/teamy_llm_service/src/lib.rs`; this is already a separate TTFT optimization.
- The default `Hybrid` decode path performs prefill with CUDA tensors but hands each generated token
  to `forward_token_hidden`, `forward_decoder_layer_single`, and the host-resident full/linear
  attention state in `crates/teamy_llm_burn_qwen35/src/burn_text.rs`. The same path calls
  `tensor_to_vec_f32` and copies full lm-head logits to the host before selecting a token.
- Optional Tracy profiling is available through `--features tracy`; it uses disabled default
  features and `only-localhost`, matching `teamy-rust-cli`. `teamy-profiler` summarizes the saved
  `.tracy` capture's CPU zones but is not a GPU-kernel profiler.

### Confirmed decisions

- The service is GPU-only; CPU inference is undesirable.
- Direct and thinking generation both remain supported. Neither mode is assumed to be the decode
  bottleneck because their recorded throughput is effectively equal.
- The existing Markdown baseline file is the source of truth for graphable performance data.

### Assumptions to validate

- Removing host transfers and host scalar attention/linear-recurrence work from the default hybrid
  decode path will be the largest current throughput improvement.
- A GPU-resident implementation can match the current hybrid path's generated token IDs for the
  fixed benchmark and focused fixtures.

## Decision gate — closed for iteration 1

| Question | Required decision | Acceptance consequence |
| --- | --- | --- |
| Dominant decode work | Closed: the 64-token capture ranks `llm_burn_tensor_to_vec_f32` first and `llm_burn_lm_head_scan` third. | The first change targets only lm-head logits/argmax; host-resident decoder-state work remains a later decision. |
| Correctness oracle | Closed: a 16-token direct, no-system-prompt oracle is recorded under item 1.3. | New CUDA argmax must emit the same IDs in the same order. |
| Bounded implementation slice | Closed: keep the existing hybrid decoder state and replace only full-logit host extraction and host argmax with CUDA argmax plus scalar ID retrieval. | No attention, linear recurrence, prompt, Vox, or cache behavior changes belong in this slice. |
| Expected gain | Closed: target **at least 1.5× direct decode throughput (>= 7.2 tok/s)**. | Benchmark results are classified as met, partially met, or missed; a miss triggers diagnosis rather than silent scope expansion. |

## Source and implementation references

| Purpose | Reference |
| --- | --- |
| Graphable baseline data | `PERFORMANCE_BASELINES.md` |
| Benchmark implementation and parser tests | `crates/teamy_llm_cli/src/main.rs` |
| Long-lived service runtime cache | `crates/teamy_llm_service/src/lib.rs` |
| Hybrid generation loop | `crates/teamy_llm_burn_qwen35/src/burn_text.rs` — `generate_with_hybrid_burn_text_runtime_with_runtime` |
| Host decode and state | `crates/teamy_llm_burn_qwen35/src/burn_text.rs` — `forward_token_hidden`, `forward_decoder_layer_single`, `forward_full_attention_single`, `forward_linear_attention_single` |
| Host transfer and lm-head selection | `crates/teamy_llm_burn_qwen35/src/burn_text.rs` — `tensor_to_vec_f32`, `greedy_next_token_from_hidden_tensor` |
| Existing CPU/incremental comparison APIs | `crates/teamy_llm_burn_qwen35/src/burn_text.rs` — `compare_with_incremental_burn_text_runtime` and hidden/layer diagnostics |
| Tracy capture and summary instructions | `README.md`; `G:\Programming\Repos\teamy-profiler\README.md` |

## Execution order

1. Capture and summarize the current performance pattern.
2. Preserve/confirm the comparable baseline data.
3. Close the decision gate, name one code slice, and record its predicted result.
4. Implement that slice with a correctness oracle.
5. Validate, benchmark, and append data.
6. Close this iteration; begin the next one at step 1 rather than extending this scope.

## Iteration 1 — CUDA lm-head argmax

### [x] 1.1 Establish comparable benchmarks, baseline storage, and profiling capability

**Work:**

- Provide a release benchmark with direct and thinking controls.
- Store append-only baseline rows in the repository.
- Enable an opt-in, localhost-only Tracy configuration compatible with `teamy-profiler`.

**Validation:**

```pwsh
cargo test -p teamy_llm_cli --features tracy
cargo clippy -p teamy_llm_cli --features tracy -- -D warnings
```

**Completion criteria:** The current source can record comparable metrics and produce an optional
CPU profiling capture without changing the normal feature set.

**Completion notes:** `5605696` added the benchmark command; `cf2cee7` added the baseline log;
`100fd7d` added localhost-only Tracy profiling. The two feature-enabled validation commands passed
on 2026-07-13. The source tree was clean after `100fd7d`.

### [x] 1.2 Capture and summarize the current decode profile

**Work:**

- Start a local Tracy capture on loopback, then run the 64-token direct profiling command below.
- Save the `.tracy` capture outside the source tree or in an explicitly chosen ignored capture
  location; do not commit large raw captures by default.
- Summarize the top 25 exclusive CPU zones with `teamy-profiler` and add the relevant names,
  aggregate time, count, capture path, and command output summary to this item's completion notes.
- Compare the top zones with the host-transfer/decode functions listed above. If they do not
  support the current hypothesis, return to the decision gate before implementation.

**Validation:**

```pwsh
cargo run --release -p teamy_llm_cli --features tracy -- benchmark --no-thinking --max-new-tokens 64 --timeout-ms 600000
```

```pwsh
G:\Programming\Repos\teamy-profiler\target\release\teamy-profiler.exe tracy csv export <capture.tracy> --top 25 --table
```

**Completion criteria:** A saved capture and its CPU-zone summary identify the observed dominant
host work during a reproducible direct decode run.

**Completion notes:** Capture `C:\Users\Teamy\Documents\Codex\2026-07-13\that\teamy-llm-profile\direct-64_2026-07-13_23-43-17.tracy` was recorded with the listed release command on 2026-07-13. Its direct result was 35.612 s TTFT, 49.107 s completion, and 4.743 tok/s for 64 tokens, consistent with the 200-token baseline. `teamy-profiler tracy csv export ... --top 25 --table` reported `llm_burn_tensor_to_vec_f32{rank=3}` as the largest self-time (13.847 s across 12,034 calls), `llm_burn_lm_head_scan` as 5.158 s across 64 calls, and `llm_burn_tensor_to_vec_f32{rank=1}` as 3.216 s across 153 calls. These results support avoiding host tensor extraction and host lm-head argmax first.

### [x] 1.3 Close the implementation slice and performance prediction

**Work:**

- Record the capture-supported bottleneck and the exact functions/state affected.
- Choose the first bounded change supported by the capture.
- Capture a small token-ID oracle from the unchanged implementation.
- Record the prediction and acceptance threshold before production decoder code changes.

**Validation:**

```pwsh
cargo run --release -p teamy_llm_cli -- benchmark --no-thinking --max-new-tokens 200 --timeout-ms 600000
```

**Completion criteria:** The next code change, correctness oracle, and measurable target are
written here before production decoder code changes.

**Completion notes:** The selected slice is CUDA lm-head argmax only: preserve the existing hybrid decoder state and move full-logit reduction/argmax from `greedy_next_token_from_hidden_tensor` to CUDA, returning only the chosen token ID to the host. The unchanged release oracle used no system prompt, direct mode, `Why is the sky blue?`, a 16-token cap, and the managed `qwopus-3.5-9b-coder-q4-k-m` artifact. Its IDs were `[760, 12515, 7701, 6105, 15048, 4016, 310, 264, 24057, 2512, 2972, 28232, 60845, 69377, 159034, 271]`. The profile shows 5.158 s of lm-head host-scan self time over the 13.495 s post-first-token interval; the conservative target is >= 7.2 tok/s (1.5× the 4.811 tok/s baseline), not the earlier whole-decoder 2–4× hypothesis.

### [x] 2.1 Prototype and reject CUDA lm-head argmax

**Work:**

- Change only the default `Hybrid` production path's lm-head selection selected by item 1.3.
- Add a CUDA tensor argmax/reduction path that accepts the already-device-resident final hidden
  state and avoids copying a complete logits vector to the host.
- Transfer only the selected token ID or the smallest equivalent scalar result needed for token
  streaming.
- Preserve cancellation checks, timeout accounting, token streaming, model runtime caching, and the
  direct/thinking public contract.
- Do not change attention, linear recurrence, decode-state representation, or existing diagnostic
  modes in this slice.

**Validation:**

```pwsh
cargo fmt --all -- --check
cargo test -p teamy_llm_burn_qwen35
cargo test -p teamy_llm_cli
```

**Completion criteria:** The prototype either meets its performance target while preserving the
oracle, or is removed with an evidence-backed rejection before another change is attempted.

**Completion notes:** The prototype replaced the host logits vector/max scan with Burn
`Tensor::argmax(2)`, then copied only its scalar integer result. A focused CPU-backend test and
`cargo test -p teamy_llm_burn_qwen35` passed (24 passed, 1 ignored). The real CUDA 16-token oracle
matched exactly. The implementation was removed after item 3.1 measured a throughput regression;
the production source therefore remains on the prior host-scan path.

### [x] 3.1 Prove correctness and measure the result

**Work:**

- Run a focused unit test for the argmax reduction and the real GPU token-ID oracle.
- Run separate process-cold direct and thinking 200-token release benchmarks. Append one row per
  completed run to `PERFORMANCE_BASELINES.md`, including datetime, mode, commit, model, token cap,
  cache state, and whether generation ended at the cap or EOS.
- Compare measured throughput and TTFT with the prediction in item 1.3. If the target is missed,
  remove the prototype, document the profile evidence, and stop this iteration rather than adding
  an unplanned fix.

**Validation:**

```pwsh
cargo test -p teamy_llm_burn_qwen35
cargo run --release -p teamy_llm_cli -- benchmark --no-thinking --max-new-tokens 200 --timeout-ms 600000
cargo run --release -p teamy_llm_cli -- benchmark --thinking --max-new-tokens 200 --timeout-ms 600000
```

**Completion criteria:** Focused correctness evidence and appended comparable metrics show whether
the predicted performance change was met; a rejected prototype is removed before the item closes.

**Completion notes:** The direct process-cold result was 39.541 s TTFT, 88.466 s completion, and
4.088 tok/s. The thinking result was 42.133 s TTFT, 89.392 s completion, and 4.232 tok/s. Both are
below their 4.811/4.804 tok/s baselines and miss the >= 7.2 tok/s target, despite exact 16-token
oracle parity. Both rows were appended to `PERFORMANCE_BASELINES.md`. The hypothesis is rejected:
the Burn/CUDA argmax plus scalar synchronization is slower than the existing host scan on this
model and driver.

### [x] 4.1 Close the rejected iteration and seed only the next one

**Work:**

- Mark the prior items complete with their commands, commit IDs, profile facts, and benchmark rows.
- State the measured conclusion and that the prediction was missed; retain no production code from
  the rejected experiment.
- Begin a new bounded iteration at item 1.2 only after a new profile identifies the next bottleneck;
  do not pre-populate its implementation tasks yet.

**Validation:**

```pwsh
git status --short
```

**Completion criteria:** The repository is clean, the completed iteration is self-explanatory, and
the next iteration begins only with a fresh profiling decision rather than speculative
implementation.

**Completion notes:** The CUDA argmax source and its temporary unit test were removed after the
regression. The append-only baseline log retains both failed measurements, and this plan retains the
capture, oracle, target, and rejection rationale. No next implementation slice is preselected; the
next safe action is a new profile-driven decision.

## Iteration 2 — caller-attributed hybrid-decode profile

### [x] 5.1 Attribute host tensor extraction to its call sites

**Work:**

- Add `#[track_caller]` metadata to the opt-in `llm_burn_tensor_to_vec_f32` Tracy span so the
  profiler separates same-rank extractions by direct source line.
- Build with `--features tracy`, capture the existing 64-token direct benchmark on loopback, and
  summarize the top exclusive CPU zones with `teamy-profiler`.
- Record the capture path, command, and largest caller-specific rank-3 transfers below. Do not
  append these feature-enabled measurements to `PERFORMANCE_BASELINES.md`.

**Validation:**

```pwsh
cargo test -p teamy_llm_burn_qwen35 --features tracy
G:\Programming\Repos\teamy-profiler\target\release\teamy-profiler.exe tracy csv export <capture.tracy> --top 25 --table
```

**Completion criteria:** The profile distinguishes the relevant decode call families well enough
to choose one production slice and a predicted improvement, or it explicitly shows that a GPU
kernel profiler is required before one can be chosen.

**Completion notes:** `#[track_caller]` now adds source line/column fields only to the existing
`extended_observability` `llm_burn_tensor_to_vec_f32` span. `cargo fmt --all -- --check` and
`cargo test -p teamy_llm_burn_qwen35 --features tracy` passed (23 passed, 1 intentionally ignored
CUDA-contract test) on 2026-07-18. Capture
`C:\Users\Teamy\Documents\Codex\2026-07-13\that\teamy-llm-profile\direct-64-caller-attributed_2026-07-18_12-34-35.tracy`
was recorded with the release direct 64-token command and a bounded localhost-only `tracy-capture`
run. Its diagnostic result was 26.802 s TTFT, 39.593 s total, and 5.004 tok/s; it is not appended
to the non-Tracy baseline log. The untruncated `teamy-profiler` export identified line 1150
(`forward_mlp_single` down projection) as the largest rank-3 extraction: 3.373 s across 2,048
calls (32 layers x 64 decode steps). Other repeated linear-attention transfers were line 1347
(1.099 s), 1355 (0.998 s), 1419 (0.984 s), 1304 (0.967 s), and 1342 (0.878 s), each 1,536 calls.
The existing host lm-head scan remained 2.931 s but is not retried after the rejected CUDA-argmax
experiment.

### [x] 5.2 Close the next decision gate

**Work:**

- Use item 5.1 evidence to select exactly one bounded production change, record the affected
  functions/state, a small token-ID oracle, and a measurable prediction before implementation.
- If the CPU trace cannot distinguish the dominant work, stop at that conclusion and select an
  appropriate GPU-kernel profiling method instead of speculating.

**Completion criteria:** The next production change is justified by the capture and has a stated
acceptance threshold; no implementation tasks are listed before that decision.

**Completion notes:** A MLP-only output change would merely move the same-sized host transfer to
the next layer because the current single-token decoder accepts and returns `Vec<f32>`. The chosen
bounded slice is therefore a CUDA-resident hidden-state bridge across the default hybrid
single-token layer loop: tensor-native RMS norm and residual additions, MLP tensors retained on
device, and tensor hidden state passed to the next layer. The existing host full/linear attention
state math, the host lm-head scan, prompt contract, cancellation, caching, and diagnostic paths are
unchanged. The prior 16-token direct no-system-prompt oracle remains the correctness gate:
`[760, 12515, 7701, 6105, 15048, 4016, 310, 264, 24057, 2512, 2972, 28232, 60845, 69377, 159034, 271]`.
The dominant eliminated transfer alone is 3.373 s of the capture's 12.791 s post-first-token time;
after allowing for added GPU norm/reduction work, the conservative direct acceptance target is
**>= 6.0 tok/s** (about 25% over the 4.811 tok/s comparable baseline).

### [x] 6.1 Implement and validate the CUDA-resident hidden-state bridge

**Work:**

- Add a tensor-native Qwen RMS norm and use CUDA tensors for hidden residuals and MLP results in
  the default hybrid single-token layer loop.
- Keep full/linear attention's existing decode-state arithmetic and the host lm-head scan intact;
  no generic CPU execution path may be selected for production generation.
- Add focused CPU-backend helper parity coverage, then verify the real CUDA 16-token token-ID
  oracle before collecting normal-release benchmark data.

**Validation:**

```pwsh
cargo fmt --all -- --check
cargo test -p teamy_llm_burn_qwen35
cargo test -p teamy_llm_cli
```

**Completion criteria:** The default hybrid path preserves the oracle and no longer performs the
line-1150 MLP output extraction for generated tokens. Comparable direct and thinking benchmarks
will be run only after this implementation item is complete.

**Completion notes:** The default hybrid generation loop now calls
`forward_token_hidden_tensor`, which retains `[1, 1, hidden]` CUDA tensors across token-decoder
layers. `forward_decoder_layer_tensor_single` performs its two residual additions and Qwen RMS
norms on the device, while `forward_mlp` returns its CUDA tensor directly. The existing
full/linear attention state math and host lm-head scan remain in place; the old `Vec<f32>`
single-token functions remain diagnostic paths and are not selected for production hybrid
generation. This removes the prior line-1150 `forward_mlp_single` output extraction from generated
tokens rather than moving the same transfer to the next layer. The new CPU-backend
`tensor_rms_norm_matches_host_reference` test passed. `cargo fmt --all -- --check`,
`cargo test -p teamy_llm_burn_qwen35` (24 passed, 1 intentionally ignored),
`cargo test -p teamy_llm_cli` (3 passed), and `cargo clippy -p teamy_llm_cli -- -D warnings` all
passed on 2026-07-18. The real CUDA direct no-system-prompt 16-token oracle matched exactly:
`[760, 12515, 7701, 6105, 15048, 4016, 310, 264, 24057, 2512, 2972, 28232, 60845, 69377, 159034, 271]`.
The append-only release, process-cold 200-token rows record direct **29.873 s TTFT, 62.770 s
completion, 6.079 tok/s** and thinking **29.672 s TTFT, 63.826 s completion, 5.856 tok/s**.
Direct exceeded the >= 6.0 tok/s prediction and is 26.4% above its 4.811 tok/s baseline;
thinking is 21.9% above its 4.804 tok/s baseline.

## Iteration 3 — fused linear-attention input projections

### [x] 7.1 Profile the accepted CUDA-hidden decoder

**Work:**

- Build the accepted source with `--features tracy`, capture the standard 64-token direct run on
  loopback, and export untruncated caller-attributed rows with `teamy-profiler`.
- Compare the remaining host-transfer families with item 5.1. Do not append feature-enabled timing
  to `PERFORMANCE_BASELINES.md`.

**Completion criteria:** The profile either identifies a next bounded production slice or shows
that GPU-kernel profiling is required before one can be selected.

**Completion notes:** `cargo test -p teamy_llm_burn_qwen35 --features tracy` passed (24 passed,
1 intentionally ignored) on 2026-07-18. Capture
`C:\Users\Teamy\Documents\Codex\2026-07-13\that\teamy-llm-profile\direct-64-post-bridge_2026-07-18_13-10-26.tracy`
was recorded using the same direct 64-token command and bounded localhost-only capture as item
5.1. Its diagnostic result was 33.210 s TTFT, 45.242 s total, and 5.319 tok/s; it is not a
normal-release baseline. The prior MLP extraction is absent. The largest remaining attributed
host transfer is linear-attention `in_proj_qkv` at line 1393: 3.066 s across 1,536 calls (24
linear layers x 64 steps). Its companion linear projections are `in_proj_b` at line 1436 (1.291
s), `in_proj_z` at line 1431 (0.883 s), and `in_proj_a` at line 1444 (0.423 s), all also 1,536
calls. The host lm-head span remains 4.715 s, but the CUDA argmax alternative was already rejected
with a real-release regression in iteration 1.

### [x] 7.2 Close the fused-projection decision gate

**Work:**

- Select one implementation slice, correctness gate, and measurable prediction from item 7.1.
- Reject any candidate that merely transfers an equally sized host buffer at the next boundary or
  requires a broad recurrent-state rewrite.

**Completion criteria:** The next change is bounded, preserves the GPU-only policy, and has a
stated acceptance threshold before source changes begin.

**Completion notes:** Selected slice: concatenate the four compatible linear-attention projection
weights along their output-row dimension once per layer, cache that fused CUDA tensor, execute one
`linear_forward_3d`, extract its concatenated output once, and split it into the existing `qkv`,
`z`, `b`, and `a` host values. The projection outputs have the same input width and are consumed
only by the existing host linear-attention state; this preserves that state and eliminates three
GPU-to-host synchronization/extraction boundaries and three small matrix launches per linear
layer/token without changing the number of output values. The original 16-token direct oracle
remains the correctness gate. Because bytes still cross the same host boundary, the conservative
direct target is **>= 6.35 tok/s** (about 4.5% above the 6.079 tok/s accepted measurement), not a
transfer-volume-based claim.

### [x] 8.1 Implement and validate fused input projections

**Work:**

- Add a lazy per-layer fused-projection CUDA cache and use it in the default hybrid
  linear-attention path; release the four superseded cached projection tensors after fusion so the
  change does not retain duplicate GPU weights.
- Split the single extracted result with manifest-derived widths and preserve the existing conv
  history, recurrence, token streaming, cancellation, and direct/thinking contract.
- Add CPU-backend projection-output parity coverage and verify the real CUDA 16-token oracle
  before taking normal-release benchmarks.

**Validation:**

```pwsh
cargo fmt --all -- --check
cargo test -p teamy_llm_burn_qwen35
cargo test -p teamy_llm_cli
```

**Completion criteria:** The hybrid linear path performs one fused projection/extraction instead
of four separate input projection/extraction pairs, preserves the oracle, and has comparable
direct and thinking measurements appended to the baseline log.

**Completion notes:** `Qwen35TextRuntime` now lazily concatenates the four compatible
linear-attention input weights along output rows into a per-layer CUDA cache. All three
linear-attention paths (prefill, stateful prefill, and default tensor-hidden single-token decode)
call one `linear_forward_3d`, extract the result once, and de-interleave each token's `qkv`, `z`,
`beta`, and `a` spans for the existing host recurrence. The cache removes the superseded four
device tensors after fusion, avoiding duplicate retained GPU weights. A two-token CPU parity test
verifies both row concatenation and sequence-aware de-interleaving. `cargo fmt --all -- --check`,
`cargo test -p teamy_llm_burn_qwen35` (25 passed, 1 intentionally ignored),
`cargo test -p teamy_llm_cli` (3 passed), and `cargo clippy -p teamy_llm_cli -- -D warnings` all
passed on 2026-07-18. The real CUDA direct no-system-prompt 16-token oracle again matched exactly:
`[760, 12515, 7701, 6105, 15048, 4016, 310, 264, 24057, 2512, 2972, 28232, 60845, 69377, 159034,
271]`. The append-only process-cold release rows record direct
**29.100 s TTFT, 55.854 s completion, 7.475 tok/s** and thinking **36.257 s TTFT, 62.829 s
completion, 7.527 tok/s**. Direct exceeds the >= 6.35 tok/s prediction and improves 23.0% over
the prior accepted 6.079 tok/s direct measurement; thinking improves 28.5% over 5.856 tok/s.

## Iteration 4 — eliminate the generated-token hidden-state lm-head round trip

### [x] 9.1 Profile the accepted fused-projection decoder

**Work:**

- Build the accepted source with `--features tracy`, capture the standard 64-token direct run on
  loopback, and export caller-attributed rows with `teamy-profiler`.
- Compare the remaining host-transfer families with item 7.1. Do not append feature-enabled timing
  to `PERFORMANCE_BASELINES.md`.

**Completion criteria:** The profile identifies whether a bounded transfer or host-work slice
remains after fused projections, without treating Tracy CPU-zone time as GPU timing.

**Completion notes:** `cargo test -p teamy_llm_burn_qwen35 --features tracy` passed (25 passed,
1 intentionally ignored) on 2026-07-18. Capture
`C:\Users\Teamy\Documents\Codex\2026-07-13\that\teamy-llm-profile\direct-64-post-fused_2026-07-18_15-12-07.tracy`
recorded the usual loopback direct 64-token run; its diagnostic result was 29.406 s TTFT,
37.692 s completion, and 7.724 tok/s, and is not a normal-release baseline. The fused-projection
extractions are gone as separate callers. The largest remaining attributed transfer family is now
the single fused linear-attention extraction at line 930 (4.156 s across 1,560 calls), which is
coupled to the current host recurrent state and therefore requires a broader rewrite. The host
lm-head span remains 3.196 s across 64 calls; the full logits extraction at line 2545 accounts
for 1.112 s. The previously tested CUDA argmax alternative remains rejected because it regressed
normal-release throughput.

### [x] 9.2 Close the device-hidden lm-head decision gate

**Work:**

- Select one small production slice and state its token-oracle gate and measurable prediction.
- Preserve the existing logits host scan and its failure behavior; do not retry the rejected CUDA
  argmax experiment or start a recurrent-state rewrite in this iteration.

**Completion criteria:** The selected change removes a real boundary in the default path without
changing the token-selection policy or expanding into the host linear-attention state.

**Completion notes:** Selected slice: for generated tokens after prefill, pass the existing
`[1, 1, hidden]` CUDA tensor directly to the lm-head projection rather than first downloading the
hidden state and immediately uploading it again. The logits will still be downloaded and scanned
on the host exactly as today, and the prefill path remains unchanged. On device lm-head failure,
the existing disabled flag and host selection path will remain available. The 16-token direct CUDA
oracle remains the correctness gate. This removes a small synchronized hidden-state round trip
rather than a large transfer, so the conservative direct target is **>= 7.60 tok/s** (about 1.7%
above the accepted 7.475 tok/s result).

### [x] 10.1 Implement and validate the device-hidden lm-head handoff

**Work:**

- Add a direct tensor-hidden lm-head selection entry point for the default post-prefill hybrid
  loop, sharing the exact host logits scan with the current tensor lm-head path.
- Retain the existing host fallback when device lm-head execution has been disabled or fails.
- Run focused tests, the CUDA oracle, and comparable direct/thinking release benchmarks before
  accepting or rejecting the prediction.

**Validation:**

```pwsh
cargo fmt --all -- --check
cargo test -p teamy_llm_burn_qwen35
cargo test -p teamy_llm_cli
```

**Completion criteria:** Generated-token lm-head projection receives the already-resident CUDA
hidden tensor, token IDs preserve the oracle, and comparable measurements are appended to the
baseline log.

**Completion notes:** The post-prefill hybrid loop now consumes its existing `[1, 1, hidden]`
CUDA tensor through `greedy_next_token_from_hidden_state_tensor`. Its CUDA lm-head projection and
host logits download/max scan are shared with the prior host-hidden tensor path, so token selection
semantics did not change. The first prefill selection remains unchanged. If the device lm-head path
is disabled or fails, the new entry point defers to the existing hidden download and host row-scan
fallback. `cargo fmt --all -- --check`, `cargo test -p teamy_llm_burn_qwen35` (25 passed, 1
intentionally ignored), `cargo test -p teamy_llm_cli` (3 passed), `cargo clippy -p
teamy_llm_burn_qwen35 -- -D warnings`, and `cargo clippy -p teamy_llm_cli -- -D warnings` all
passed on 2026-07-18. The real CUDA direct no-system-prompt 16-token oracle again matched exactly:
`[760, 12515, 7701, 6105, 15048, 4016, 310, 264, 24057, 2512, 2972, 28232, 60845, 69377, 159034,
271]`. The append-only process-cold release rows record direct **36.437 s TTFT, 62.653 s
completion, 7.629 tok/s** and thinking **27.405 s TTFT, 53.126 s completion, 7.776 tok/s**.
Direct exceeds the >= 7.60 tok/s prediction by 2.1% and improves 2.1% over its 7.475 tok/s
pre-change measurement; thinking improves 3.3% over 7.527 tok/s. Direct TTFT was higher than the
previous process-cold sample, so the accepted result is limited to post-first-token throughput and
does not claim a TTFT improvement.

## Next safe action

Capture a fresh caller-attributed profile of this accepted handoff before selecting another
production change. Compare the remaining host linear-attention-state transfer with the lm-head
span, and choose only one new bounded slice supported by that capture.

## Overall completion criteria for an iteration

- The performance pattern was captured before code changes and the next change was explicitly
  predicted.
- The implementation preserves the chosen token-output oracle and GPU-only production policy.
- Release benchmarks use the baseline prompt/token cap and are appended without replacing history.
- Validation evidence and commit provenance appear under the relevant completed items.
- The plan leaves exactly one next safe action for the following optimization iteration.

## Risk register

| Risk | Guardrail |
| --- | --- |
| Tensor-state rewrite changes Qwen3.5 attention or linear-recurrence semantics. | Capture a token-ID oracle first; add focused parity tests and use existing comparison/diagnostic APIs. |
| A partial GPU implementation silently calls a CPU path. | Profile before/after; review the default hybrid loop for host tensor extraction; retain diagnostic modes as non-default only. |
| Benchmark numbers are not comparable. | Keep release build, exact prompt, 200-token cap, separate mode/process runs, and required comments in `PERFORMANCE_BASELINES.md`. |
| Tracy instrumentation changes timings or omits GPU time. | Use it only to locate CPU work; measure final performance without Tracy and use a GPU profiler only if kernel timing becomes necessary. |
| GPU memory or long prompts regress. | Preserve bounded token/prompt tests and add an explicit memory/long-prompt validation only if the selected tensor state changes its allocation model. |
