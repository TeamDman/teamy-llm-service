# Performance optimization loop

**Plan status:** Active
**Primary implementation root:** `G:\Programming\Repos\teamy-llm-service` on `agent/async-model-preparation`
**Last updated:** 2026-07-13

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

**In scope for the current iteration:** profile and, if the profile confirms the present hypothesis,
replace the default hybrid decoder's host-resident per-token work with GPU-resident decode state and
GPU token selection.

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

## Decision gate before implementation

| Question | Required decision | Acceptance consequence |
| --- | --- | --- |
| Dominant decode work | Use a 64-token Tracy capture to name the highest exclusive CPU zones and confirm that host conversion, CPU attention/recurrent math, or host argmax dominates. | If the capture contradicts the hypothesis, revise the current iteration before editing the decoder. |
| Correctness oracle | Choose and record a small fixed prompt/token-ID fixture plus a real-model comparison path before replacing the current decoder. | New GPU decode must match the oracle; a throughput gain with token drift is not accepted. |
| Bounded implementation slice | Decide whether one slice can keep hidden state, full-attention K/V state, linear-attention recurrence, and lm-head argmax on CUDA together. If not, choose the smallest independently correct subset. | The plan names one slice and one prediction before code changes begin. |
| Expected gain | Record a conservative prediction against the 4.811 tok/s direct baseline: **2–4× decode throughput (9.6–19.2 tok/s)** is the initial hypothesis. | Benchmark results are classified as met, partially met, or missed; a miss triggers diagnosis rather than silent scope expansion. |

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

## Iteration 1 — GPU-resident hybrid decode

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

### [~] 1.2 Capture and summarize the current decode profile

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

### [ ] 1.3 Close the implementation slice and performance prediction

**Work:**

- Record the capture-supported bottleneck and the exact functions/state affected.
- Choose a single bounded change. The expected choice is CUDA-resident hidden state, full-attention
  K/V state, linear-attention recurrent state, and lm-head argmax for the default hybrid loop, but
  do not start it unless item 1.2 supports it.
- Capture a small token-ID oracle from the unchanged implementation and state its prompt, token
  cap, model artifact, and expected IDs or comparison procedure.
- Confirm or revise the 2–4× throughput prediction and state the acceptance threshold.

**Validation:**

```pwsh
cargo run --release -p teamy_llm_cli -- benchmark --no-thinking --max-new-tokens 200 --timeout-ms 600000
```

**Completion criteria:** The next code change, correctness oracle, and measurable target are
written here before production decoder code changes.

### [ ] 2.1 Implement the selected GPU-resident decode slice

**Work:**

- Change only the default `Hybrid` production path selected by item 1.3.
- Keep tensors and recurrent decode state on the CUDA device through decoder layers; avoid
  `tensor_to_vec_f32`/`to_data()` in the per-token default path except for the minimum scalar data
  required to stream the selected token.
- Perform greedy token selection on CUDA when supported, rather than copying a complete logits
  vector to the host.
- Preserve cancellation checks, timeout accounting, token streaming, model runtime caching, and the
  direct/thinking public contract.
- Keep existing full/incremental paths explicitly diagnostic rather than silently becoming a CPU
  production fallback.

**Validation:**

```pwsh
cargo fmt --all -- --check
cargo test -p teamy_llm_burn_qwen35
cargo test -p teamy_llm_cli
```

**Completion criteria:** The selected decode path is CUDA-resident to the extent declared in item
1.3, preserves the recorded oracle, and does not create a normal CPU inference fallback.

### [ ] 3.1 Prove correctness and measure the result

**Work:**

- Add the smallest focused tests that prove the new tensor-state/argmax behavior and token parity.
- Run the current workspace checks and the ignored CUDA-contract test on the configured GPU host.
- Run separate process-cold direct and thinking 200-token release benchmarks. Append one row per
  completed run to `PERFORMANCE_BASELINES.md`, including datetime, mode, commit, model, token cap,
  cache state, and whether generation ended at the cap or EOS.
- Compare measured throughput and TTFT with the prediction in item 1.3. If the target is missed,
  document the profile evidence and stop this iteration rather than adding an unplanned fix.

**Validation:**

```pwsh
cargo test --workspace
cargo clippy --all-targets --all-features -- -D warnings
cargo test -p teamy_llm_burn_qwen35 installed_cuda_runtime_matches_contract_when_cuda_path_is_configured -- --ignored
cargo run --release -p teamy_llm_cli -- benchmark --no-thinking --max-new-tokens 200 --timeout-ms 600000
cargo run --release -p teamy_llm_cli -- benchmark --thinking --max-new-tokens 200 --timeout-ms 600000
```

**Completion criteria:** Correctness evidence, clean validation, and appended comparable metrics
show whether the predicted performance change was met.

### [ ] 4.1 Close the iteration and seed only the next one

**Work:**

- Mark the prior items complete with their commands, commit IDs, profile facts, and benchmark rows.
- State the measured conclusion and whether the prediction was met, partially met, or missed.
- Begin a new bounded iteration at item 1.2 only after a new profile identifies the next bottleneck;
  do not pre-populate its implementation tasks yet.

**Validation:**

```pwsh
git status --short
```

**Completion criteria:** The repository is clean, the completed iteration is self-explanatory, and
the first unfinished task is a fresh profiling decision rather than speculative implementation.

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
