# Teamy LLM Service

Persistent local LLM service for Teamy tools.

V1 exposes Facet-serializable request/response/event models, extracts the Teamy-Studio Burn Qwen3.5 path, and keeps the Vox-facing service boundary isolated from backend internals.

## Workspace

- `teamy_llm_core`: shared service types.
- `teamy_llm_burn_qwen35`: Teamy-Studio Burn Qwen3.5 backend extraction and Python reference/export harness.
- `teamy_llm_julia_common`: shared publisher-compatible Julia encoding and qualification fixtures.
- `teamy_llm_burn_julia` and `teamy_llm_onnx_julia`: native and ONNX finite-choice providers.
- `teamy_llm_makepad_gguf`: resident quantized CUDA generation with pinned weight and prompt-token checks.
- `teamy_llm_service`: in-process service facade, model registry/cache, serialized per-model generation, and optional Vox trait/API.
- `teamy_llm_cli`: local commands for health, model inspection/preparation, and prompt generation.

The workspace pins `facet`, `figue`, and `vox` to the same TeamDman/facet git revision used by Cloud-Terrastodon.

## Finite-choice core contract

The additive [finite-choice module](crates/teamy_llm_core/src/finite_choice.rs)
defines Facet request/result types and a provider seam. Input and candidate options
can each contain text and encoded visual bytes. Each provider declares supported
modalities independently. Unsupported visual input fails before provider invocation.
Results preserve option IDs and order, finite logits and normalized softmax probabilities.
Those probabilities are not calibrated confidence.

The [complete example](crates/teamy_llm_core/examples/finite_choice_contract.rs)
uses fixed synthetic logits. It compiles and checks the contract without weights,
an image decoder, a GPU or network access:

```powershell
cargo run --offline --locked -p teamy_llm_core --example finite_choice_contract
cargo test --offline --locked -p teamy_llm_core
```

Julia-1's declared profile is text-only with 2 to 20 choices. The core module is independent
of model execution. The local `teamy-llm decide` command connects it to native Burn and ONNX
adapters; it does not add a service RPC or image interpretation. Both adapters have passed
six independent publisher-reference cases with exact encoding/order and fixed numerical tolerances.
This validates the tested cases, rather than establishing calibration or full-context performance.

## Real decision models

Supply version 1 JSON on stdin or with `--request '@choice.json'`. The request contains
`input: {text, visuals: []}` and `options: [{id, content: {text, visuals: []}}]`.
Use explicit local model locations; no command downloads weights implicitly.

```powershell
teamy-llm decide --help
Get-Content choice.json -Raw | teamy-llm decide --provider native --model julia-1 --model-dir '<native-model-dir>' --device cuda
teamy-llm decide --provider onnx --model julia-1 --model-dir '<onnx-model-dir>' --runtime-library '<onnxruntime.dll>' --device cuda --allow-cpu-fallback --request '@choice.json'
```

The ONNX CUDA flag explicitly permits mixed placement: the qualified graph runs matrix
operations on CUDA and integer shape/index operations on CPU. Without that flag, this graph
fails strict CUDA preparation. `--profile-prefix` records placement and returns its timestamped
trace path in the JSON result. `--repeat` reuses the loaded provider and records each inference
duration. Machine results remain on stdout; logs remain on stderr, including with JSON output.

## Quantized OrcaRouter prompts

The new `qwen3.8-27b-uncensored-q4-k-m` model uses Makepad's native CUDA GGUF executor, with
resident quantized weights and independent Hugging Face token-ID verification before prefill.
The existing Qwopus Burn model and configured default are retained. This slice supports text
prompts; it does not implement vision or tool calls for the new model.
The Orca session has a 1,024-token context shared by the rendered prompt and requested output.
Omitting explicit model selection warns and lists the available models.

```powershell
teamy-llm model list
teamy-llm prompt --model qwen3.8-27b-uncensored-q4-k-m --no-thinking --max-new-tokens 64 'Why is the sky blue?'
teamy-llm benchmark --model qwen3.8-27b-uncensored-q4-k-m --no-thinking --repeat 3 --max-new-tokens 32 --log-level debug --log-file '<capture.ndjson>'
```

`--log-level` aliases `--log-filter`; `--log-file` appends NDJSON to an explicit file.
For local builds, `scripts/cargo-cuda.ps1 -CargoArguments @('build', '--release', '--locked', '-p', 'teamy_llm_cli')`
uses the existing CUDA toolkit and fails if native kernels cannot be built. See
[performance analysis](https://teamdman.github.io/teamy-docs/performance-analysis.html) for the
template and profiler practices behind these captures.

## CLI cancellation

The local CLI accepts the same cancellation options as `teamy-rust-cli`:

```text
teamy-llm prompt --stop-after-duration 2m --max-new-tokens 200 "Why is the sky blue?"
teamy-llm prompt --stop-after "LLM model weights loaded" --max-new-tokens 200 "Why is the sky blue?"
```

`--stop-after-duration` accepts `humantime` durations such as `500ms`, `2s`, and `1m`.
`--stop-after` matches a tracing span or event message. Cancellation is cooperative: it is checked
between model-loading and generation operations, so it cannot preempt an already-running Burn/CUDA
kernel.

## Thinking modes

The CLI defaults to direct answers using each model's qualified prompt template. Use `--thinking`
when deliberate reasoning is useful. Direct mode defaults to 256 new tokens; Thinking defaults to
1,024 for the legacy model and 512 for explicitly named Orca selection. With an Orca `--model-dir`,
set `--max-new-tokens` explicitly to leave room for the prompt. That flag overrides all defaults.

```text
teamy-llm prompt --no-thinking --max-new-tokens 128 "Why is the sky blue?"
teamy-llm prompt --thinking --max-new-tokens 1024 "Compare two approaches and justify the choice."
```

A long-lived `TeamyLlmService` also retains the default hybrid CUDA runtime per model after its
first request. This reuses lazily populated GPU weight caches for later requests; preparing a model
invalidates its cached runtime so updated weights are never reused.
The service also caches tokenizers, reloading when file length or modification time changes.
Runtime reuse requires the same service process; separate CLI prompt invocations remain cold.
Switching between Burn and GGUF releases the competing runtime to limit GPU memory pressure.

## Performance baseline

The benchmark command defaults to the fixed prompt `Why is the sky blue?`, a 200-token cap, and
both native prompt modes. It prints one machine-readable row per mode:

```text
cargo run --release -p teamy_llm_cli -- benchmark --timeout-ms 600000
```

`time_to_first_token_ms` starts when the request is submitted, so it includes model loading and
prompt processing. `time_to_completion_ms` covers the entire request. `tokens_per_second` is the
token-event emission throughput (`(generated token count - 1) / (completion - first event)`). A first
event can contain empty text, and the GGUF adapter can delay events for UTF-8 completion; these are
end-to-end callback timings, not kernel-only decode or first-visible-text timings. Run each
mode in a new process (`--no-thinking` or `--thinking`) for a process-cold comparison; run both in
one invocation to observe the long-lived CUDA runtime cache's warm follow-up request. Use
`--repeat 3` with one explicit thinking mode to compare process-cold and resident requests.

## CPU profiling

The Burn backend has optional Tracy CPU zones around the decode path. This is an opt-in feature:
the normal build has no Tracy dependency. Its client disables default features and uses
`only-localhost`, matching `teamy-rust-cli`; it will not use broadcast discovery or request a
Windows firewall exception. With the Tracy capture UI listening, enable it only for the profiling
build:

```text
cargo run --release -p teamy_llm_cli --features tracy -- benchmark --no-thinking --max-new-tokens 64 --timeout-ms 600000
```

Save the resulting `.tracy` capture, then use the local `teamy-profiler` repository to rank the
exclusive CPU time of its spans:

```text
teamy-profiler tracy csv export <capture.tracy> --top 25 --table
```

This exposes host-side work and synchronization in the current hybrid decode path. It does not
replace a GPU-timing tool when kernel-level CUDA timings are required.

## CUDA runtime

Prompt generation requires CUDA and does not fall back to CPU inference. Julia decisions accept
an explicit CPU device as well as CUDA. The workspace pins Cudarc to the
CUDA 13.3 ABI and checks the following before inference:

- NVIDIA driver series 580 or newer, with `nvidia-smi` reporting CUDA Version 13.3 or newer.
- `CUDA_PATH` set to the CUDA 13.3 Toolkit installation.
- `%CUDA_PATH%\bin\x64\nvrtc64_130_0.dll` and `nvrtc-builtins64_133.dll` present.

Install the CUDA 13.3 Toolkit with its `nvrtc_13.3` component, open a fresh terminal, then rebuild.
The program adds `%CUDA_PATH%\bin\x64` to its own DLL search path before it loads Cudarc, matching
the CUDA 13.3.1 Windows layout.
Because Burn compiles CUDA code to PTX with NVRTC, a driver which merely meets CUDA 13.x minor-version
compatibility but reports an older CUDA version cannot run it. Update the NVIDIA driver before running
this program if it reports less than 13.3.

The unit tests check the CUDA contract. The ignored hardware test validates the configured Toolkit and
driver on a compatible machine:

```text
cargo test --workspace cuda_13
cargo test -p teamy_llm_burn_qwen35 installed_cuda_runtime_matches_contract_when_cuda_path_is_configured -- --ignored
```
