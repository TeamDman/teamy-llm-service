# Julia-1 ONNX adapter

This crate runs the pinned Julia-1 ONNX graph through `ort =2.0.0-rc.12` and
ONNX Runtime 1.24.2. It shares the Rust encoder and finite-choice contract with
the native Julia provider. It accepts text-only requests with 2–20 stable option
IDs; it does not generate text or execute the chosen option.

The caller supplies local model and runtime paths. `runtime-pins.json` records
the official Microsoft Windows CUDA13 archive and inspected DLL identities.
The adapter never downloads or searches PATH for the core ONNX runtime. It
verifies the loaded runtime version and reports hashes of the runtime, graph
and external weights. Expected hashes can also be supplied to reject changed
artifacts before use.

## Device policy

CPU is an explicit device. CUDA registration errors are fatal. CUDA disables
CPU fallback unless `allow_cpu_fallback` is explicitly enabled. The pinned
graph assigns some nodes to CPU under ONNX Runtime's default placement, so
strict CUDA preparation currently fails before inference.

An explicitly permitted mixed CPU/CUDA session reports that permission. GPU EP
registration alone does not establish node placement. `profile_prefix` enables
an ORT trace; `finish_profiling()` returns its actual timestamped filename.
`scripts/summarize-profile.py --require-heavy-cuda` checks recorded CUDA
MatMul/Attention-family execution and reports CPU nodes separately. Its scope
is the profiled session and inputs, not every future request.

## Local validation

```powershell
cargo test -p teamy_llm_onnx_julia --all-targets --locked -j2
cargo build -p teamy_llm_onnx_julia --examples --locked -j2
```

`julia_onnx_smoke` accepts a model directory, explicit runtime library and
`cpu|cuda`. `julia_onnx_parity` also accepts independently acquired JSON
reference fixtures. Both require `--allow-cpu-fallback` to opt into mixed CUDA
placement. For the parity example, `TEAMY_ORT_PROFILE_PREFIX` enables the trace.
They acquire no artifacts and execute no model-source Python.

The six independent CPU reference cases pass their predeclared combined
absolute/relative logit and probability tolerances with exact encoded arrays,
IDs/order and selected options. This verifies those cases; it does not establish
calibrated confidence or authority to act on a choice. The explicitly allowed
mixed CUDA session passes the same six independent GPU reference cases. Its
trace records MatMul/FusedMatMul/Gemm on CUDA and integer shape/index operations
on CPU. Strict CUDA remains rejected for this graph.

Parity case timings cover inference only. The harness also reloads the tokenizer
to check each fixture; those times are not end-to-end CLI latency measurements.

Cancellation is cooperative before/after session preparation and inference,
and between hash blocks. It cannot interrupt an already executing GPU kernel.
