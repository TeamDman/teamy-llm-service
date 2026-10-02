# Native Julia inference

Rust defines ModernBERT and Julia's decision head over Burn 0.21. CUDA is explicit;
CPU NdArray is an explicit debugging/reference device. Neither selection silently falls back.
No Python, LibTorch, TorchScript or Makepad runtime executes the numerical model.

CUDA currently uses a slower correctness configuration: chunked FP32 elementwise
multiplication and summation, rather than Burn's accelerated matrix multiplication
which may select TF32 internally. CPU uses ordinary NdArray FP32 matrix multiplication.
The CUDA multiplication temporary is bounded to `batch × heads × 32 × 128 × 32`
floats; complete attention matrices still scale with the square of sequence length.
This is a numerical baseline, not an optimised CUDA throughput claim.

The implementation loads the pinned FP32 safetensors architecture, retains backend weights,
applies bidirectional/full or sliding attention with RoPE, then the two pre-norm decision
layers and marker scorer. It does not run the optional action head or training/dropout paths.

The shared encoder preserves exact option order and rejects unsupported visuals. Callers
must validate the request before loading a provider. Cancellation is checked between loading
tensors and model layers; a running kernel/readback is not interrupted. The first implementation
uses dense attention masks, so memory and speed must be measured before expanding context.

Port references:

- [Julia model](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/julia/model.py)
- [ModernBERT](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/modernbert/modeling_modernbert.py)

These publisher algorithms carry their upstream Apache-2.0 attribution; this workspace's
new implementation uses its declared license. Artifact acquisition/checksums and independent
Python/ONNX parity receipts remain separate from configuration and tensor-shape checks.

The `julia_parity` example accepts an explicit model directory, completed independent fixture
document and `cpu` or `cuda`. It requires exact encoded arrays and IDs, selected-ID agreement,
logits within `1e-4 + 1e-4 × |reference|`, and probabilities within
`1e-5 + 1e-4 × |reference|`. These gates are fixed before inference. It emits a JSON receipt
and exits nonzero on failure; reported case times include first-use effects and are not a
steady-state benchmark. No model acquisition or file modification occurs.
