# Julia independent Python reference

This harness uses the pinned publisher's `JuliaDecisionModel`, `sequence` and `Collator`. It produces fixtures independently of Rust encoding or numerical implementations. It deliberately bypasses Julia's optional router, Bend bridge, compilation and INT8 paths.

The frozen Linux AMD64 environment uses Python 3.12.15, uv 0.11.26, Torch 2.11.0+cu128 and Transformers 5.0.0. Julia's upstream project requires Transformers `>=5.0,<5.1`, so the existing Qwen reference's Transformers 5.9.0 is not reused. The image digests are in `Containerfile`; `uv.lock` retains exact transitive packages and artifact checksums.

The copied `vendor/julia` source comes from SupersonicLabs/Julia-1 revision `a85b127321d580d65176c89ced8273f305745d85`, declared Apache-2.0 by its model card. See `vendor/SOURCE.json` and the retained publisher README. Native weights and the ONNX revision `82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8` are acquired separately, never committed with this harness.

Inputs are text-only and preserve state, question, choice type, ordered option IDs/descriptions and limits. Defaults are 1024 context tokens, 256 head tokens and `Which option should be chosen?`. Publisher strict encoding rejects reserved markers, lossy option/question budgets and truncated context. The publisher collator pads to the next multiple of eight.

The reference uses float32, math SDPA, no autocast or TF32, seed 42 and deterministic algorithm checks. Golden outputs include exact token/mask/marker arrays, selected intermediate marker vectors, logits, unit-temperature softmax probabilities, input digests, timings and environment provenance. These scores are not calibrated confidence. Fixtures are never overwritten.

Run only in the reviewed image with read-only artifacts, no network, a non-root user and a dedicated output mount. The model, tokenizer and vendored source checksums are verified before inference. Missing requested CUDA fails; an explicitly selected CPU run is a distinct reference, not silent fallback. Container isolation does not remove native-code or host-driver risks.

From the harness source directory, the standard-library contract tests validate ordered IDs, explicit defaults, rejected unsupported fields and artifact checksum failures without inference. Run them with the pinned image's Python and the harness mounted read-only:

```sh
python -m unittest test_reference.py
```

Example inside the container:

```sh
python /app/reference.py --model-dir /artifacts/native \
  --artifact-manifest /inputs/artifact-manifest.json \
  --output /outputs/fixtures.json --device cuda:0 \
  --image-id sha256:<recorded-built-image-id>
```
