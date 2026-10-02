# Julia encoding contract

Both providers consume one batch-one `EncodedJuliaRequest`. The core projection treats
`input.text` as state, uses choice type, and takes the question from `EncodingPolicy`.
The default question is `Which option should be chosen?`; callers can replace it explicitly.
Loading accepts the publisher's native `tokenizer/tokenizer.json` or ONNX `tokenizer.json`
layout. A directory containing both is ambiguous and fails. Tokenizer metadata cannot redirect
the loader to an arbitrary file, and PAD/SEP/CLS/MASK identities must match the Julia contract.

Limits default to 1024 sequence tokens and 256 question/option tokens. Requests must fit
losslessly; option descriptions are limited to 48 tokens. Reserved marker/special tokens,
unsupported images and invalid IDs fail. Padding follows the publisher's eight-token collator.
Literal added special tokens also fail; natural text tokenized to the unknown-token ID remains
valid. This is stricter than the publisher's mask-only literal check and avoids ambiguous
special-token text entering the fixed sequence grammar.

`encode_parts` preserves distinct state/question/type for independent reference fixtures.
It does not infer or acquire model files. Original 8192/512 limits require an explicit policy;
accepting a longer context does not establish task accuracy or a safe memory budget.

Encoding references:

- [Julia source](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/julia/data.py)
- [Publisher Rust encoder](https://huggingface.co/SupersonicLabs/Julia-1-ONNX/blob/82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8/rust/src/lib.rs)

The local implementation adapts that documented algorithm to the existing Facet contracts;
it does not embed the publisher's Python runtime.
