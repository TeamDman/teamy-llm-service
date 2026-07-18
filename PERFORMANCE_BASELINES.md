# Performance baselines

This log tracks release-build generation measurements for later comparison and plotting. Append one
row for every completed benchmark run; do not overwrite earlier measurements.

| Datetime (America/Toronto) | Mode | TTFT (s) | Completion (s) | Tokens/s | Comment |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-07-13T23:08:53-04:00 | direct | 43.263 | 84.837 | 4.811 | Release build, process-cold, `Why is the sky blue?`, 200-token cap, commit `5605696`. Timestamp recorded when this log was created immediately after the measurement. |
| 2026-07-13T23:08:53-04:00 | thinking | 40.846 | 82.476 | 4.804 | Release build, process-cold, `Why is the sky blue?`, 200-token cap, commit `5605696`. Timestamp recorded when this log was created immediately after the measurement. |
| 2026-07-14T00:01:53-04:00 | direct | 39.541 | 88.466 | 4.088 | Release build, process-cold, `Why is the sky blue?`, 200-token cap. CUDA lm-head argmax experiment based on `2e4b4cb`; token oracle matched but the implementation was reverted after this regression. Timestamp logged shortly after the run. |
| 2026-07-14T00:01:53-04:00 | thinking | 42.133 | 89.392 | 4.232 | Release build, process-cold, `Why is the sky blue?`, 200-token cap. CUDA lm-head argmax experiment based on `2e4b4cb`; token oracle matched but the implementation was reverted after this regression. Timestamp logged shortly after the run. |
| 2026-07-18T12:54:52-04:00 | direct | 29.873 | 62.770 | 6.079 | Release build, process-cold, `Why is the sky blue?`, 200-token cap (ended at cap), model `qwopus-3.5-9b-coder-q4-k-m`. CUDA-resident hidden-state bridge working tree based on `71faafd`; 16-token CUDA oracle matched before this run. |
| 2026-07-18T12:54:52-04:00 | thinking | 29.672 | 63.826 | 5.856 | Release build, process-cold, `Why is the sky blue?`, 200-token cap (ended at cap), model `qwopus-3.5-9b-coder-q4-k-m`. CUDA-resident hidden-state bridge working tree based on `71faafd`; 16-token CUDA oracle matched before this run. |
| 2026-07-18T13:32:35-04:00 | direct | 29.100 | 55.854 | 7.475 | Release build, process-cold, `Why is the sky blue?`, 200-token cap (ended at cap), model `qwopus-3.5-9b-coder-q4-k-m`. Fused linear-attention input-projection working tree based on `9e7f017`; 16-token CUDA oracle matched before this run. |
| 2026-07-18T13:32:35-04:00 | thinking | 36.257 | 62.829 | 7.527 | Release build, process-cold, `Why is the sky blue?`, 200-token cap (ended at cap), model `qwopus-3.5-9b-coder-q4-k-m`. Fused linear-attention input-projection working tree based on `9e7f017`; 16-token CUDA oracle matched before this run. |
| 2026-07-18T15:25:44-04:00 | direct | 36.437 | 62.653 | 7.629 | Release build, process-cold, `Why is the sky blue?`, 200-token cap (ended at cap), model `qwopus-3.5-9b-coder-q4-k-m`. Device-hidden lm-head handoff working tree based on `9161878`; 16-token CUDA oracle matched before this run. Timestamp logged after the paired runs. |
| 2026-07-18T15:25:44-04:00 | thinking | 27.405 | 53.126 | 7.776 | Release build, process-cold, `Why is the sky blue?`, 200-token cap (ended at cap), model `qwopus-3.5-9b-coder-q4-k-m`. Device-hidden lm-head handoff working tree based on `9161878`; 16-token CUDA oracle matched before this run. Timestamp logged after the paired runs. |

## Measurement conventions

- Run a release build with a fixed prompt and token cap, for example:
  `cargo run --release -p teamy_llm_cli -- benchmark --no-thinking --max-new-tokens 200 --timeout-ms 600000`
  or replace `--no-thinking` with `--thinking`.
- TTFT starts when the generation request is submitted. It includes CUDA validation, model
  initialization, prompt processing, and the first decode step.
- Completion is the time from request submission to the final generated token. Record whether the
  run ended at the token cap or EOS in the comment.
- Tokens/s is post-first-token decode throughput: generated token count divided by the duration
  from the first token to completion.
- Put the Git commit, build profile, model, prompt/token cap, cache state, and any material
  environmental change in the comment. Use ISO 8601 timestamps with the UTC offset.
