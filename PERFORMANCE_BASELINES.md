# Performance baselines

This log tracks release-build generation measurements for later comparison and plotting. Append one
row for every completed benchmark run; do not overwrite earlier measurements.

| Datetime (America/Toronto) | Mode | TTFT (s) | Completion (s) | Tokens/s | Comment |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-07-13T23:08:53-04:00 | direct | 43.263 | 84.837 | 4.811 | Release build, process-cold, `Why is the sky blue?`, 200-token cap, commit `5605696`. Timestamp recorded when this log was created immediately after the measurement. |
| 2026-07-13T23:08:53-04:00 | thinking | 40.846 | 82.476 | 4.804 | Release build, process-cold, `Why is the sky blue?`, 200-token cap, commit `5605696`. Timestamp recorded when this log was created immediately after the measurement. |

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
