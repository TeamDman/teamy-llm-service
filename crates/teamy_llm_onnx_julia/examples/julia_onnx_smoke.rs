//! Real local-model smoke check. The caller supplies verified local model/runtime artifacts.
//! No acquisition, network, Python execution or candidate actions occur.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use eyre::{Result, ensure, eyre};
use teamy_cancellation::CancellationToken;
use teamy_llm_core::ModelSelector;
use teamy_llm_core::finite_choice::{
    ChoiceContent, FiniteChoiceOption, FiniteChoiceRequest, finite_choice,
};
use teamy_llm_onnx_julia::{JuliaOnnxOptions, JuliaOnnxProvider, OnnxDevice};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3 || args.len() == 4,
        "Usage: julia_onnx_smoke MODEL_DIR RUNTIME_DLL cpu|cuda [--allow-cpu-fallback]"
    );
    let device = match args[2].as_str() {
        "cpu" => OnnxDevice::Cpu,
        "cuda" => OnnxDevice::Cuda { device_id: 0 },
        _ => return Err(eyre!("Device must be cpu or cuda")),
    };
    let mut options =
        JuliaOnnxOptions::new(PathBuf::from(&args[0]), PathBuf::from(&args[1]), device);
    if let Some(flag) = args.get(3) {
        ensure!(
            flag == "--allow-cpu-fallback" && matches!(options.device, OnnxDevice::Cuda { .. }),
            "Only CUDA accepts the explicit --allow-cpu-fallback flag"
        );
        options.allow_cpu_fallback = true;
    }
    let started = Instant::now();
    let mut provider = JuliaOnnxProvider::load(options, CancellationToken::new())?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Runtime/provider audit: {:?}", provider.audit());
    let request = FiniteChoiceRequest {
        selector: ModelSelector::managed("julia-1-onnx"),
        input: ChoiceContent::text("A repository contains Rust source code and a Cargo.toml file."),
        options: vec![
            FiniteChoiceOption {
                id: "writing".into(),
                content: ChoiceContent::text("Writing programs"),
            },
            FiniteChoiceOption {
                id: "playing".into(),
                content: ChoiceContent::text("Playing games"),
            },
        ],
    };
    let started = Instant::now();
    let result = finite_choice(&mut provider, &request)?;
    let first_request_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let repeated = finite_choice(&mut provider, &request)?;
    let warm_request_ms = started.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        result
            .scores
            .iter()
            .map(|score| score.option_id.as_str())
            .eq(["writing", "playing"]),
        "Option IDs or order changed"
    );
    ensure!(
        repeated
            .scores
            .iter()
            .zip(&result.scores)
            .all(|(a, b)| (a.logit - b.logit).abs() <= 1e-5),
        "Repeated inference changed logits"
    );
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "{{\"version\":1,\"load_ms\":{load_ms},\"first_request_ms\":{first_request_ms},\"warm_request_ms\":{warm_request_ms},\"scores\":[{{\"option_id\":\"writing\",\"logit\":{},\"probability\":{}}},{{\"option_id\":\"playing\",\"logit\":{},\"probability\":{}}}]}}",
        result.scores[0].logit,
        result.scores[0].probability,
        result.scores[1].logit,
        result.scores[1].probability
    )?;
    stdout.flush()?;
    Ok(())
}
