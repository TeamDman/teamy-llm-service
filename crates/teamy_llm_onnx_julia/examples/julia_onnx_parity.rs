//! Compare real ONNX logits against independently acquired Python reference fixtures.
//! Local artifacts only; the predeclared fixture tolerances are never adjusted here.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use eyre::{Result, ensure, eyre};
use facet::Facet;
use sha2::{Digest, Sha256};
use teamy_cancellation::CancellationToken;
use teamy_llm_core::ModelSelector;
use teamy_llm_core::finite_choice::{
    ChoiceContent, FiniteChoiceOption, FiniteChoiceRequest, FiniteChoiceResult,
};
use teamy_llm_julia_common::{
    EncodedJuliaRequest, EncodingPolicy, JuliaEncoder, JuliaQuestionType, JuliaTextOption,
    julia_capabilities,
};
use teamy_llm_onnx_julia::{JuliaOnnxOptions, JuliaOnnxProvider, OnnxAudit, OnnxDevice};

#[derive(Facet)]
struct Fixtures {
    version: u32,
    comparison_policy: Tolerances,
    cases: Vec<Case>,
}

#[derive(Clone, Facet)]
struct Tolerances {
    logit_abs_tolerance: f64,
    logit_rel_tolerance: f64,
    probability_abs_tolerance: f64,
    probability_rel_tolerance: f64,
}

#[derive(Facet)]
struct Case {
    id: String,
    state: String,
    question: String,
    #[facet(rename = "type")]
    kind: String,
    max_tokens: usize,
    head_tokens: usize,
    options: Vec<JuliaTextOption>,
    encoded: EncodedJuliaRequest,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
    selected_option_id: String,
}

#[derive(Facet)]
struct CaseReport {
    id: String,
    encoding_matches: bool,
    logit_tolerance_passed: bool,
    probability_tolerance_passed: bool,
    selected_option_matches: bool,
    max_logit_abs_error: f64,
    max_probability_abs_error: f64,
    elapsed_ms: f64,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
}

#[derive(Facet)]
struct Report {
    version: u32,
    reference_sha256: String,
    tolerances: Tolerances,
    audit: OnnxAudit,
    load_ms: f64,
    passed: bool,
    cases: Vec<CaseReport>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4 || args.len() == 5,
        "Usage: julia_onnx_parity MODEL_DIR RUNTIME_DLL FIXTURES_JSON cpu|cuda [--allow-cpu-fallback]"
    );
    let device = match args[3].as_str() {
        "cpu" => OnnxDevice::Cpu,
        "cuda" => OnnxDevice::Cuda { device_id: 0 },
        _ => return Err(eyre!("Device must be cpu or cuda")),
    };
    let mut options =
        JuliaOnnxOptions::new(PathBuf::from(&args[0]), PathBuf::from(&args[1]), device);
    options.profile_prefix = std::env::var_os("TEAMY_ORT_PROFILE_PREFIX").map(PathBuf::from);
    let profiling = options.profile_prefix.is_some();
    if let Some(flag) = args.get(4) {
        ensure!(
            flag == "--allow-cpu-fallback" && matches!(options.device, OnnxDevice::Cuda { .. }),
            "Only CUDA accepts the explicit --allow-cpu-fallback flag"
        );
        options.allow_cpu_fallback = true;
    }
    let bytes = std::fs::read(&args[2])?;
    let reference_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let fixtures: Fixtures = facet_json::from_str(std::str::from_utf8(&bytes)?)?;
    ensure!(
        fixtures.version == 1 && !fixtures.cases.is_empty(),
        "Unsupported or empty reference fixture"
    );
    let tolerances = &fixtures.comparison_policy;
    ensure!(
        [
            tolerances.logit_abs_tolerance,
            tolerances.logit_rel_tolerance,
            tolerances.probability_abs_tolerance,
            tolerances.probability_rel_tolerance
        ]
        .iter()
        .all(|value| value.is_finite() && *value > 0.0),
        "Invalid reference tolerances"
    );
    let started = Instant::now();
    let mut provider = JuliaOnnxProvider::load(options, CancellationToken::new())?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut reports = Vec::new();
    for case in fixtures.cases {
        let kind = match case.kind.as_str() {
            "choice" => JuliaQuestionType::Choice,
            "score" => JuliaQuestionType::Score,
            "noul" => JuliaQuestionType::Noul,
            _ => return Err(eyre!("Unknown reference question type")),
        };
        ensure!(
            case.max_tokens == provider.audit().max_tokens
                && case.head_tokens == provider.audit().head_tokens,
            "Fixture policy differs from the session's declared encoding policy"
        );
        let encoder = JuliaEncoder::load(
            &PathBuf::from(&args[0]),
            EncodingPolicy {
                max_tokens: case.max_tokens,
                head_tokens: case.head_tokens,
                question: case.question.clone(),
            },
        )?;
        let encoded = encoder.encode_parts(&case.state, &case.question, kind, &case.options)?;
        let encoding_matches = encoded == case.encoded;
        ensure!(
            encoding_matches,
            "Encoding differs from independent reference for {}",
            case.id
        );
        ensure!(
            case.logits.len() == case.options.len()
                && case.probabilities.len() == case.options.len()
                && case
                    .logits
                    .iter()
                    .chain(&case.probabilities)
                    .all(|n| n.is_finite()),
            "Invalid reference scores"
        );
        let started = Instant::now();
        let logits = provider.infer_encoded(&encoded)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let request = FiniteChoiceRequest {
            selector: ModelSelector::managed("julia-1-onnx-reference"),
            input: ChoiceContent::text(case.state),
            options: case
                .options
                .iter()
                .map(|option| FiniteChoiceOption {
                    id: option.id.clone(),
                    content: ChoiceContent::text(option.text.clone()),
                })
                .collect(),
        };
        let validated = request.validate_for(&julia_capabilities())?;
        let scores = FiniteChoiceResult::from_logits(&validated, logits.clone())?;
        let probabilities: Vec<f64> = scores
            .scores
            .iter()
            .map(|score| score.probability)
            .collect();
        let logit_tolerance_passed = within(
            &logits,
            &case.logits,
            tolerances.logit_abs_tolerance,
            tolerances.logit_rel_tolerance,
        );
        let probability_tolerance_passed = within(
            &probabilities,
            &case.probabilities,
            tolerances.probability_abs_tolerance,
            tolerances.probability_rel_tolerance,
        );
        let selected_option_matches = scores
            .best_option()
            .is_some_and(|score| score.option_id == case.selected_option_id);
        reports.push(CaseReport {
            id: case.id,
            encoding_matches,
            logit_tolerance_passed,
            probability_tolerance_passed,
            selected_option_matches,
            max_logit_abs_error: max_error(&logits, &case.logits),
            max_probability_abs_error: max_error(&probabilities, &case.probabilities),
            elapsed_ms,
            logits,
            probabilities,
        });
    }
    if profiling {
        let trace = provider.finish_profiling()?;
        eprintln!("ORT profile: {}", trace.display());
    }
    let passed = reports.iter().all(|case| {
        case.encoding_matches
            && case.logit_tolerance_passed
            && case.probability_tolerance_passed
            && case.selected_option_matches
    });
    let report = Report {
        version: 1,
        reference_sha256,
        tolerances: fixtures.comparison_policy,
        audit: provider.audit().clone(),
        load_ms,
        passed,
        cases: reports,
    };
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(facet_json::to_string_pretty(&report)?.as_bytes())?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    ensure!(
        passed,
        "ONNX/reference comparison failed; retain the typed report"
    );
    Ok(())
}

fn within(actual: &[f64], reference: &[f64], absolute: f64, relative: f64) -> bool {
    actual.len() == reference.len()
        && actual
            .iter()
            .zip(reference)
            .all(|(a, b)| a.is_finite() && (a - b).abs() <= absolute + relative * b.abs())
}

fn max_error(actual: &[f64], reference: &[f64]) -> f64 {
    actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}
