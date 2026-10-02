//! Compare real native inference with independently generated publisher fixtures.
//! Usage: julia_parity <model-directory> <fixtures.json> <cpu|cuda>
//! Model files are selected explicitly; this example neither acquires nor modifies them.

use std::{path::Path, time::Instant};

use eyre::{ensure, eyre};
use facet::Facet;
use teamy_cancellation::CtrlCHandler;
use teamy_llm_burn_julia::{NativeDevice, NativeJuliaMetadata, NativeJuliaProvider};
use teamy_llm_julia_common::{
    EncodedJuliaRequest, EncodingPolicy, JuliaEncoder, JuliaQuestionType, JuliaTextOption,
};

#[derive(Facet)]
struct FixtureDocument {
    version: u32,
    status: String,
    reference: String,
    comparison_policy: ComparisonPolicy,
    cases: Vec<FixtureCase>,
}

#[derive(Clone, Facet)]
struct ComparisonPolicy {
    encoding: String,
    logit_abs_tolerance: f64,
    logit_rel_tolerance: f64,
    probability_abs_tolerance: f64,
    probability_rel_tolerance: f64,
}

#[derive(Facet)]
struct FixtureCase {
    id: String,
    state: String,
    question: String,
    #[facet(rename = "type")]
    question_type: String,
    max_tokens: usize,
    head_tokens: usize,
    options: Vec<JuliaTextOption>,
    encoded: EncodedJuliaRequest,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
    selected_option_id: String,
}

#[derive(Facet)]
struct ParityReport {
    version: u32,
    passed: bool,
    reference: String,
    metadata: NativeJuliaMetadata,
    comparison_policy: ComparisonPolicy,
    cases: Vec<CaseReport>,
}

#[derive(Facet)]
struct CaseReport {
    id: String,
    passed: bool,
    encoding_exact: bool,
    selected_option_id: String,
    selected_id_exact: bool,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
    max_logit_abs_error: f64,
    max_probability_abs_error: f64,
    inference_millis: f64,
}

fn main() -> eyre::Result<()> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        arguments.len() == 3,
        "Usage: julia_parity <model-directory> <fixtures.json> <cpu|cuda>"
    );
    let device = match arguments[2].to_str() {
        Some("cpu") => NativeDevice::Cpu,
        Some("cuda") => NativeDevice::Cuda,
        _ => return Err(eyre!("The device must explicitly be cpu or cuda")),
    };
    let model_dir = Path::new(&arguments[0]);
    let fixture_path = Path::new(&arguments[1]);
    ensure!(
        std::fs::metadata(fixture_path)?.len() <= 16 * 1024 * 1024,
        "Fixture payload exceeds 16 MiB"
    );
    let document: FixtureDocument = facet_json::from_str(&std::fs::read_to_string(fixture_path)?)?;
    ensure!(
        document.version == 1
            && document.status == "completed"
            && document.reference == "pinned-publisher-python-independent-of-rust"
            && !document.cases.is_empty()
            && document.cases.len() <= 100,
        "Unsupported or incomplete fixture document"
    );
    let tolerance = &document.comparison_policy;
    // Predeclared before inference. Changing a fixture cannot quietly loosen this gate.
    ensure!(
        tolerance.encoding == "exact"
            && tolerance.logit_abs_tolerance == 1e-4
            && tolerance.logit_rel_tolerance == 1e-4
            && tolerance.probability_abs_tolerance == 1e-5
            && tolerance.probability_rel_tolerance == 1e-4,
        "Fixture acceptance policy differs from the frozen parity gate"
    );
    let first = &document.cases[0];
    let cancellation = CtrlCHandler::default().install()?;
    let mut provider = NativeJuliaProvider::load(
        model_dir,
        device,
        EncodingPolicy {
            max_tokens: first.max_tokens,
            head_tokens: first.head_tokens,
            question: first.question.clone(),
        },
        cancellation.clone(),
    )?;
    let mut cases = Vec::new();
    for case in &document.cases {
        cancellation.bail_if_cancelled()?;
        ensure!(
            case.question_type == "choice",
            "This fixture runner expects choice cases"
        );
        let encoder = JuliaEncoder::load(
            model_dir,
            EncodingPolicy {
                max_tokens: case.max_tokens,
                head_tokens: case.head_tokens,
                question: case.question.clone(),
            },
        )?;
        let encoded = encoder.encode_parts(
            &case.state,
            &case.question,
            JuliaQuestionType::Choice,
            &case.options,
        )?;
        let encoding_exact = encoded == case.encoded;
        ensure!(
            encoding_exact,
            "Shared encoding differs from the independent reference for {}",
            case.id
        );
        ensure!(
            case.logits.len() == encoded.option_count()
                && case.probabilities.len() == encoded.option_count(),
            "Fixture result length differs from option count"
        );
        ensure!(
            case.logits
                .iter()
                .chain(&case.probabilities)
                .all(|value| value.is_finite()),
            "Nonfinite fixture values"
        );
        let started = Instant::now();
        let logits = provider.infer_encoded(&encoded)?;
        let inference_millis = started.elapsed().as_secs_f64() * 1000.0;
        let probabilities = softmax(&logits);
        let mut best = 0;
        for index in 1..logits.len() {
            if logits[index] > logits[best] {
                best = index;
            }
        }
        let selected_option_id = encoded.option_ids[best].clone();
        let selected_id_exact = selected_option_id == case.selected_option_id;
        let logits_ok = within(
            &logits,
            &case.logits,
            tolerance.logit_abs_tolerance,
            tolerance.logit_rel_tolerance,
        );
        let probabilities_ok = within(
            &probabilities,
            &case.probabilities,
            tolerance.probability_abs_tolerance,
            tolerance.probability_rel_tolerance,
        );
        cases.push(CaseReport {
            id: case.id.clone(),
            passed: encoding_exact && selected_id_exact && logits_ok && probabilities_ok,
            encoding_exact,
            selected_option_id,
            selected_id_exact,
            max_logit_abs_error: max_error(&logits, &case.logits),
            max_probability_abs_error: max_error(&probabilities, &case.probabilities),
            logits,
            probabilities,
            inference_millis,
        });
    }
    let report = ParityReport {
        version: 1,
        passed: cases.iter().all(|case| case.passed),
        reference: document.reference,
        metadata: provider.metadata().clone(),
        comparison_policy: document.comparison_policy,
        cases,
    };
    println!("{}", facet_json::to_string_pretty(&report)?);
    if !report.passed {
        std::process::exit(2);
    }
    Ok(())
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponents = values
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let total = exponents.iter().sum::<f64>();
    exponents.into_iter().map(|value| value / total).collect()
}

fn within(actual: &[f64], expected: &[f64], absolute: f64, relative: f64) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            (actual - expected).abs() <= absolute + relative * expected.abs()
        })
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}
