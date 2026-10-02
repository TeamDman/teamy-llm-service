//! Synthetic contract demonstration: no Julia model, weights, image decoder, GPU or network.

use teamy_llm_core::ModelSelector;
use teamy_llm_core::finite_choice::{
    BackendCapabilities, ChoiceContent, FiniteChoiceCapabilities, FiniteChoiceError,
    FiniteChoiceErrorKind, FiniteChoiceOption, FiniteChoiceProvider, FiniteChoiceRequest,
    FiniteChoiceResult, ValidatedFiniteChoiceRequest, VisualFormat, VisualInput, finite_choice,
};

/// Deliberately fixed scores; this tests the interface, not model quality or inference correctness.
#[derive(Default)]
struct SyntheticProvider {
    calls: usize,
}

impl FiniteChoiceProvider for SyntheticProvider {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            text_generation: false,
            finite_choice: Some(FiniteChoiceCapabilities::julia_1_text_only()),
        }
    }

    fn infer(
        &mut self,
        request: &ValidatedFiniteChoiceRequest<'_>,
    ) -> Result<FiniteChoiceResult, FiniteChoiceError> {
        self.calls += 1;
        // Fixture values are in the exact order of the two options below.
        FiniteChoiceResult::from_logits(request, vec![1_000.0, 1_001.0])
    }
}

fn main() -> Result<(), FiniteChoiceError> {
    let mut provider = SyntheticProvider::default();
    let request = FiniteChoiceRequest {
        selector: ModelSelector::managed("synthetic-fixture"),
        input: ChoiceContent::text("Choose a category for this fixture."),
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
    let result = finite_choice(&mut provider, &request)?;
    assert_eq!(provider.calls, 1);
    assert_eq!(result.scores[0].option_id, "writing");
    assert_eq!(result.scores[1].option_id, "playing");
    assert_eq!(result.best_option().unwrap().option_id, "playing");
    assert!(
        result
            .scores
            .iter()
            .all(|score| score.probability.is_finite())
    );

    let mut visual_request = request.clone();
    visual_request.input.visuals.push(VisualInput {
        format: VisualFormat::Png,
        // Placeholder bytes are deliberately not an image; decoding is outside this contract.
        bytes: vec![1],
    });
    let error = finite_choice(&mut provider, &visual_request).unwrap_err();
    assert_eq!(error.kind, FiniteChoiceErrorKind::UnsupportedVisual);
    assert_eq!(
        provider.calls, 1,
        "unsupported vision never reaches the provider"
    );
    println!(
        "Synthetic contract passed: ordered scores, stable softmax, unsupported vision rejected."
    );
    Ok(())
}
