use teamy_llm_core::ModelSelector;
use teamy_llm_core::finite_choice::{
    BackendCapabilities, ChoiceContent, ContentCapabilities, ContentLocation,
    FiniteChoiceCapabilities, FiniteChoiceError, FiniteChoiceErrorKind, FiniteChoiceOption,
    FiniteChoiceProvider, FiniteChoiceRequest, FiniteChoiceResult, ValidatedFiniteChoiceRequest,
    VisualFormat, VisualInput, finite_choice,
};

fn capabilities() -> BackendCapabilities {
    BackendCapabilities {
        text_generation: false,
        finite_choice: Some(FiniteChoiceCapabilities::julia_1_text_only()),
    }
}

fn request() -> FiniteChoiceRequest {
    FiniteChoiceRequest {
        selector: ModelSelector::managed("synthetic"),
        input: ChoiceContent::text("Which option?"),
        options: vec![
            FiniteChoiceOption {
                id: "first".into(),
                content: ChoiceContent::text("First"),
            },
            FiniteChoiceOption {
                id: "second".into(),
                content: ChoiceContent::text("Second"),
            },
        ],
    }
}

fn visual(format: VisualFormat) -> VisualInput {
    VisualInput {
        format,
        bytes: vec![1],
    }
}

#[test]
fn julia_profile_has_text_only_bounds_without_generation() {
    let capabilities = capabilities();
    assert!(!capabilities.text_generation);
    let profile = capabilities.finite_choice.as_ref().unwrap();
    assert_eq!((profile.min_options, profile.max_options), (2, 20));
    assert!(profile.input.visual_formats.is_empty());
    assert!(profile.options.visual_formats.is_empty());
    let mut request = request();
    assert!(request.validate_for(&capabilities).is_ok());
    request.options.truncate(1);
    assert_eq!(
        request.validate_for(&capabilities).unwrap_err().kind,
        FiniteChoiceErrorKind::InvalidRequest
    );
    request.options = (0..21)
        .map(|index| FiniteChoiceOption {
            id: index.to_string(),
            content: ChoiceContent::text("Option"),
        })
        .collect();
    assert_eq!(
        request.validate_for(&capabilities).unwrap_err().kind,
        FiniteChoiceErrorKind::InvalidRequest
    );
    request.options.truncate(20);
    assert!(request.validate_for(&capabilities).is_ok());
}

#[test]
fn rejects_empty_content_ids_and_duplicate_ids() {
    let capabilities = capabilities();
    for text in [None, Some("   ".into())] {
        let mut request = request();
        request.input.text = text;
        let error = request.validate_for(&capabilities).unwrap_err();
        assert_eq!(error.kind, FiniteChoiceErrorKind::InvalidRequest);
        assert_eq!(error.location, Some(ContentLocation::Input));
    }
    for id in [" ", "first"] {
        let mut request = request();
        request.options[1].id = id.into();
        let error = request.validate_for(&capabilities).unwrap_err();
        assert_eq!(error.kind, FiniteChoiceErrorKind::InvalidRequest);
        assert_eq!(error.location, Some(ContentLocation::Option { index: 1 }));
    }
    let mut request = request();
    request.options[0].content.text = None;
    assert_eq!(
        request.validate_for(&capabilities).unwrap_err().location,
        Some(ContentLocation::Option { index: 0 })
    );
}

#[test]
fn input_and_option_vision_are_independent_and_never_dropped() {
    let mut capabilities = capabilities();
    let profile = capabilities.finite_choice.as_mut().unwrap();
    profile.input.visual_formats.push(VisualFormat::Png);
    let mut request = request();
    request.input.visuals.push(visual(VisualFormat::Png));
    assert!(request.validate_for(&capabilities).is_ok());
    request.options[1]
        .content
        .visuals
        .push(visual(VisualFormat::Png));
    let error = request.validate_for(&capabilities).unwrap_err();
    assert_eq!(error.kind, FiniteChoiceErrorKind::UnsupportedVisual);
    assert_eq!(error.location, Some(ContentLocation::Option { index: 1 }));
    capabilities
        .finite_choice
        .as_mut()
        .unwrap()
        .options
        .visual_formats
        .push(VisualFormat::Png);
    let validated = request.validate_for(&capabilities).unwrap();
    assert_eq!(validated.request().input.visuals[0].bytes, vec![1]);
    assert_eq!(
        validated.request().options[1].content.visuals[0].bytes,
        vec![1]
    );
    request.input.visuals[0].format = VisualFormat::Jpeg;
    assert_eq!(
        request.validate_for(&capabilities).unwrap_err().kind,
        FiniteChoiceErrorKind::UnsupportedVisual
    );
    request.input.visuals[0].format = VisualFormat::Png;
    request.input.visuals[0].bytes.clear();
    assert_eq!(
        request.validate_for(&capabilities).unwrap_err().kind,
        FiniteChoiceErrorKind::InvalidRequest
    );
}

#[test]
fn visual_only_content_requires_supported_text_when_text_is_present() {
    let mut capabilities = capabilities();
    capabilities.finite_choice.as_mut().unwrap().input = ContentCapabilities {
        text: false,
        visual_formats: vec![VisualFormat::Png],
    };
    let mut request = request();
    request.input.visuals.push(visual(VisualFormat::Png));
    assert_eq!(
        request.validate_for(&capabilities).unwrap_err().kind,
        FiniteChoiceErrorKind::UnsupportedText
    );
    request.input.text = None;
    assert!(request.validate_for(&capabilities).is_ok());
}

#[test]
fn generation_only_and_malformed_capability_profiles_are_rejected() {
    let request = request();
    let generation_only = BackendCapabilities {
        text_generation: true,
        finite_choice: None,
    };
    assert_eq!(
        request.validate_for(&generation_only).unwrap_err().kind,
        FiniteChoiceErrorKind::UnsupportedTask
    );
    let mut invalid = Vec::new();
    let mut low_minimum = capabilities();
    low_minimum.finite_choice.as_mut().unwrap().min_options = 1;
    invalid.push(low_minimum);
    let mut reversed = capabilities();
    reversed.finite_choice.as_mut().unwrap().max_options = 1;
    invalid.push(reversed);
    let mut no_modality = capabilities();
    no_modality.finite_choice.as_mut().unwrap().input.text = false;
    invalid.push(no_modality);
    let mut duplicate_formats = capabilities();
    duplicate_formats
        .finite_choice
        .as_mut()
        .unwrap()
        .input
        .visual_formats = vec![VisualFormat::Png, VisualFormat::Png];
    invalid.push(duplicate_formats);
    for invalid in invalid {
        assert_eq!(
            request.validate_for(&invalid).unwrap_err().kind,
            FiniteChoiceErrorKind::InvalidCapabilities
        );
    }
}

#[test]
fn stable_softmax_preserves_order_and_first_tie() {
    let request = request();
    let validated = request.validate_for(&capabilities()).unwrap();
    for logits in [
        vec![1_000.0, 1_001.0],
        vec![-1_000.0, -999.0],
        vec![-f64::MAX, f64::MAX],
    ] {
        let result = FiniteChoiceResult::from_logits(&validated, logits).unwrap();
        result.validate(&validated).unwrap();
        assert_eq!(result.scores[0].option_id, "first");
        assert_eq!(result.scores[1].option_id, "second");
        assert_eq!(result.best_option().unwrap().option_id, "second");
        assert!(
            result
                .scores
                .iter()
                .all(|score| score.probability.is_finite())
        );
    }
    let tied = FiniteChoiceResult::from_logits(&validated, vec![4.0, 4.0]).unwrap();
    assert_eq!(tied.best_option().unwrap().option_id, "first");
    assert_eq!(tied.scores[0].probability, 0.5);
    for logits in [
        vec![],
        vec![1.0],
        vec![f64::NAN, 1.0],
        vec![1.0, f64::INFINITY],
        vec![f64::NEG_INFINITY, 1.0],
    ] {
        assert_eq!(
            FiniteChoiceResult::from_logits(&validated, logits)
                .unwrap_err()
                .kind,
            FiniteChoiceErrorKind::InvalidResult
        );
    }
}

#[test]
fn rejects_malformed_provider_scores_and_inconsistent_probabilities() {
    let request = request();
    let validated = request.validate_for(&capabilities()).unwrap();
    let valid = FiniteChoiceResult::from_logits(&validated, vec![0.0, 1.0]).unwrap();
    let mut invalid = Vec::new();
    let mut missing = valid.clone();
    missing.scores.pop();
    invalid.push(missing);
    let mut reordered = valid.clone();
    reordered.scores.swap(0, 1);
    invalid.push(reordered);
    let mut wrong_id = valid.clone();
    wrong_id.scores[1].option_id = "unknown".into();
    invalid.push(wrong_id);
    for probability in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let mut bad_probability = valid.clone();
        bad_probability.scores[0].probability = probability;
        invalid.push(bad_probability);
    }
    let mut non_normalized = valid.clone();
    non_normalized.scores[0].probability = 0.1;
    non_normalized.scores[1].probability = 0.1;
    invalid.push(non_normalized);
    let mut inconsistent = valid.clone();
    inconsistent.scores[0].probability = 0.5;
    inconsistent.scores[1].probability = 0.5;
    invalid.push(inconsistent);
    let mut bad_logit = valid.clone();
    bad_logit.scores[0].logit = f64::NAN;
    invalid.push(bad_logit);
    for invalid in invalid {
        assert_eq!(
            invalid.validate(&validated).unwrap_err().kind,
            FiniteChoiceErrorKind::InvalidResult
        );
    }
}

struct FixtureProvider {
    calls: usize,
    malformed: bool,
    fail: bool,
}

impl FiniteChoiceProvider for FixtureProvider {
    fn capabilities(&self) -> BackendCapabilities {
        capabilities()
    }

    fn infer(
        &mut self,
        request: &ValidatedFiniteChoiceRequest<'_>,
    ) -> Result<FiniteChoiceResult, FiniteChoiceError> {
        self.calls += 1;
        if self.fail {
            return Err(FiniteChoiceError::new(
                FiniteChoiceErrorKind::ProviderFailure,
                "Synthetic failure",
            ));
        }
        let mut result = FiniteChoiceResult::from_logits(request, vec![0.0, 1.0])?;
        if self.malformed {
            result.scores.reverse();
        }
        Ok(result)
    }
}

#[test]
fn checked_entrypoint_rejects_vision_before_execution_and_validates_provider_output() {
    let mut provider = FixtureProvider {
        calls: 0,
        malformed: false,
        fail: false,
    };
    let mut request = request();
    finite_choice(&mut provider, &request).unwrap();
    assert_eq!(provider.calls, 1);
    request.input.visuals.push(visual(VisualFormat::Png));
    assert_eq!(
        finite_choice(&mut provider, &request).unwrap_err().kind,
        FiniteChoiceErrorKind::UnsupportedVisual
    );
    assert_eq!(provider.calls, 1);
    request.input.visuals.clear();
    request.options[0]
        .content
        .visuals
        .push(visual(VisualFormat::Png));
    assert_eq!(
        finite_choice(&mut provider, &request).unwrap_err().location,
        Some(ContentLocation::Option { index: 0 })
    );
    assert_eq!(provider.calls, 1);
    request.options[0].content.visuals.clear();
    provider.malformed = true;
    assert_eq!(
        finite_choice(&mut provider, &request).unwrap_err().kind,
        FiniteChoiceErrorKind::InvalidResult
    );
    assert_eq!(provider.calls, 2);
    provider.fail = true;
    let error = finite_choice(&mut provider, &request).unwrap_err();
    assert_eq!(error.kind, FiniteChoiceErrorKind::ProviderFailure);
    assert_eq!(error.message, "Synthetic failure");
    assert_eq!(provider.calls, 3);
}
