//! Typed finite-choice inference contracts, independent of model execution and transports.
//!
//! [`finite_choice`] validates capabilities and input before invoking a provider, then checks its
//! output. Visual bytes are carried in memory; this module never reads files, fetches URLs, decodes
//! images or loads models. A provider remains responsible for decoding and interpreting its inputs.
//! See `examples/finite_choice_contract.rs` for an executable, deliberately synthetic provider.

use arbitrary::Arbitrary;
use facet::Facet;

use crate::ModelSelector;

/// Supported encoded image containers. Declaring a format does not prove that bytes decode.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub enum VisualFormat {
    Png,
    Jpeg,
    Webp,
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct VisualInput {
    pub format: VisualFormat,
    pub bytes: Vec<u8>,
}

/// Content of either the query or one candidate option. At least one modality must be present.
#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct ChoiceContent {
    pub text: Option<String>,
    pub visuals: Vec<VisualInput>,
}

impl ChoiceContent {
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            visuals: Vec::new(),
        }
    }
}

/// IDs must be nonempty and unique within one request. Results preserve this exact option order.
#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct FiniteChoiceOption {
    pub id: String,
    pub content: ChoiceContent,
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct FiniteChoiceRequest {
    pub selector: ModelSelector,
    pub input: ChoiceContent,
    pub options: Vec<FiniteChoiceOption>,
}

/// An empty `visual_formats` list explicitly means that visual input is unsupported.
#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct ContentCapabilities {
    pub text: bool,
    pub visual_formats: Vec<VisualFormat>,
}

impl ContentCapabilities {
    #[must_use]
    pub fn text_only() -> Self {
        Self {
            text: true,
            visual_formats: Vec::new(),
        }
    }

    fn validate(&self) -> Result<(), FiniteChoiceError> {
        if !self.text && self.visual_formats.is_empty() {
            return Err(FiniteChoiceError::new(
                FiniteChoiceErrorKind::InvalidCapabilities,
                "Content capabilities must support at least one modality",
            ));
        }
        for (index, format) in self.visual_formats.iter().enumerate() {
            if self.visual_formats[..index].contains(format) {
                return Err(FiniteChoiceError::new(
                    FiniteChoiceErrorKind::InvalidCapabilities,
                    "Visual formats must not contain duplicates",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct FiniteChoiceCapabilities {
    pub min_options: usize,
    pub max_options: usize,
    pub input: ContentCapabilities,
    pub options: ContentCapabilities,
}

impl FiniteChoiceCapabilities {
    /// Julia-1's declared text-only, 2–20 option contract; this does not load or implement Julia.
    #[must_use]
    pub fn julia_1_text_only() -> Self {
        Self {
            min_options: 2,
            max_options: 20,
            input: ContentCapabilities::text_only(),
            options: ContentCapabilities::text_only(),
        }
    }

    fn validate(&self) -> Result<(), FiniteChoiceError> {
        if self.min_options < 2 || self.max_options < self.min_options {
            return Err(FiniteChoiceError::new(
                FiniteChoiceErrorKind::InvalidCapabilities,
                "Finite-choice option bounds must satisfy 2 <= min_options <= max_options",
            ));
        }
        self.input.validate()?;
        self.options.validate()
    }
}

/// Finite-choice support does not imply generation support, or the reverse.
///
/// These are provider declarations, not an implementation of either inference task or permission
/// to execute one. Input and option vision are declared independently by the finite-choice profile.
#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct BackendCapabilities {
    pub text_generation: bool,
    pub finite_choice: Option<FiniteChoiceCapabilities>,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub enum ContentLocation {
    Input,
    Option { index: usize },
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub enum FiniteChoiceErrorKind {
    InvalidCapabilities,
    InvalidRequest,
    UnsupportedTask,
    UnsupportedText,
    UnsupportedVisual,
    InvalidResult,
    ProviderFailure,
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct FiniteChoiceError {
    pub kind: FiniteChoiceErrorKind,
    pub location: Option<ContentLocation>,
    pub message: String,
}

impl FiniteChoiceError {
    #[must_use]
    pub fn new(kind: FiniteChoiceErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            location: None,
            message: message.into(),
        }
    }

    fn at(mut self, location: ContentLocation) -> Self {
        self.location = Some(location);
        self
    }
}

impl std::fmt::Display for FiniteChoiceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)?;
        if let Some(location) = self.location {
            write!(formatter, " ({location:?})")?;
        }
        Ok(())
    }
}

impl std::error::Error for FiniteChoiceError {}

/// A borrowed request that passed validation for a provider's declared capabilities.
///
/// Its private field prevents callers from constructing this wrapper without validation. This
/// proves structural compatibility only, not decoded image validity, model readiness or authority.
#[derive(Debug)]
pub struct ValidatedFiniteChoiceRequest<'a> {
    request: &'a FiniteChoiceRequest,
}

impl ValidatedFiniteChoiceRequest<'_> {
    #[must_use]
    pub fn request(&self) -> &FiniteChoiceRequest {
        self.request
    }
}

impl FiniteChoiceRequest {
    pub fn validate_for(
        &self,
        capabilities: &BackendCapabilities,
    ) -> Result<ValidatedFiniteChoiceRequest<'_>, FiniteChoiceError> {
        let Some(profile) = &capabilities.finite_choice else {
            return Err(FiniteChoiceError::new(
                FiniteChoiceErrorKind::UnsupportedTask,
                "This backend does not declare finite-choice inference",
            ));
        };
        profile.validate()?;
        if !(profile.min_options..=profile.max_options).contains(&self.options.len()) {
            return Err(FiniteChoiceError::new(
                FiniteChoiceErrorKind::InvalidRequest,
                format!(
                    "Expected {}..={} options, received {}",
                    profile.min_options,
                    profile.max_options,
                    self.options.len()
                ),
            ));
        }
        validate_content(&self.input, &profile.input, ContentLocation::Input)?;
        for (index, option) in self.options.iter().enumerate() {
            let location = ContentLocation::Option { index };
            if option.id.trim().is_empty() {
                return Err(FiniteChoiceError::new(
                    FiniteChoiceErrorKind::InvalidRequest,
                    "Option IDs must be nonempty",
                )
                .at(location));
            }
            if self.options[..index]
                .iter()
                .any(|previous| previous.id == option.id)
            {
                return Err(FiniteChoiceError::new(
                    FiniteChoiceErrorKind::InvalidRequest,
                    "Option IDs must be unique",
                )
                .at(location));
            }
            validate_content(&option.content, &profile.options, location)?;
        }
        Ok(ValidatedFiniteChoiceRequest { request: self })
    }
}

fn validate_content(
    content: &ChoiceContent,
    capabilities: &ContentCapabilities,
    location: ContentLocation,
) -> Result<(), FiniteChoiceError> {
    if content.text.is_none() && content.visuals.is_empty() {
        return Err(FiniteChoiceError::new(
            FiniteChoiceErrorKind::InvalidRequest,
            "Content must contain text, visuals, or both",
        )
        .at(location));
    }
    if let Some(text) = &content.text {
        if !capabilities.text {
            return Err(FiniteChoiceError::new(
                FiniteChoiceErrorKind::UnsupportedText,
                "Text is unsupported at this content location",
            )
            .at(location));
        }
        if text.trim().is_empty() {
            return Err(FiniteChoiceError::new(
                FiniteChoiceErrorKind::InvalidRequest,
                "Present text must be nonempty",
            )
            .at(location));
        }
    }
    for visual in &content.visuals {
        if !capabilities.visual_formats.contains(&visual.format) {
            return Err(FiniteChoiceError::new(
                FiniteChoiceErrorKind::UnsupportedVisual,
                format!("Visual format {:?} is unsupported here", visual.format),
            )
            .at(location));
        }
        if visual.bytes.is_empty() {
            return Err(FiniteChoiceError::new(
                FiniteChoiceErrorKind::InvalidRequest,
                "Visual bytes must be nonempty",
            )
            .at(location));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct FiniteChoiceScore {
    pub option_id: String,
    pub logit: f64,
    /// Unit-temperature softmax of the ordered logits. This is not calibrated confidence.
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct FiniteChoiceResult {
    /// Exactly one entry per input option, preserving its ID and position.
    pub scores: Vec<FiniteChoiceScore>,
}

/// Absolute tolerance for probability normalization and unit-temperature softmax consistency.
pub const PROBABILITY_TOLERANCE: f64 = 1.0e-6;

impl FiniteChoiceResult {
    /// Constructs ordered scores using max-shifted softmax, avoiding overflow for large logits.
    pub fn from_logits(
        request: &ValidatedFiniteChoiceRequest<'_>,
        logits: Vec<f64>,
    ) -> Result<Self, FiniteChoiceError> {
        if logits.len() != request.request.options.len() {
            return Err(invalid_result("Logit count must equal option count"));
        }
        let probabilities = softmax(&logits)?;
        let scores = request
            .request
            .options
            .iter()
            .zip(logits.into_iter().zip(probabilities))
            .map(|(option, (logit, probability))| FiniteChoiceScore {
                option_id: option.id.clone(),
                logit,
                probability,
            })
            .collect();
        Ok(Self { scores })
    }

    /// Rejects missing, reordered, non-finite or inconsistent provider output.
    pub fn validate(
        &self,
        request: &ValidatedFiniteChoiceRequest<'_>,
    ) -> Result<(), FiniteChoiceError> {
        if self.scores.len() != request.request.options.len() {
            return Err(invalid_result("Score count must equal option count"));
        }
        for (score, option) in self.scores.iter().zip(&request.request.options) {
            if score.option_id != option.id {
                return Err(invalid_result(
                    "Scores must preserve the exact input option IDs and order",
                ));
            }
            if !score.probability.is_finite() || !(0.0..=1.0).contains(&score.probability) {
                return Err(invalid_result("Probabilities must be finite and in [0, 1]"));
            }
        }
        let probability_sum: f64 = self.scores.iter().map(|score| score.probability).sum();
        if (probability_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(invalid_result("Probabilities must sum to one"));
        }
        let logits: Vec<_> = self.scores.iter().map(|score| score.logit).collect();
        let expected = softmax(&logits)?;
        for (score, expected_probability) in self.scores.iter().zip(expected) {
            if (score.probability - expected_probability).abs() > PROBABILITY_TOLERANCE {
                return Err(invalid_result(
                    "Probabilities must match unit-temperature softmax of the logits",
                ));
            }
        }
        Ok(())
    }

    /// Returns the highest-logit score; exact ties retain the first input option.
    ///
    /// Call after validation when consuming a result constructed outside `from_logits`.
    #[must_use]
    pub fn best_option(&self) -> Option<&FiniteChoiceScore> {
        let mut best = self.scores.first()?;
        for score in &self.scores[1..] {
            if score.logit > best.logit {
                best = score;
            }
        }
        Some(best)
    }
}

fn invalid_result(message: impl Into<String>) -> FiniteChoiceError {
    FiniteChoiceError::new(FiniteChoiceErrorKind::InvalidResult, message)
}

fn softmax(logits: &[f64]) -> Result<Vec<f64>, FiniteChoiceError> {
    if logits.is_empty() || logits.iter().any(|logit| !logit.is_finite()) {
        return Err(invalid_result("Logits must be nonempty and finite"));
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = logits.iter().map(|logit| (logit - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|weight| weight / total).collect())
}

/// Provider seam for a future inference adapter. It imposes no transport, runtime or model engine.
pub trait FiniteChoiceProvider {
    fn capabilities(&self) -> BackendCapabilities;

    fn infer(
        &mut self,
        request: &ValidatedFiniteChoiceRequest<'_>,
    ) -> Result<FiniteChoiceResult, FiniteChoiceError>;
}

/// Performs deterministic contract checks around one provider call; no inference is built in.
pub fn finite_choice(
    provider: &mut (impl FiniteChoiceProvider + ?Sized),
    request: &FiniteChoiceRequest,
) -> Result<FiniteChoiceResult, FiniteChoiceError> {
    let validated = request.validate_for(&provider.capabilities())?;
    let result = provider.infer(&validated)?;
    result.validate(&validated)?;
    Ok(result)
}
