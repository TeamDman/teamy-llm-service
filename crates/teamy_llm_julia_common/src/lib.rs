//! One lossless Julia-1 text encoder shared by the Rust and ONNX providers.
//!
//! Serialization follows the publisher's pinned `julia/data.py` and ONNX Rust encoder.
//! Inference uses batch one and the publisher's padding-to-eight collator. This module
//! does not load numerical weights, infer, acquire artifacts or silently discard images.

use std::path::Path;

use eyre::{WrapErr, ensure};
use facet::Facet;
use teamy_llm_core::finite_choice::{
    BackendCapabilities, FiniteChoiceCapabilities, FiniteChoiceError, FiniteChoiceErrorKind,
    ValidatedFiniteChoiceRequest,
};
use tokenizers::Tokenizer;

pub const JULIA_MODEL_REVISION: &str = "a85b127321d580d65176c89ced8273f305745d85";
pub const JULIA_ONNX_REVISION: &str = "82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8";
pub const DEFAULT_QUESTION: &str = "Which option should be chosen?";
pub const JULIA_VOCAB_SIZE: i64 = 256_000;

#[derive(Clone, Debug, PartialEq, Eq, Facet)]
pub struct EncodingPolicy {
    pub max_tokens: usize,
    pub head_tokens: usize,
    pub question: String,
}

impl Default for EncodingPolicy {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            head_tokens: 256,
            question: DEFAULT_QUESTION.into(),
        }
    }
}

impl EncodingPolicy {
    pub fn validate(&self) -> Result<(), FiniteChoiceError> {
        if !(16..=8192).contains(&self.max_tokens)
            || self.head_tokens < 16
            || self
                .head_tokens
                .checked_add(4)
                .is_none_or(|n| n >= self.max_tokens)
            || self.question.trim().is_empty()
        {
            return Err(invalid(
                "Invalid Julia limits or empty question: head must leave context within 8192 tokens",
            ));
        }
        Ok(())
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Facet)]
pub enum JuliaQuestionType {
    Choice,
    Score,
    Noul,
}

impl JuliaQuestionType {
    pub const fn id(self) -> i64 {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Facet)]
pub struct JuliaTextOption {
    pub id: String,
    pub text: String,
}

/// Flat batch-one arrays. `raw_input_ids` precedes collator padding; providers feed
/// `input_ids` and `attention_mask`. Markers retain the caller's actual option count.
#[derive(Clone, Debug, PartialEq, Eq, Facet)]
#[facet(deny_unknown_fields)]
pub struct EncodedJuliaRequest {
    pub version: u32,
    pub raw_input_ids: Vec<i64>,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub marker_positions: Vec<i64>,
    pub marker_mask: Vec<bool>,
    pub qtype: i64,
    pub option_ids: Vec<String>,
}

impl EncodedJuliaRequest {
    pub fn token_count(&self) -> usize {
        self.input_ids.len()
    }

    pub fn option_count(&self) -> usize {
        self.option_ids.len()
    }

    /// Validate transport arrays before any tensor allocation or session invocation.
    pub fn validate(&self) -> Result<(), FiniteChoiceError> {
        let raw = self.raw_input_ids.len();
        if self.version != 1
            || !(2..=20).contains(&self.option_count())
            || self.token_count() > 8192
            || raw < 4
            || raw > self.token_count()
            || self.token_count().saturating_sub(raw) > 7
            || self.attention_mask.len() != self.token_count()
            || self.marker_positions.len() != self.option_count()
            || self.marker_mask.len() != self.option_count()
            || !(0..=2).contains(&self.qtype)
            || (self.qtype == 2 && self.option_count() != 2)
            || self.raw_input_ids.first() != Some(&2)
            || self.raw_input_ids.last() != Some(&1)
            || self.input_ids.get(..raw) != Some(self.raw_input_ids.as_slice())
        {
            return Err(invalid(
                "Invalid encoded Julia version, lengths, type or sequence",
            ));
        }
        for (index, (&token, &mask)) in self.input_ids.iter().zip(&self.attention_mask).enumerate()
        {
            if !(0..JULIA_VOCAB_SIZE).contains(&token)
                || mask != i64::from(index < raw)
                || (index >= raw && token != 0)
            {
                return Err(invalid(
                    "Encoded Julia token or attention padding is invalid",
                ));
            }
        }
        for (index, (&position, &present)) in self
            .marker_positions
            .iter()
            .zip(&self.marker_mask)
            .enumerate()
        {
            let Ok(position) = usize::try_from(position) else {
                return Err(invalid("Negative Julia marker position"));
            };
            if !present
                || position >= raw
                || self.input_ids[position] != 4
                || (index > 0 && self.marker_positions[index - 1] >= position as i64)
                || self.option_ids[index].trim().is_empty()
                || self.option_ids[..index].contains(&self.option_ids[index])
            {
                return Err(invalid(
                    "Invalid Julia marker mask, position or option identity",
                ));
            }
        }
        if self
            .raw_input_ids
            .iter()
            .filter(|&&token| token == 4)
            .count()
            != self.option_count()
        {
            return Err(invalid("Ambiguous Julia markers"));
        }
        Ok(())
    }
}

pub fn julia_capabilities() -> BackendCapabilities {
    BackendCapabilities {
        text_generation: false,
        finite_choice: Some(FiniteChoiceCapabilities::julia_1_text_only()),
    }
}

pub struct JuliaEncoder {
    tokenizer: Tokenizer,
    policy: EncodingPolicy,
    special_literals: Vec<String>,
}

impl JuliaEncoder {
    pub fn load(model_dir: &Path, policy: EncodingPolicy) -> eyre::Result<Self> {
        // The native publisher bundle uses tokenizer/tokenizer.json; its ONNX bundle is flat.
        // Only these two fixed layouts are accepted, with no traversal from artifact metadata.
        let flat = model_dir.join("tokenizer.json");
        let nested = model_dir.join("tokenizer/tokenizer.json");
        ensure!(
            !(flat.is_file() && nested.is_file()),
            "Ambiguous Julia tokenizer: both published layouts are present"
        );
        let tokenizer_path = if nested.is_file() { nested } else { flat };
        let json = std::fs::read_to_string(tokenizer_path)
            .wrap_err("Read the selected Julia tokenizer")?;
        Self::from_tokenizer_json(&json, policy)
    }

    pub fn from_tokenizer_json(json: &str, policy: EncodingPolicy) -> eyre::Result<Self> {
        policy.validate()?;
        let mut tokenizer = Tokenizer::from_bytes(json.as_bytes())
            .map_err(|error| eyre::eyre!("Invalid Julia tokenizer: {error}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| eyre::eyre!("Tokenizer truncation: {error}"))?;
        tokenizer.with_padding(None);
        ensure!(
            tokenizer.token_to_id("<mask>") == Some(4),
            "Julia tokenizer requires mask ID 4"
        );
        ensure!(
            tokenizer.token_to_id("<pad>") == Some(0)
                && tokenizer.token_to_id("<eos>") == Some(1)
                && tokenizer.token_to_id("<bos>") == Some(2),
            "Julia tokenizer lacks the required PAD/SEP/CLS identities"
        );
        let special_literals = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter(|token| token.special && !token.content.is_empty())
            .map(|token| token.content)
            .collect();
        Ok(Self {
            tokenizer,
            policy,
            special_literals,
        })
    }

    pub fn policy(&self) -> &EncodingPolicy {
        &self.policy
    }

    /// Core projection: input text is state; policy carries the explicit/default question.
    pub fn encode(
        &self,
        request: &ValidatedFiniteChoiceRequest<'_>,
    ) -> Result<EncodedJuliaRequest, FiniteChoiceError> {
        let request = request.request();
        // A validated wrapper may have been validated for a different provider's capabilities.
        request.validate_for(&julia_capabilities())?;
        let options = request
            .options
            .iter()
            .map(|option| JuliaTextOption {
                id: option.id.clone(),
                text: option.content.text.clone().unwrap_or_default(),
            })
            .collect::<Vec<_>>();
        self.encode_parts(
            request.input.text.as_deref().unwrap_or_default(),
            &self.policy.question,
            JuliaQuestionType::Choice,
            &options,
        )
    }

    pub fn encode_parts(
        &self,
        state: &str,
        question: &str,
        kind: JuliaQuestionType,
        options: &[JuliaTextOption],
    ) -> Result<EncodedJuliaRequest, FiniteChoiceError> {
        self.policy.validate()?;
        if question.trim().is_empty()
            || !(2..=20).contains(&options.len())
            || (kind == JuliaQuestionType::Noul && options.len() != 2)
        {
            return Err(invalid(
                "Julia requires a question and 2–20 supplied options",
            ));
        }
        for (index, option) in options.iter().enumerate() {
            if option.id.trim().is_empty()
                || option.text.trim().is_empty()
                || options[..index]
                    .iter()
                    .any(|previous| previous.id == option.id)
            {
                return Err(invalid(
                    "Julia options require distinct IDs and nonempty text",
                ));
            }
        }
        let head = self.tokens(&format!("{} question: {question}", kind.name()))?;
        let option_tokens = options
            .iter()
            .map(|option| self.tokens(&format!(" {}", option.text)))
            .collect::<Result<Vec<_>, _>>()?;
        if option_tokens.iter().any(|tokens| tokens.len() > 48) {
            return Err(invalid("Option exceeds the 48-token Julia contract"));
        }
        let mut marked = option_tokens
            .iter()
            .map(|tokens| {
                let mut result = vec![4_i64];
                result.extend(tokens);
                result
            })
            .collect::<Vec<_>>();
        let mut budget =
            self.policy.head_tokens as isize - marked.iter().map(Vec::len).sum::<usize>() as isize;
        if budget < 16 {
            let cap = ((self.policy.head_tokens.saturating_sub(16)) / options.len()).max(4);
            for tokens in &mut marked {
                tokens.truncate(cap);
            }
            budget = self.policy.head_tokens as isize
                - marked.iter().map(Vec::len).sum::<usize>() as isize;
        }
        if budget < 0
            || head.len() > budget as usize
            || marked
                .iter()
                .zip(&option_tokens)
                .any(|(a, b)| a.len() != b.len() + 1)
        {
            return Err(invalid(
                "Question/options exceed the lossless Julia head budget",
            ));
        }
        let mut ids = vec![2_i64];
        ids.extend(head);
        ids.push(1);
        let mut marker_positions = Vec::with_capacity(options.len());
        for tokens in marked {
            marker_positions.push(ids.len() as i64);
            ids.extend(tokens);
        }
        ids.push(1);
        let state_tokens = self.tokens(state)?;
        let room = self.policy.max_tokens.saturating_sub(ids.len() + 1);
        if room < 1 || state_tokens.len() > room {
            return Err(invalid("State exceeds the lossless Julia context budget"));
        }
        ids.extend(state_tokens);
        ids.push(1);
        let raw_input_ids = ids.clone();
        let raw_length = ids.len();
        let padded_length = raw_length
            .div_ceil(8)
            .saturating_mul(8)
            .min(self.policy.max_tokens);
        ids.resize(padded_length, 0);
        let mut attention_mask = vec![0; padded_length];
        attention_mask[..raw_length].fill(1);
        let result = EncodedJuliaRequest {
            version: 1,
            raw_input_ids,
            input_ids: ids,
            attention_mask,
            marker_positions,
            marker_mask: vec![true; options.len()],
            qtype: kind.id(),
            option_ids: options.iter().map(|option| option.id.clone()).collect(),
        };
        result.validate()?;
        Ok(result)
    }

    fn tokens(&self, text: &str) -> Result<Vec<i64>, FiniteChoiceError> {
        if text.contains("<mask>")
            || self
                .special_literals
                .iter()
                .any(|token| text.contains(token))
        {
            return Err(invalid("Reserved Julia marker/special token in request"));
        }
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| invalid(format!("Julia tokenizer failed: {error}")))?;
        let ids = encoded
            .get_ids()
            .iter()
            .map(|&token| i64::from(token))
            .collect::<Vec<_>>();
        if ids.iter().any(|token| matches!(token, 0 | 1 | 2 | 4)) {
            return Err(invalid("Reserved Julia special token in request"));
        }
        Ok(ids)
    }
}

fn invalid(message: impl Into<String>) -> FiniteChoiceError {
    FiniteChoiceError::new(FiniteChoiceErrorKind::InvalidRequest, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> JuliaEncoder {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":4,"content":"<mask>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"<mask>":4,"choice":5,"question:":6,"First?":7,"one":8,"two":9,"state":10},"unk_token":"<unk>"}}"#;
        JuliaEncoder::from_tokenizer_json(json, EncodingPolicy::default()).unwrap()
    }

    fn options() -> Vec<JuliaTextOption> {
        vec![
            JuliaTextOption {
                id: "a".into(),
                text: "one".into(),
            },
            JuliaTextOption {
                id: "b".into(),
                text: "two".into(),
            },
        ]
    }

    #[test]
    fn preserves_sequence_markers_order_and_eight_token_padding() {
        let actual = encoder()
            .encode_parts("state", "First?", JuliaQuestionType::Choice, &options())
            .unwrap();
        assert_eq!(actual.raw_input_ids, [2, 5, 6, 7, 1, 4, 8, 4, 9, 1, 10, 1]);
        assert_eq!(actual.marker_positions, [5, 7]);
        assert_eq!(actual.option_ids, ["a", "b"]);
        assert_eq!(actual.input_ids.len(), 16);
        assert_eq!(
            actual.attention_mask,
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        );
    }

    #[test]
    fn refuses_marker_injection_and_lossy_option_budget() {
        assert!(
            encoder()
                .encode_parts(
                    "state <mask>",
                    "First?",
                    JuliaQuestionType::Choice,
                    &options()
                )
                .is_err()
        );
        assert!(
            encoder()
                .encode_parts(
                    "state",
                    "First? <bos>",
                    JuliaQuestionType::Choice,
                    &options()
                )
                .is_err()
        );
        let mut oversized = options();
        oversized[0].text = "one ".repeat(49);
        assert!(
            encoder()
                .encode_parts("state", "First?", JuliaQuestionType::Choice, &oversized)
                .is_err()
        );
    }

    #[test]
    fn rejects_corrupted_transport_markers_and_padding() {
        let mut encoded = encoder()
            .encode_parts("state", "First?", JuliaQuestionType::Choice, &options())
            .unwrap();
        encoded.marker_positions[1] = encoded.marker_positions[0];
        assert!(encoded.validate().is_err());
        let mut encoded = encoder()
            .encode_parts("state", "First?", JuliaQuestionType::Choice, &options())
            .unwrap();
        *encoded.input_ids.last_mut().unwrap() = 3;
        assert!(encoded.validate().is_err());
    }

    #[test]
    fn rejects_padding_beyond_the_publisher_collator_bound() {
        let mut encoded = encoder()
            .encode_parts("state", "First?", JuliaQuestionType::Choice, &options())
            .unwrap();
        encoded.input_ids.resize(32, 0);
        encoded.attention_mask.resize(32, 0);
        assert!(encoded.validate().is_err());
    }
}
