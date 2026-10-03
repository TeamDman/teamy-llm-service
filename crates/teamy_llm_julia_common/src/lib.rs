//! One lossless Julia-1 text encoder shared by the Rust and ONNX providers.
//!
//! Serialization follows the publisher's pinned `julia/data.py` and ONNX Rust encoder.
//! Inference uses batch one and the publisher's padding-to-eight collator. This module
//! does not load numerical weights, infer, acquire artifacts or silently discard images.

use std::{path::Path, sync::Arc};

use eyre::{WrapErr, ensure};
use facet::Facet;
use teamy_llm_core::finite_choice::{
    BackendCapabilities, FiniteChoiceCapabilities, FiniteChoiceError, FiniteChoiceErrorKind,
    ValidatedFiniteChoiceRequest,
};
use tokenizers::{
    DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer,
    TokenizerImpl,
};

mod streaming_bpe;
use streaming_bpe::StreamingBpe;

type BpeTokenizer = TokenizerImpl<
    StreamingBpe,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

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

/// Clones share the parsed tokenizer while retaining their own encoding policy.
#[derive(Clone)]
pub struct JuliaEncoder {
    tokenizer: Arc<Tokenizer>,
    policy: EncodingPolicy,
    special_literals: Vec<String>,
}

impl JuliaEncoder {
    #[tracing::instrument(level = "debug", name = "julia_tokenizer_load", skip_all)]
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

    #[tracing::instrument(level = "debug", name = "julia_tokenizer_parse", skip_all)]
    pub fn from_tokenizer_json(json: &str, policy: EncodingPolicy) -> eyre::Result<Self> {
        policy.validate()?;
        // The published modern BPE bundle streams directly into upstream's builder, avoiding
        // ModelWrapper's JSON value and BPE's untagged merge-table intermediate/reparse.
        // Other upstream model types retain the established generic parsing behavior.
        let mut tokenizer = match BpeTokenizer::from_bytes(json.as_bytes()) {
            Ok(tokenizer) => {
                tracing::debug!(generic_fallback = false, "Julia tokenizer parsed");
                tokenizer.into()
            }
            Err(_) => {
                let tokenizer = Tokenizer::from_bytes(json.as_bytes())
                    .map_err(|error| eyre::eyre!("Invalid Julia tokenizer: {error}"))?;
                tracing::debug!(generic_fallback = true, "Julia tokenizer parsed");
                tokenizer
            }
        };
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
            tokenizer: Arc::new(tokenizer),
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

    const BPE_JSON: &str = r#"{
        "version":"1.0", "truncation":null, "padding":null,
        "added_tokens":[{"id":4,"content":"<mask>","single_word":false,
            "lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
        "pre_tokenizer":{"type":"Metaspace","replacement":"▁",
            "prepend_scheme":"always","split":true},
        "post_processor":null,
        "decoder":{"type":"Metaspace","replacement":"▁",
            "prepend_scheme":"always","split":true},
        "model":{"type":"BPE","dropout":null,"unk_token":"<unk>",
            "continuing_subword_prefix":null,"end_of_word_suffix":null,
            "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
            "vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"<mask>":4,
                "▁":5,"a":6,"b":7,"ab":8,"▁ab":9,"ba":10,"▁ba":11,
                "c":12,"h":13,"o":14,"i":15,"e":16,"q":17,"u":18,
                "s":19,"t":20,"n":21,":":22},
            "merges":[["a","b"],["▁","ab"],["b","a"],["▁","ba"]]}
    }"#;

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
    fn typed_bpe_matches_generic_normalization_pretokenization_and_merges() {
        let typed: Tokenizer = BpeTokenizer::from_bytes(BPE_JSON.as_bytes())
            .unwrap()
            .into();
        let generic = Tokenizer::from_bytes(BPE_JSON.as_bytes()).unwrap();
        for text in ["ab ab", " ba", "choice question: ab", "▁ab", "ab\nば"] {
            assert_eq!(
                typed.encode(text, false).unwrap(),
                generic.encode(text, false).unwrap(),
                "Tokenizer behavior differs for {text:?}"
            );
        }
        let merged = typed.encode("ab ab", false).unwrap();
        assert_eq!(merged.get_ids(), &[9, 9]);
        assert_eq!(typed.decode(merged.get_ids(), true).unwrap(), "ab ab");
        assert_eq!(typed.encode(" ba", false).unwrap().get_ids(), &[11]);
    }

    #[test]
    fn typed_bpe_preserves_julia_encoding_and_special_rejection() {
        let typed = JuliaEncoder::from_tokenizer_json(BPE_JSON, EncodingPolicy::default()).unwrap();
        let generic_tokenizer = Tokenizer::from_bytes(BPE_JSON.as_bytes()).unwrap();
        let generic = JuliaEncoder {
            special_literals: generic_tokenizer
                .get_added_tokens_decoder()
                .into_values()
                .filter(|token| token.special && !token.content.is_empty())
                .map(|token| token.content)
                .collect(),
            tokenizer: Arc::new(generic_tokenizer),
            policy: EncodingPolicy::default(),
        };
        let options = [
            JuliaTextOption {
                id: "a".into(),
                text: "ab".into(),
            },
            JuliaTextOption {
                id: "b".into(),
                text: "ba".into(),
            },
        ];
        assert_eq!(
            typed
                .encode_parts("ab ab", "ab", JuliaQuestionType::Choice, &options)
                .unwrap(),
            generic
                .encode_parts("ab ab", "ab", JuliaQuestionType::Choice, &options)
                .unwrap(),
        );
        for encoder in [&typed, &generic] {
            assert!(
                encoder
                    .encode_parts("ab <mask>", "ab", JuliaQuestionType::Choice, &options)
                    .is_err()
            );
        }
    }

    #[test]
    fn streaming_bpe_preserves_all_builder_flags_and_nullable_defaults() {
        let empty_merges = BPE_JSON.replace(
            "[[\"a\",\"b\"],[\"▁\",\"ab\"],[\"b\",\"a\"],[\"▁\",\"ba\"]]",
            "[]",
        );
        let cases = [
            BPE_JSON.replace("\"dropout\":null", "\"dropout\":0.25"),
            BPE_JSON.replace("\"unk_token\":\"<unk>\"", "\"unk_token\":null"),
            empty_merges.replace(
                "\"continuing_subword_prefix\":null",
                "\"continuing_subword_prefix\":\"##\"",
            ),
            empty_merges.replace(
                "\"end_of_word_suffix\":null",
                "\"end_of_word_suffix\":\"$\"",
            ),
            BPE_JSON.replace("\"fuse_unk\":false", "\"fuse_unk\":true"),
            BPE_JSON.replace("\"byte_fallback\":false", "\"byte_fallback\":true"),
            BPE_JSON.replace("\"ignore_merges\":false", "\"ignore_merges\":true"),
            BPE_JSON
                .replace("\"fuse_unk\":false", "\"fuse_unk\":null")
                .replace("\"byte_fallback\":false", "\"byte_fallback\":null")
                .replace("\"ignore_merges\":false", "\"ignore_merges\":null"),
        ];
        for json in cases {
            let streamed: Tokenizer = BpeTokenizer::from_bytes(json.as_bytes()).unwrap().into();
            let generic = Tokenizer::from_bytes(json.as_bytes()).unwrap();
            assert_eq!(streamed.get_model(), generic.get_model());
        }
    }

    #[test]
    fn legacy_bpe_merges_retain_generic_fallback_and_exact_julia_encoding() {
        let legacy = BPE_JSON.replace(
            "[[\"a\",\"b\"],[\"▁\",\"ab\"],[\"b\",\"a\"],[\"▁\",\"ba\"]]",
            "[\"a b\",\"▁ ab\",\"b a\",\"▁ ba\"]",
        );
        assert!(BpeTokenizer::from_bytes(legacy.as_bytes()).is_err());
        let legacy_encoder =
            JuliaEncoder::from_tokenizer_json(&legacy, EncodingPolicy::default()).unwrap();
        let modern_encoder =
            JuliaEncoder::from_tokenizer_json(BPE_JSON, EncodingPolicy::default()).unwrap();
        let options = [
            JuliaTextOption {
                id: "a".into(),
                text: "ab".into(),
            },
            JuliaTextOption {
                id: "b".into(),
                text: "ba".into(),
            },
        ];
        assert_eq!(
            legacy_encoder
                .encode_parts("ab ba ab", "ab", JuliaQuestionType::Choice, &options)
                .unwrap(),
            modern_encoder
                .encode_parts("ab ba ab", "ab", JuliaQuestionType::Choice, &options)
                .unwrap()
        );
    }

    #[test]
    fn streaming_bpe_rejects_non_bpe_models_and_invalid_dropout() {
        let wrong_type = BPE_JSON.replace("\"type\":\"BPE\"", "\"type\":\"WordPiece\"");
        assert!(BpeTokenizer::from_bytes(wrong_type.as_bytes()).is_err());
        for dropout in [-0.25, 1.25] {
            let malformed = BPE_JSON.replace("\"dropout\":null", &format!("\"dropout\":{dropout}"));
            assert!(BpeTokenizer::from_bytes(malformed.as_bytes()).is_err());
            assert!(Tokenizer::from_bytes(malformed.as_bytes()).is_err());
            assert!(
                JuliaEncoder::from_tokenizer_json(&malformed, EncodingPolicy::default()).is_err()
            );
        }
    }

    #[test]
    fn streaming_bpe_rejects_malformed_tuple_and_merge_output_without_vocabulary_entry() {
        for malformed in [
            BPE_JSON.replace("[\"a\",\"b\"]", "[\"a\",\"b\",\"ab\"]"),
            BPE_JSON.replace("\"ab\":8,", ""),
        ] {
            assert!(BpeTokenizer::from_bytes(malformed.as_bytes()).is_err());
            assert!(Tokenizer::from_bytes(malformed.as_bytes()).is_err());
            assert!(
                JuliaEncoder::from_tokenizer_json(&malformed, EncodingPolicy::default()).is_err()
            );
        }
    }

    #[test]
    fn malformed_bpe_is_not_accepted_by_generic_fallback() {
        let malformed = BPE_JSON.replace("[\"a\",\"b\"]", "[\"missing\",\"b\"]");
        assert!(BpeTokenizer::from_bytes(malformed.as_bytes()).is_err());
        assert!(Tokenizer::from_bytes(malformed.as_bytes()).is_err());
        assert!(JuliaEncoder::from_tokenizer_json(&malformed, EncodingPolicy::default()).is_err());
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
    fn clone_shares_tokenizer_and_preserves_complete_encoding() {
        let encoder = encoder();
        let cloned = encoder.clone();
        assert!(Arc::ptr_eq(&encoder.tokenizer, &cloned.tokenizer));
        assert_eq!(encoder.policy(), cloned.policy());
        let encode = |encoder: &JuliaEncoder| {
            encoder
                .encode_parts("state", "First?", JuliaQuestionType::Choice, &options())
                .unwrap()
        };
        assert_eq!(encode(&encoder), encode(&cloned));
        assert!(
            cloned
                .encode_parts(
                    "state <mask>",
                    "First?",
                    JuliaQuestionType::Choice,
                    &options()
                )
                .is_err()
        );
    }

    #[test]
    fn cloned_policy_is_independent_and_survives_original_drop() {
        let encoder = encoder();
        let original_policy = encoder.policy().clone();
        let mut cloned = encoder.clone();
        cloned.policy.max_tokens = 64;
        cloned.policy.head_tokens = 32;
        assert_eq!(encoder.policy(), &original_policy);
        drop(encoder);
        assert_eq!(cloned.policy().max_tokens, 64);
        let actual = cloned
            .encode_parts("state", "First?", JuliaQuestionType::Choice, &options())
            .unwrap();
        assert_eq!(actual.raw_input_ids, [2, 5, 6, 7, 1, 4, 8, 4, 9, 1, 10, 1]);
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
