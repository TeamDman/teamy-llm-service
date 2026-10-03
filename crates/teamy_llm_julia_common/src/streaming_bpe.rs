//! Stream modern BPE merge tuples directly into the upstream model builder.
//!
//! Tokenizers 0.22.2's BPE deserializer first captures its untagged tuple/legacy
//! merge list before deserializing it again. The pinned Julia bundle contains
//! modern tuples, so this adapter avoids that large intermediate. The upstream
//! BPE builder, encoding algorithms and tokenizer components remain authoritative.

use std::{
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
};

use serde::{
    Deserialize, Deserializer,
    de::{Error, MapAccess, Visitor},
};
use tokenizers::{
    Model, ModelWrapper, Token,
    models::bpe::{BPE, BpeBuilder, Merges, Vocab},
};

pub(crate) struct StreamingBpe(BPE);

impl From<StreamingBpe> for ModelWrapper {
    fn from(model: StreamingBpe) -> Self {
        Self::BPE(model.0)
    }
}

impl Model for StreamingBpe {
    type Trainer = <BPE as Model>::Trainer;

    fn tokenize(&self, sequence: &str) -> tokenizers::Result<Vec<Token>> {
        self.0.tokenize(sequence)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.0.get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.0.get_vocab_size()
    }

    fn save(&self, folder: &Path, prefix: Option<&str>) -> tokenizers::Result<Vec<PathBuf>> {
        self.0.save(folder, prefix)
    }

    fn get_trainer(&self) -> Self::Trainer {
        self.0.get_trainer()
    }
}

const FIELDS: &[&str] = &[
    "type",
    "dropout",
    "unk_token",
    "continuing_subword_prefix",
    "end_of_word_suffix",
    "fuse_unk",
    "byte_fallback",
    "ignore_merges",
    "vocab",
    "merges",
];

impl<'de> Deserialize<'de> for StreamingBpe {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_struct("BPE", FIELDS, StreamingBpeVisitor)
    }
}

struct StreamingBpeVisitor;

impl<'de> Visitor<'de> for StreamingBpeVisitor {
    type Value = StreamingBpe;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a BPE model with modern merge tuples")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut builder = BpeBuilder::new();
        let mut vocab: Option<Vocab> = None;
        let mut merges: Option<Merges> = None;
        let mut bpe_type = false;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => {
                    let model_type = map.next_value::<String>()?;
                    if model_type != "BPE" {
                        return Err(A::Error::custom("Streaming tokenizer model must be BPE"));
                    }
                    bpe_type = true;
                }
                "dropout" => {
                    if let Some(value) = map.next_value::<Option<f32>>()? {
                        builder = builder.dropout(value);
                    }
                }
                "unk_token" => {
                    if let Some(value) = map.next_value::<Option<String>>()? {
                        builder = builder.unk_token(value);
                    }
                }
                "continuing_subword_prefix" => {
                    if let Some(value) = map.next_value::<Option<String>>()? {
                        builder = builder.continuing_subword_prefix(value);
                    }
                }
                "end_of_word_suffix" => {
                    if let Some(value) = map.next_value::<Option<String>>()? {
                        builder = builder.end_of_word_suffix(value);
                    }
                }
                "fuse_unk" => {
                    if let Some(value) = map.next_value::<Option<bool>>()? {
                        builder = builder.fuse_unk(value);
                    }
                }
                "byte_fallback" => {
                    if let Some(value) = map.next_value::<Option<bool>>()? {
                        builder = builder.byte_fallback(value);
                    }
                }
                "ignore_merges" => {
                    if let Some(value) = map.next_value::<Option<bool>>()? {
                        builder = builder.ignore_merges(value);
                    }
                }
                "vocab" => vocab = Some(map.next_value()?),
                // No untagged enum: deserialize tuple strings exactly once.
                "merges" => merges = Some(map.next_value()?),
                // Upstream's typed visitor does not consume unknown-field values.
                // Defer future/legacy shapes to the established generic fallback.
                _ => return Err(A::Error::unknown_field(&key, FIELDS)),
            }
        }
        if !bpe_type {
            return Err(A::Error::missing_field("type"));
        }
        let vocab = vocab.ok_or_else(|| A::Error::missing_field("vocab"))?;
        let merges = merges.ok_or_else(|| A::Error::missing_field("merges"))?;
        builder
            .vocab_and_merges(vocab, merges)
            .build()
            .map(StreamingBpe)
            .map_err(A::Error::custom)
    }
}
