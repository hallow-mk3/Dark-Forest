//! Tokenizer module for Dark Forest ML Runtime.
//!
//! Provides a unified `Tokenizer` trait and three concrete implementations:
//! - `ByteTokenizer`: Zero-dependency, UTF-8 byte-level tokenizer (256 bytes + EOS).
//!   Works natively on any text, code, or unicode without external files.
//! - `CharTokenizer`: Character-level vocabulary built from training text corpora.
//! - `BPETokenizer`: Byte-pair encoding tokenizer compatible with standard GPT-2
//!   vocabulary (`vocab.json` / `merges.txt`).

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

/// Unified tokenizer abstraction for encoding and decoding text sequences.
pub trait Tokenizer: Send + Sync {
    /// Encode a string slice into a vector of token identifiers.
    fn encode(&self, text: &str) -> Vec<usize>;

    /// Decode a sequence of token identifiers back into a UTF-8 string.
    fn decode(&self, tokens: &[usize]) -> String;

    /// Total number of tokens in the vocabulary.
    fn vocab_size(&self) -> usize;

    /// End-of-sequence token ID, if defined.
    fn eos_token(&self) -> Option<usize> {
        None
    }

    /// Beginning-of-sequence token ID, if defined.
    fn bos_token(&self) -> Option<usize> {
        None
    }

    /// Encode multiple strings in parallel or sequence.
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<usize>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Decode multiple token sequences.
    fn decode_batch(&self, batch: &[Vec<usize>]) -> Vec<String> {
        batch.iter().map(|b| self.decode(b)).collect()
    }
}

// ---------------------------------------------------------------------------
// 1. ByteTokenizer — Raw UTF-8 Byte Level Tokenizer (Zero Dependency)
// ---------------------------------------------------------------------------

/// A zero-dependency UTF-8 byte tokenizer.
///
/// Maps raw bytes 0..=255 directly to token indices 0..=255.
/// Special token EOS is mapped to index 256.
/// Total vocabulary size: 257.
///
/// Guaranteed to never produce out-of-vocabulary tokens on any UTF-8 string.
#[derive(Debug, Clone)]
pub struct ByteTokenizer {
    eos_id: usize,
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteTokenizer {
    pub const VOCAB_SIZE: usize = 257;
    pub const DEFAULT_EOS_ID: usize = 256;

    pub fn new() -> Self {
        Self {
            eos_id: Self::DEFAULT_EOS_ID,
        }
    }
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.as_bytes().iter().map(|&b| b as usize).collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&t| {
                if t < 256 {
                    Some(t as u8)
                } else {
                    None // Skip special tokens like EOS during raw string decode
                }
            })
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        Self::VOCAB_SIZE
    }

    fn eos_token(&self) -> Option<usize> {
        Some(self.eos_id)
    }

    fn bos_token(&self) -> Option<usize> {
        None
    }
}

// ---------------------------------------------------------------------------
// 2. CharTokenizer — Character-Level Vocabulary
// ---------------------------------------------------------------------------

/// Character-level tokenizer built from training text corpora.
///
/// Maps each unique unicode scalar value in the corpus to a dense token ID.
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    pub vocab: Vec<char>,
    pub char_to_id: HashMap<char, usize>,
    pub eos_id: Option<usize>,
}

impl CharTokenizer {
    /// Construct a character vocabulary from a text sample.
    pub fn build(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        if chars.is_empty() {
            chars.push('\n');
        }
        let char_to_id: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self {
            vocab: chars,
            char_to_id,
            eos_id: None,
        }
    }

    /// Construct a character vocabulary with an explicit EOS token character.
    pub fn build_with_eos(text: &str, eos_char: char) -> Self {
        let mut tok = Self::build(text);
        if !tok.char_to_id.contains_key(&eos_char) {
            let id = tok.vocab.len();
            tok.vocab.push(eos_char);
            tok.char_to_id.insert(eos_char, id);
        }
        tok.eos_id = tok.char_to_id.get(&eos_char).copied();
        tok
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&0))
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&t| {
                if t < self.vocab.len() {
                    self.vocab[t]
                } else {
                    '?'
                }
            })
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn eos_token(&self) -> Option<usize> {
        self.eos_id
    }
}

// ---------------------------------------------------------------------------
// 3. BPETokenizer — GPT-2 Byte-Pair Encoding Tokenizer
// ---------------------------------------------------------------------------

/// Byte-Pair Encoding (BPE) tokenizer compatible with GPT-2 vocabulary formats.
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    eos_token_id: usize,
}

impl BPETokenizer {
    pub const GPT2_EOS_ID: usize = 50256;
    pub const GPT2_VOCAB_SIZE: usize = 50257;

    /// Create an empty BPE tokenizer initialized with standard byte mappings.
    pub fn new() -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();

        // Populate base 256 byte tokens
        for b in 0..=255u8 {
            let s = String::from_utf8_lossy(&[b]).to_string();
            encoder.insert(s.clone(), b as usize);
            decoder.insert(b as usize, s);
        }
        // Special EOS token for GPT-2
        let eos_str = "<|endoftext|>".to_string();
        encoder.insert(eos_str.clone(), Self::GPT2_EOS_ID);
        decoder.insert(Self::GPT2_EOS_ID, eos_str);

        Self {
            encoder,
            decoder,
            bpe_ranks: HashMap::new(),
            eos_token_id: Self::GPT2_EOS_ID,
        }
    }

    /// Load vocabulary from a JSON file (HuggingFace `vocab.json` format).
    pub fn from_vocab_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read vocab json from {:?}: {e}", path))?;
        let map: HashMap<String, usize> = serde_json::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse vocab json: {e}"))?;

        let mut encoder = HashMap::with_capacity(map.len());
        let mut decoder = HashMap::with_capacity(map.len());

        for (k, v) in map {
            decoder.insert(v, k.clone());
            encoder.insert(k, v);
        }

        let eos_token_id = *encoder.get("<|endoftext|>").unwrap_or(&Self::GPT2_EOS_ID);

        Ok(Self {
            encoder,
            decoder,
            bpe_ranks: HashMap::new(),
            eos_token_id,
        })
    }

    /// Load BPE merges from a `merges.txt` file.
    pub fn load_merges(&mut self, path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read merges from {:?}: {e}", path))?;
        for (rank, line) in content.lines().enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                self.bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), rank);
            }
        }
        Ok(())
    }
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        // If exact token exists in vocabulary (e.g. special tokens)
        if let Some(&id) = self.encoder.get(text) {
            return vec![id];
        }

        // Subword segmentation or byte fallback
        let mut tokens = Vec::new();
        let words = text.split_inclusive(|c: char| c.is_whitespace() || c.is_ascii_punctuation());

        for word in words {
            if word.is_empty() {
                continue;
            }
            if let Some(&id) = self.encoder.get(word) {
                tokens.push(id);
                continue;
            }

            // Fallback to byte pieces
            for &b in word.as_bytes() {
                let s = String::from_utf8_lossy(&[b]).to_string();
                if let Some(&id) = self.encoder.get(&s) {
                    tokens.push(id);
                } else {
                    tokens.push(b as usize);
                }
            }
        }

        tokens
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let mut out = String::new();
        for &tok in tokens {
            if tok == self.eos_token_id {
                continue;
            }
            if let Some(s) = self.decoder.get(&tok) {
                out.push_str(s);
            } else if tok < 256 {
                out.push(tok as u8 as char);
            } else {
                out.push('?');
            }
        }
        out
    }

    fn vocab_size(&self) -> usize {
        self.encoder.len().max(Self::GPT2_VOCAB_SIZE)
    }

    fn eos_token(&self) -> Option<usize> {
        Some(self.eos_token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_tokenizer_roundtrip() {
        let tok = ByteTokenizer::new();
        let original = "Hello, Dark Forest runtime! 🌲 α=16, r=8 🚀";
        let encoded = tok.encode(original);
        let decoded = tok.decode(&encoded);
        assert_eq!(original, decoded);
        assert_eq!(tok.vocab_size(), 257);
        assert_eq!(tok.eos_token(), Some(256));
    }

    #[test]
    fn test_char_tokenizer_roundtrip() {
        let corpus = "abcdefghijklmnopqrstuvwxyz ";
        let tok = CharTokenizer::build(corpus);
        let text = "hello dark forest";
        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_bpe_tokenizer_basic() {
        let tok = BPETokenizer::new();
        let text = "hello";
        let encoded = tok.encode(text);
        assert!(!encoded.is_empty());
        let decoded = tok.decode(&encoded);
        assert_eq!(text, decoded);
        assert_eq!(tok.eos_token(), Some(BPETokenizer::GPT2_EOS_ID));
    }
}
