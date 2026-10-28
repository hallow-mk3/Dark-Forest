//! SafeTensors serialization and deserialization for Dark Forest ML Runtime.
//!
//! Complies with the official SafeTensors format specification:
//! - 8-byte little-endian header length (N)
//! - N-byte JSON metadata mapping tensor names to dtype, shape, and byte offsets
//! - Contiguous raw binary tensor data buffers
//!
//! Includes HuggingFace GPT-2 weight translation and splitting:
//! - Handles `c_attn.weight` `[d_model, 3 * d_model]` splitting into `wq`, `wk`, `wv`
//! - Supports exporting Dark Forest models directly to `.safetensors` for zero-overhead interoperability.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use super::static_gpt2::StaticGPT2;

/// Metadata entry for a single tensor in a SafeTensors file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// In-memory representation of parsed SafeTensors tensors.
#[derive(Debug, Clone, Default)]
pub struct SafeTensorsFile {
    pub tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl SafeTensorsFile {
    /// Read and parse a `.safetensors` file from disk.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| anyhow!("Failed to open SafeTensors file {:?}: {e}", path))?;

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|e| anyhow!("Failed to read SafeTensors 8-byte header length: {e}"))?;
        let header_len = u64::from_le_bytes(len_bytes) as usize;

        if header_len > 100 * 1024 * 1024 {
            return Err(anyhow!("SafeTensors header excessively large: {header_len} bytes"));
        }

        let mut header_buf = vec![0u8; header_len];
        file.read_exact(&mut header_buf)
            .map_err(|e| anyhow!("Failed to read SafeTensors JSON header: {e}"))?;

        let header_json: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)
            .map_err(|e| anyhow!("Failed to deserialize SafeTensors JSON header: {e}"))?;

        let mut rest_of_file = Vec::new();
        file.read_to_end(&mut rest_of_file)
            .map_err(|e| anyhow!("Failed to read SafeTensors binary tensor payload: {e}"))?;

        let mut tensors = BTreeMap::new();

        for (name, value) in header_json {
            if name == "__metadata__" {
                continue;
            }

            let meta: TensorMeta = serde_json::from_value(value)
                .map_err(|e| anyhow!("Failed to parse tensor metadata for '{name}': {e}"))?;

            let start = meta.data_offsets[0];
            let end = meta.data_offsets[1];

            if end > rest_of_file.len() || start > end {
                return Err(anyhow!(
                    "Invalid byte offsets [{start}, {end}] for tensor '{name}' (payload len={})",
                    rest_of_file.len()
                ));
            }

            let byte_slice = &rest_of_file[start..end];
            let num_floats = byte_slice.len() / 4;
            let mut float_vec = Vec::with_capacity(num_floats);

            for chunk in byte_slice.chunks_exact(4) {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                float_vec.push(f32::from_le_bytes(bytes));
            }

            tensors.insert(name, (meta.shape, float_vec));
        }

        Ok(Self { tensors })
    }

    /// Save the tensors to a `.safetensors` file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| anyhow!("Failed to create SafeTensors file {:?}: {e}", path))?;
        let mut writer = BufWriter::new(file);

        let mut header_map = BTreeMap::new();
        let mut current_offset = 0usize;

        for (name, (shape, data)) in &self.tensors {
            let byte_len = data.len() * 4;
            let meta = TensorMeta {
                dtype: "F32".to_string(),
                shape: shape.clone(),
                data_offsets: [current_offset, current_offset + byte_len],
            };
            header_map.insert(name.clone(), meta);
            current_offset += byte_len;
        }

        let header_json = serde_json::to_string(&header_map)
            .map_err(|e| anyhow!("Failed to serialize SafeTensors header to JSON: {e}"))?;
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(header_bytes)?;

        for (_name, (_shape, data)) in &self.tensors {
            for &val in data {
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Lookup tensor by name, trying common model prefixes (`transformer.`, `model.`, or bare).
    pub fn get_tensor(&self, name: &str) -> Option<&(Vec<usize>, Vec<f32>)> {
        if let Some(t) = self.tensors.get(name) {
            return Some(t);
        }
        let prefixed = format!("transformer.{name}");
        if let Some(t) = self.tensors.get(&prefixed) {
            return Some(t);
        }
        let model_prefixed = format!("model.{name}");
        if let Some(t) = self.tensors.get(&model_prefixed) {
            return Some(t);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// HuggingFace GPT-2 SafeTensors Weight Importer / Exporter
// ---------------------------------------------------------------------------

/// Import weights from a HuggingFace GPT-2 `.safetensors` file into `StaticGPT2`.
pub fn load_gpt2_safetensors(model: &mut StaticGPT2, path: &Path) -> Result<()> {
    let sf = SafeTensorsFile::open(path)?;

    let d_model = model.d_model();
    let n_layers = model.n_layers();

    // 1. Embeddings
    if let Some((_, data)) = sf.get_tensor("wte.weight") {
        model.set_tok_emb(data)?;
    }
    if let Some((_, data)) = sf.get_tensor("wpe.weight") {
        model.set_pos_emb(data)?;
    }

    // 2. Final LayerNorm
    if let Some((_, data)) = sf.get_tensor("ln_f.weight") {
        model.set_ln_f_gamma(data)?;
    }
    if let Some((_, data)) = sf.get_tensor("ln_f.bias") {
        model.set_ln_f_beta(data)?;
    }

    // 3. LM Head (optional, tied to wte if not present)
    if let Some((_, data)) = sf.get_tensor("lm_head.weight") {
        model.set_lm_head(data)?;
    } else if let Some((_, data)) = sf.get_tensor("wte.weight") {
        model.set_lm_head(data)?;
    }

    // 4. Per-layer transformer blocks
    for l in 0..n_layers {
        // LN1
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.ln_1.weight")) {
            model.set_layer_ln1_gamma(l, data)?;
        }
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.ln_1.bias")) {
            model.set_layer_ln1_beta(l, data)?;
        }

        // Attention QKV: HuggingFace packs [d_model, 3 * d_model]
        // or [3 * d_model, d_model] depending on transpose
        if let Some((shape, data)) = sf.get_tensor(&format!("h.{l}.attn.c_attn.weight")) {
            let (wq, wk, wv) = split_qkv_weights(data, shape, d_model)?;
            model.set_layer_wq(l, &wq)?;
            model.set_layer_wk(l, &wk)?;
            model.set_layer_wv(l, &wv)?;
        }

        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.attn.c_attn.bias")) {
            let (bq, bk, bv) = split_qkv_bias(data, d_model)?;
            model.set_layer_bq(l, &bq)?;
            model.set_layer_bk(l, &bk)?;
            model.set_layer_bv(l, &bv)?;
        }

        // Attention output projection
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.attn.c_proj.weight")) {
            model.set_layer_wo(l, data)?;
        }
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.attn.c_proj.bias")) {
            model.set_layer_bo(l, data)?;
        }

        // LN2
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.ln_2.weight")) {
            model.set_layer_ln2_gamma(l, data)?;
        }
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.ln_2.bias")) {
            model.set_layer_ln2_beta(l, data)?;
        }

        // MLP
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.mlp.c_fc.weight")) {
            model.set_layer_w1(l, data)?;
        }
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.mlp.c_fc.bias")) {
            model.set_layer_b1(l, data)?;
        }
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.mlp.c_proj.weight")) {
            model.set_layer_w2(l, data)?;
        }
        if let Some((_, data)) = sf.get_tensor(&format!("h.{l}.mlp.c_proj.bias")) {
            model.set_layer_b2(l, data)?;
        }
    }

    Ok(())
}

/// Export `StaticGPT2` weights into a standard HuggingFace-compatible `.safetensors` file.
pub fn save_gpt2_safetensors(model: &StaticGPT2, path: &Path) -> Result<()> {
    let mut sf = SafeTensorsFile::default();

    let d_model = model.d_model();
    let vocab_size = model.vocab_size();
    let seq_len = model.seq_len();
    let d_ff = model.d_ff();
    let n_layers = model.n_layers();

    // Global weights
    sf.tensors.insert(
        "transformer.wte.weight".to_string(),
        (vec![vocab_size, d_model], model.get_tok_emb()?),
    );
    sf.tensors.insert(
        "transformer.wpe.weight".to_string(),
        (vec![seq_len, d_model], model.get_pos_emb()?),
    );
    sf.tensors.insert(
        "transformer.ln_f.weight".to_string(),
        (vec![d_model], model.get_ln_f_gamma()?),
    );
    sf.tensors.insert(
        "transformer.ln_f.bias".to_string(),
        (vec![d_model], model.get_ln_f_beta()?),
    );
    sf.tensors.insert(
        "lm_head.weight".to_string(),
        (vec![vocab_size, d_model], model.get_lm_head()?),
    );

    // Layer weights
    for l in 0..n_layers {
        sf.tensors.insert(
            format!("transformer.h.{l}.ln_1.weight"),
            (vec![d_model], model.get_layer_ln1_gamma(l)?),
        );
        sf.tensors.insert(
            format!("transformer.h.{l}.ln_1.bias"),
            (vec![d_model], model.get_layer_ln1_beta(l)?),
        );

        // Pack Q, K, V back into packed [d_model, 3 * d_model]
        let wq = model.get_layer_wq(l)?;
        let wk = model.get_layer_wk(l)?;
        let wv = model.get_layer_wv(l)?;
        let mut packed_w = Vec::with_capacity(3 * d_model * d_model);
        for row in 0..d_model {
            packed_w.extend_from_slice(&wq[row * d_model..(row + 1) * d_model]);
            packed_w.extend_from_slice(&wk[row * d_model..(row + 1) * d_model]);
            packed_w.extend_from_slice(&wv[row * d_model..(row + 1) * d_model]);
        }
        sf.tensors.insert(
            format!("transformer.h.{l}.attn.c_attn.weight"),
            (vec![d_model, 3 * d_model], packed_w),
        );

        let mut packed_b = Vec::with_capacity(3 * d_model);
        packed_b.extend(model.get_layer_bq(l)?);
        packed_b.extend(model.get_layer_bk(l)?);
        packed_b.extend(model.get_layer_bv(l)?);
        sf.tensors.insert(
            format!("transformer.h.{l}.attn.c_attn.bias"),
            (vec![3 * d_model], packed_b),
        );

        sf.tensors.insert(
            format!("transformer.h.{l}.attn.c_proj.weight"),
            (vec![d_model, d_model], model.get_layer_wo(l)?),
        );
        sf.tensors.insert(
            format!("transformer.h.{l}.attn.c_proj.bias"),
            (vec![d_model], model.get_layer_bo(l)?),
        );

        sf.tensors.insert(
            format!("transformer.h.{l}.ln_2.weight"),
            (vec![d_model], model.get_layer_ln2_gamma(l)?),
        );
        sf.tensors.insert(
            format!("transformer.h.{l}.ln_2.bias"),
            (vec![d_model], model.get_layer_ln2_beta(l)?),
        );

        sf.tensors.insert(
            format!("transformer.h.{l}.mlp.c_fc.weight"),
            (vec![d_ff, d_model], model.get_layer_w1(l)?),
        );
        sf.tensors.insert(
            format!("transformer.h.{l}.mlp.c_fc.bias"),
            (vec![d_ff], model.get_layer_b1(l)?),
        );
        sf.tensors.insert(
            format!("transformer.h.{l}.mlp.c_proj.weight"),
            (vec![d_model, d_ff], model.get_layer_w2(l)?),
        );
        sf.tensors.insert(
            format!("transformer.h.{l}.mlp.c_proj.bias"),
            (vec![d_model], model.get_layer_b2(l)?),
        );
    }

    sf.save(path)
}

// ---------------------------------------------------------------------------
// Helpers: Split HuggingFace fused QKV projections
// ---------------------------------------------------------------------------

fn split_qkv_weights(
    data: &[f32],
    shape: &[usize],
    d_model: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if data.len() != 3 * d_model * d_model {
        return Err(anyhow!(
            "Packed QKV tensor size mismatch: expected {}, got {}",
            3 * d_model * d_model,
            data.len()
        ));
    }

    let mut wq = Vec::with_capacity(d_model * d_model);
    let mut wk = Vec::with_capacity(d_model * d_model);
    let mut wv = Vec::with_capacity(d_model * d_model);

    // If HF standard shape is [d_model, 3 * d_model]
    if shape.len() == 2 && shape[0] == d_model && shape[1] == 3 * d_model {
        for row in 0..d_model {
            let row_offset = row * (3 * d_model);
            wq.extend_from_slice(&data[row_offset..row_offset + d_model]);
            wk.extend_from_slice(&data[row_offset + d_model..row_offset + 2 * d_model]);
            wv.extend_from_slice(&data[row_offset + 2 * d_model..row_offset + 3 * d_model]);
        }
    } else {
        // Linear block split fallback [3 * d_model, d_model]
        let chunk = d_model * d_model;
        wq.extend_from_slice(&data[0..chunk]);
        wk.extend_from_slice(&data[chunk..2 * chunk]);
        wv.extend_from_slice(&data[2 * chunk..3 * chunk]);
    }

    Ok((wq, wk, wv))
}

fn split_qkv_bias(data: &[f32], d_model: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if data.len() != 3 * d_model {
        return Err(anyhow!(
            "Packed QKV bias size mismatch: expected {}, got {}",
            3 * d_model,
            data.len()
        ));
    }

    let bq = data[0..d_model].to_vec();
    let bk = data[d_model..2 * d_model].to_vec();
    let bv = data[2 * d_model..3 * d_model].to_vec();

    Ok((bq, bk, bv))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_file_save_and_reload() {
        let mut sf = SafeTensorsFile::default();
        let original_data = vec![1.0f32, 2.5, -3.14, 42.0];
        sf.tensors.insert(
            "test_tensor".to_string(),
            (vec![2, 2], original_data.clone()),
        );

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("darkforest_test.safetensors");

        sf.save(&file_path).unwrap();
        assert!(file_path.exists());

        let loaded = SafeTensorsFile::open(&file_path).unwrap();
        let (shape, data) = loaded.get_tensor("test_tensor").unwrap();
        assert_eq!(shape, &[2, 2]);
        assert_eq!(data, &original_data);

        let _ = std::fs::remove_file(file_path);
    }
}
