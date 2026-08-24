//! Neural network module layer abstractions.

pub mod attention;
pub mod embedding;
pub mod linear;
pub mod transformer;

pub use attention::MultiHeadAttention;
pub use embedding::{Embedding, PosEmbedding};
pub use linear::Linear;
pub use transformer::{GPT2Config, TransformerBlock, GPT2};
