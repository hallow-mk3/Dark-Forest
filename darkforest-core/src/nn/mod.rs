//! Neural network module layer abstractions — full PyTorch parity.

pub mod activation_modules;
pub mod attention;
pub mod batchnorm;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod linear;
pub mod lora;
pub mod pool;
pub mod quantization;
pub mod rnn;
pub mod sequential;
pub mod transformer;

pub use activation_modules::*;
pub use attention::MultiHeadAttention;
pub use batchnorm::{BatchNorm1d, BatchNorm2d};
pub use conv::{Conv1d, Conv2d};
pub use dropout::{Dropout, Dropout2d};
pub use embedding::{Embedding, PosEmbedding};
pub use linear::Linear;
pub use lora::{LoRAConfig, LoRALinear};
pub use pool::{AdaptiveAvgPool2d, AvgPool2d, MaxPool2d};
pub use quantization::{dequantize_nf4_cpu, quantize_nf4_cpu, QLoRALinear, NF4_TABLE};
pub use rnn::{GRUCell, LSTMCell, RNNCell, GRU, LSTM};
pub use sequential::Sequential;
pub use transformer::{GPT2Config, TransformerBlock, GPT2};
