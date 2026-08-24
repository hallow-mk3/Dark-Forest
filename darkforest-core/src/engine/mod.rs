//! Static execution engine — bypasses `Value`/Arc/Mutex for maximum throughput.
//!
//! `StaticGPT2` stores all weights and pre-allocated activation/gradient buffers
//! as raw `DeviceTensor` handles. Forward and backward are implemented as direct
//! CUDA kernel sequences: zero dynamic allocation, zero graph construction,
//! zero mutex overhead per step.

pub mod static_gpt2;
pub use static_gpt2::StaticGPT2;
