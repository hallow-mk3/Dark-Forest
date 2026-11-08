//! Distributed Cluster Coordination & Collective Communications
//!
//! Enables Dark Forest to scale from single-device workstations to multi-GPU nodes
//! and multi-machine high-performance compute (HPC) clusters.
//!
//! Subsystems:
//! - **Environment & Config (`ClusterConfig`)**: Slurm and torchrun environment variable autodetection
//! - **Transport Layer (`Transport`, `ChannelHub`, `TcpTransport`)**: In-memory and cross-node network sockets
//! - **Collective Operations (`collectives`)**: Bandwidth-optimal Ring All-Reduce, Broadcast, All-Gather, Reduce-Scatter
//! - **Distributed Data Parallel (`DistributedDataParallel`)**: Multi-node gradient averaging and parameter sync

pub mod collectives;
pub mod config;
pub mod ddp;
pub mod transport;

pub use collectives::{all_gather, all_reduce_ring, broadcast, reduce_scatter, ReduceOp};
pub use config::{ClusterBackend, ClusterConfig};
pub use ddp::DistributedDataParallel;
pub use transport::{ChannelHub, ChannelTransport, TcpTransport, Transport};

