//! Cluster Configuration & Environment Initialization
//!
//! Compatible with standard cluster schedulers and orchestrators:
//! - Slurm (`SLURM_PROCID`, `SLURM_NTASKS`)
//! - Torchrun / PyTorch Distributed (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`)
//! - MPI / Custom orchestrators

use anyhow::{bail, Result};
use std::env;

/// Backend communication mechanism used by the cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterBackend {
    /// In-memory crossbeam/thread channels (ultra-fast for local multi-device simulation & testing).
    Channel,
    /// High-performance TCP sockets for distributed multi-node LAN / cluster deployment.
    Tcp,
}

/// Node configuration within a distributed cluster.
#[derive(Clone, Debug)]
pub struct ClusterConfig {
    /// Global rank of this process `[0 .. world_size)`. Rank 0 is the master coordinator.
    pub rank: usize,
    /// Total number of processes participating in the cluster across all nodes.
    pub world_size: usize,
    /// Device / GPU index local to the current host machine.
    pub local_rank: usize,
    /// Hostname or IP address of the master coordination node (Rank 0).
    pub master_addr: String,
    /// Port of the master node.
    pub master_port: u16,
    /// Communication backend.
    pub backend: ClusterBackend,
}

impl ClusterConfig {
    /// Creates a manual configuration for a cluster process.
    pub fn new(
        rank: usize,
        world_size: usize,
        local_rank: usize,
        master_addr: impl Into<String>,
        master_port: u16,
        backend: ClusterBackend,
    ) -> Result<Self> {
        if world_size == 0 {
            bail!("world_size must be at least 1");
        }
        if rank >= world_size {
            bail!("rank {} out of bounds for world_size {}", rank, world_size);
        }
        Ok(Self {
            rank,
            world_size,
            local_rank,
            master_addr: master_addr.into(),
            master_port,
            backend,
        })
    }

    /// Automatically initializes cluster configuration from environment variables.
    /// Supports both PyTorch/torchrun (`RANK`, `WORLD_SIZE`) and Slurm (`SLURM_PROCID`).
    pub fn from_env() -> Result<Self> {
        let rank: usize = env::var("RANK")
            .or_else(|_| env::var("SLURM_PROCID"))
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .unwrap_or(0);

        let world_size: usize = env::var("WORLD_SIZE")
            .or_else(|_| env::var("SLURM_NTASKS"))
            .unwrap_or_else(|_| "1".to_string())
            .parse()
            .unwrap_or(1);

        let local_rank: usize = env::var("LOCAL_RANK")
            .or_else(|_| env::var("SLURM_LOCALID"))
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .unwrap_or(0);

        let master_addr = env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());
        let master_port: u16 = env::var("MASTER_PORT")
            .unwrap_or_else(|_| "29500".to_string())
            .parse()
            .unwrap_or(29500);

        let backend = if world_size > 1 && master_addr != "127.0.0.1" {
            ClusterBackend::Tcp
        } else {
            ClusterBackend::Channel
        };

        Self::new(rank, world_size, local_rank, master_addr, master_port, backend)
    }

    /// Returns true if this process is the master coordinator (Rank 0).
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }

    /// Master address string `host:port`.
    pub fn master_endpoint(&self) -> String {
        format!("{}:{}", self.master_addr, self.master_port)
    }
}

