//! High-Performance Distributed Collective Communication Primitives
//!
//! Features:
//! - **Ring All-Reduce**: Bandwidth-optimal 2-phase ring algorithm transferring exactly $2 \frac{W-1}{W} N$ elements
//! - **Broadcast**: Coordinates master parameters (Rank 0) across all cluster nodes
//! - **All-Gather**: Collects distributed tensor chunks into full contiguous tensors
//! - **Reduce-Scatter**: Reduces tensors across ranks while scattering equal partitions to each node

use super::transport::Transport;
use anyhow::{bail, Result};

/// Collective reduction operation type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// Executes a bandwidth-optimal **Ring All-Reduce** across all cluster ranks.
///
/// Algorithm (Baidu / NCCL Ring All-Reduce):
/// 1. Array of size $N$ is partitioned into $W$ chunks (where $W = \text{world\_size}$).
/// 2. **Scatter-Reduce (Phase 1)**: For $W - 1$ steps, rank $r$ sends chunk $c$ to rank $(r+1) \pmod W$
///    and receives chunk $c-1$ from $(r-1) \pmod W$, accumulating incoming data into its local chunk.
///    After $W - 1$ steps, each rank holds the complete reduction for 1 chunk.
/// 3. **All-Gather (Phase 2)**: For $W - 1$ steps, each rank circulates its fully-reduced chunk around
///    the ring so all ranks receive all reduced chunks.
///
/// Total communication volume: $2 \times \frac{W-1}{W} N$, which is asymptotically optimal!
pub fn all_reduce_ring<T: Transport + ?Sized>(
    transport: &T,
    data: &mut [f32],
    op: ReduceOp,
) -> Result<()> {
    let w = transport.world_size();
    let r = transport.rank();

    if w <= 1 {
        return Ok(());
    }

    let n = data.len();
    if n == 0 {
        return Ok(());
    }

    // Partition array into w equal-sized chunks (padded if needed)
    let chunk_size = (n + w - 1) / w;
    let padded_len = chunk_size * w;
    let mut padded_data = if n < padded_len {
        let mut p = data.to_vec();
        p.resize(padded_len, 0.0);
        p
    } else {
        data.to_vec()
    };

    let next_rank = (r + 1) % w;
    let prev_rank = (r + w - 1) % w;

    // ------------------------------------------------------------------------
    // Phase 1: Scatter-Reduce (W - 1 steps)
    // ------------------------------------------------------------------------
    for step in 0..(w - 1) {
        let send_chunk_idx = (r + 2 * w - step) % w;
        let recv_chunk_idx = (r + 2 * w - step - 1) % w;

        let send_start = send_chunk_idx * chunk_size;
        let send_slice = &padded_data[send_start..send_start + chunk_size];

        // Send to next_rank and recv from prev_rank
        transport.send_to(next_rank, send_slice)?;
        let received = transport.recv_from(prev_rank, chunk_size)?;

        let recv_start = recv_chunk_idx * chunk_size;
        let target_slice = &mut padded_data[recv_start..recv_start + chunk_size];

        for i in 0..chunk_size {
            match op {
                ReduceOp::Sum | ReduceOp::Mean => target_slice[i] += received[i],
                ReduceOp::Max => target_slice[i] = target_slice[i].max(received[i]),
                ReduceOp::Min => target_slice[i] = target_slice[i].min(received[i]),
            }
        }
    }

    // ------------------------------------------------------------------------
    // Phase 2: All-Gather (W - 1 steps)
    // ------------------------------------------------------------------------
    for step in 0..(w - 1) {
        let send_chunk_idx = (r + 2 * w - step + 1) % w;
        let recv_chunk_idx = (r + 2 * w - step) % w;

        let send_start = send_chunk_idx * chunk_size;
        let send_slice = &padded_data[send_start..send_start + chunk_size];

        transport.send_to(next_rank, send_slice)?;
        let received = transport.recv_from(prev_rank, chunk_size)?;

        let recv_start = recv_chunk_idx * chunk_size;
        padded_data[recv_start..recv_start + chunk_size].copy_from_slice(&received);
    }

    // If mean reduction, divide by world_size
    if op == ReduceOp::Mean {
        let scale = 1.0 / (w as f32);
        for x in &mut padded_data {
            *x *= scale;
        }
    }

    // Copy back into original data buffer
    data.copy_from_slice(&padded_data[..n]);
    Ok(())
}

/// Broadcasts a tensor from `root_rank` to all other ranks in the cluster.
pub fn broadcast<T: Transport + ?Sized>(
    transport: &T,
    data: &mut [f32],
    root_rank: usize,
) -> Result<()> {
    let w = transport.world_size();
    let r = transport.rank();

    if w <= 1 {
        return Ok(());
    }

    if root_rank >= w {
        bail!("root_rank {} out of bounds", root_rank);
    }

    if r == root_rank {
        // Root sends to all other ranks
        for dst in 0..w {
            if dst != root_rank {
                transport.send_to(dst, data)?;
            }
        }
    } else {
        // Other ranks receive from root
        let received = transport.recv_from(root_rank, data.len())?;
        data.copy_from_slice(&received);
    }

    Ok(())
}

/// Gathers equal-sized chunks from all ranks and concatenates them into a single global tensor.
pub fn all_gather<T: Transport + ?Sized>(
    transport: &T,
    local_chunk: &[f32],
) -> Result<Vec<f32>> {
    let w = transport.world_size();
    let r = transport.rank();

    if w <= 1 {
        return Ok(local_chunk.to_vec());
    }

    let chunk_len = local_chunk.len();
    let mut full_buffer = vec![0.0f32; chunk_len * w];

    // Place local slice
    let local_start = r * chunk_len;
    full_buffer[local_start..local_start + chunk_len].copy_from_slice(local_chunk);

    // Exchange with all peers
    for peer in 0..w {
        if peer != r {
            transport.send_to(peer, local_chunk)?;
        }
    }

    for peer in 0..w {
        if peer != r {
            let received = transport.recv_from(peer, chunk_len)?;
            let start = peer * chunk_len;
            full_buffer[start..start + chunk_len].copy_from_slice(&received);
        }
    }

    Ok(full_buffer)
}

/// Reduces tensors across all ranks and scatters a $1/W$ partitioned slice to each rank.
pub fn reduce_scatter<T: Transport + ?Sized>(
    transport: &T,
    data: &[f32],
    op: ReduceOp,
) -> Result<Vec<f32>> {
    let mut reduced = data.to_vec();
    all_reduce_ring(transport, &mut reduced, op)?;

    let w = transport.world_size();
    let r = transport.rank();
    let chunk_size = (reduced.len() + w - 1) / w;

    let start = r * chunk_size;
    let end = (start + chunk_size).min(reduced.len());

    Ok(reduced[start..end].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::transport::ChannelHub;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_ring_all_reduce_sum_4_ranks() {
        let world_size = 4;
        let hub = ChannelHub::new(world_size);
        let len = 8; // 8 floats per rank

        // Rank 0: [1, 1, 1, 1, 1, 1, 1, 1]
        // Rank 1: [2, 2, 2, 2, 2, 2, 2, 2]
        // Rank 2: [3, 3, 3, 3, 3, 3, 3, 3]
        // Rank 3: [4, 4, 4, 4, 4, 4, 4, 4]
        // Expected Sum: [10, 10, 10, 10, 10, 10, 10, 10]
        let mut handles = Vec::new();

        for rank in 0..world_size {
            let hub_clone = Arc::clone(&hub);
            let handle = thread::spawn(move || {
                let transport = ChannelHub::connect_rank(hub_clone, rank);
                let val = (rank + 1) as f32;
                let mut data = vec![val; len];

                all_reduce_ring(&transport, &mut data, ReduceOp::Sum).unwrap();
                data
            });
            handles.push(handle);
        }

        for h in handles {
            let res = h.join().unwrap();
            assert_eq!(res, vec![10.0; len]);
        }
    }

    #[test]
    fn test_ring_all_reduce_mean_4_ranks() {
        let world_size = 4;
        let hub = ChannelHub::new(world_size);
        let len = 4;

        let mut handles = Vec::new();
        for rank in 0..world_size {
            let hub_clone = Arc::clone(&hub);
            let handle = thread::spawn(move || {
                let transport = ChannelHub::connect_rank(hub_clone, rank);
                let val = (rank + 1) as f32;
                let mut data = vec![val; len];

                all_reduce_ring(&transport, &mut data, ReduceOp::Mean).unwrap();
                data
            });
            handles.push(handle);
        }

        // Mean of [1, 2, 3, 4] = 2.5
        for h in handles {
            let res = h.join().unwrap();
            assert_eq!(res, vec![2.5; len]);
        }
    }

    #[test]
    fn test_broadcast_from_master() {
        let world_size = 3;
        let hub = ChannelHub::new(world_size);
        let mut handles = Vec::new();

        for rank in 0..world_size {
            let hub_clone = Arc::clone(&hub);
            let handle = thread::spawn(move || {
                let transport = ChannelHub::connect_rank(hub_clone, rank);
                let mut data = if rank == 0 {
                    vec![42.0, 99.0, 1337.0]
                } else {
                    vec![0.0, 0.0, 0.0]
                };

                broadcast(&transport, &mut data, 0).unwrap();
                data
            });
            handles.push(handle);
        }

        for h in handles {
            let res = h.join().unwrap();
            assert_eq!(res, vec![42.0, 99.0, 1337.0]);
        }
    }
}

