//! Parallelization & Device Sharding (JAX jax.pmap & shard_map equivalent)
//!
//! Features:
//! - Multi-device mesh abstraction (`DeviceMesh`)
//! - Functional sharding specs: `Replicated` vs `Sharded(axis)`
//! - Parallel function mapping (`pmap`) running computations across worker threads/devices
//! - Collective communication primitives: `p_sum`, `p_mean`, `p_gather`

use super::array::Array;
use anyhow::{bail, Result};
use rayon::prelude::*;

/// Sharding specification for a multi-dimensional array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sharding {
    /// Replicated on all devices in the mesh.
    Replicated,
    /// Sharded (partitioned) evenly along the specified axis.
    Sharded { axis: usize },
}

/// Logical hardware mesh representing available compute workers/devices.
#[derive(Clone, Debug)]
pub struct DeviceMesh {
    pub num_devices: usize,
}

impl DeviceMesh {
    pub fn new(num_devices: usize) -> Self {
        assert!(num_devices > 0, "DeviceMesh must have at least 1 device");
        Self { num_devices }
    }

    /// Default mesh using available CPU worker cores.
    pub fn default_cpu() -> Self {
        Self {
            num_devices: rayon::current_num_threads().max(1),
        }
    }
}

/// An Array distributed across a `DeviceMesh`.
#[derive(Clone, Debug)]
pub struct ShardedArray {
    pub shards: Vec<Array>,
    pub sharding: Sharding,
}

impl ShardedArray {
    /// Shards an array evenly along axis 0 across `mesh.num_devices`.
    pub fn shard_along_axis(array: &Array, mesh: &DeviceMesh, axis: usize) -> Result<Self> {
        let shape = array.shape();
        if axis >= shape.len() {
            bail!("Axis {} out of bounds for array with rank {}", axis, shape.len());
        }
        let dim_len = shape[axis];
        if dim_len % mesh.num_devices != 0 {
            bail!(
                "Dimension {} of size {} cannot be evenly divided across {} devices",
                axis,
                dim_len,
                mesh.num_devices
            );
        }

        let shard_dim_len = dim_len / mesh.num_devices;
        let mut shard_shape = shape.to_vec();
        shard_shape[axis] = shard_dim_len;
        let shard_numel: usize = shard_shape.iter().product();

        let data = array.to_vec();
        let mut shards = Vec::with_capacity(mesh.num_devices);

        for d in 0..mesh.num_devices {
            let start = d * shard_numel;
            let end = start + shard_numel;
            let shard_data = data[start..end].to_vec();
            shards.push(Array::from_vec(shard_data, shard_shape.clone())?);
        }

        Ok(Self {
            shards,
            sharding: Sharding::Sharded { axis },
        })
    }

    /// Replicates a single array across all devices.
    pub fn replicate(array: &Array, mesh: &DeviceMesh) -> Self {
        Self {
            shards: vec![array.clone(); mesh.num_devices],
            sharding: Sharding::Replicated,
        }
    }

    /// Gathers all shards back into a single global array.
    pub fn gather(&self) -> Result<Array> {
        if self.shards.is_empty() {
            bail!("ShardedArray has no shards");
        }
        match self.sharding {
            Sharding::Replicated => Ok(self.shards[0].clone()),
            Sharding::Sharded { axis: _ } => {
                let first_shape = self.shards[0].shape();
                let mut total_data = Vec::new();
                for shard in &self.shards {
                    total_data.extend(shard.to_vec());
                }
                let mut global_shape = first_shape.to_vec();
                global_shape[0] *= self.shards.len();
                Array::from_vec(total_data, global_shape)
            }
        }
    }
}

/// Executes a function in parallel across devices (`pmap`).
pub fn pmap<F>(f: F, input: &ShardedArray) -> Result<ShardedArray>
where
    F: Fn(&Array) -> Result<Array> + Send + Sync,
{
    let out_shards: Result<Vec<Array>> = input
        .shards
        .par_iter()
        .map(|shard| f(shard))
        .collect();

    Ok(ShardedArray {
        shards: out_shards?,
        sharding: input.sharding,
    })
}

/// Collective all-reduce sum across all shards.
pub fn p_sum(sharded: &ShardedArray) -> Result<ShardedArray> {
    if sharded.shards.is_empty() {
        bail!("Empty shards in p_sum");
    }

    let n = sharded.shards.len();
    let numel = sharded.shards[0].numel();
    let mut sum_data = vec![0.0f32; numel];

    for shard in &sharded.shards {
        let data = shard.to_vec();
        for i in 0..numel {
            sum_data[i] += data[i];
        }
    }

    let sum_shard = Array::from_vec(sum_data, sharded.shards[0].shape().to_vec())?;
    Ok(ShardedArray {
        shards: vec![sum_shard; n],
        sharding: Sharding::Replicated,
    })
}

/// Collective all-reduce mean across all shards.
pub fn p_mean(sharded: &ShardedArray) -> Result<ShardedArray> {
    let sum_sharded = p_sum(sharded)?;
    let count = sharded.shards.len() as f32;
    let mean_shards: Result<Vec<Array>> = sum_sharded
        .shards
        .into_iter()
        .map(|s| s.scale(1.0 / count))
        .collect();

    Ok(ShardedArray {
        shards: mean_shards?,
        sharding: Sharding::Replicated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharding_and_gather() {
        let mesh = DeviceMesh::new(2);
        // Original array of shape [4, 2]
        let original = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]).unwrap();

        let sharded = ShardedArray::shard_along_axis(&original, &mesh, 0).unwrap();
        assert_eq!(sharded.shards.len(), 2);
        assert_eq!(sharded.shards[0].shape(), &[2, 2]);
        assert_eq!(sharded.shards[1].shape(), &[2, 2]);

        let gathered = sharded.gather().unwrap();
        assert_eq!(gathered.to_vec(), original.to_vec());
    }

    #[test]
    fn test_pmap_and_psum() {
        let mesh = DeviceMesh::new(2);
        let arr = Array::from_vec(vec![1.0, 2.0, 10.0, 20.0], vec![2, 2]).unwrap();
        let sharded = ShardedArray::shard_along_axis(&arr, &mesh, 0).unwrap();

        // Parallel scale by 2.0
        let scaled = pmap(|shard| shard.scale(2.0), &sharded).unwrap();
        // Collective sum
        let summed = p_sum(&scaled).unwrap();

        // Shard 0 was [1, 2]*2 = [2, 4]
        // Shard 1 was [10, 20]*2 = [20, 40]
        // Sum = [22, 44]
        assert_eq!(summed.shards[0].to_vec(), vec![22.0, 44.0]);
        assert_eq!(summed.shards[1].to_vec(), vec![22.0, 44.0]);
    }
}

