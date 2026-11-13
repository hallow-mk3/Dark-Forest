//! Distributed Cluster Integration Tests (Slurm, Torchrun, Multi-Node Ring All-Reduce)

use darkforest_core::cluster::*;
use darkforest_core::autograd::Value;
use darkforest_core::Tensor;
use std::sync::Arc;
use std::thread;

#[test]
fn test_cluster_environment_variable_detection() {
    std::env::set_var("RANK", "3");
    std::env::set_var("WORLD_SIZE", "8");
    std::env::set_var("LOCAL_RANK", "1");
    std::env::set_var("MASTER_ADDR", "10.0.0.1");
    std::env::set_var("MASTER_PORT", "29500");

    let cfg = ClusterConfig::from_env().unwrap();
    assert_eq!(cfg.rank, 3);
    assert_eq!(cfg.world_size, 8);
    assert_eq!(cfg.local_rank, 1);
    assert_eq!(cfg.master_addr, "10.0.0.1");
    assert_eq!(cfg.master_port, 29500);
    assert!(!cfg.is_master());
    assert_eq!(cfg.master_endpoint(), "10.0.0.1:29500");
}

#[test]
fn test_cluster_8_node_ring_all_reduce_scaling() {
    let world_size = 8;
    let hub = ChannelHub::new(world_size);
    let num_elements = 1024; // 1024-dim parameter vector per rank

    let mut handles = Vec::new();

    for rank in 0..world_size {
        let hub_clone = Arc::clone(&hub);
        let handle = thread::spawn(move || {
            let transport = ChannelHub::connect_rank(hub_clone, rank);
            // Each node initializes buffer with rank value
            let mut data = vec![rank as f32; num_elements];

            all_reduce_ring(&transport, &mut data, ReduceOp::Sum).unwrap();
            data
        });
        handles.push(handle);
    }

    // Sum of 0..7 = 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28.0
    for h in handles {
        let reduced = h.join().unwrap();
        assert_eq!(reduced.len(), num_elements);
        for &val in &reduced {
            assert_eq!(val, 28.0);
        }
    }
}

#[test]
fn test_multi_node_ddp_training_step() {
    let world_size = 4;
    let hub = ChannelHub::new(world_size);
    let mut handles = Vec::new();

    for rank in 0..world_size {
        let hub_clone = Arc::clone(&hub);
        let handle = thread::spawn(move || {
            let transport: Arc<dyn Transport> = Arc::new(ChannelHub::connect_rank(hub_clone, rank));

            // Each rank holds a copy of model weights: shape [3]
            let weights = Value::leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap());

            // Simulate local micro-batch gradient computation on rank's local shard:
            // Rank 0: [1.0, 2.0, 3.0]
            // Rank 1: [2.0, 4.0, 6.0]
            // Rank 2: [3.0, 6.0, 9.0]
            // Rank 3: [4.0, 8.0, 12.0]
            let factor = (rank + 1) as f32;
            let local_grads = Tensor::from_vec(vec![1.0 * factor, 2.0 * factor, 3.0 * factor], vec![3]).unwrap();
            weights.accumulate_grad(&local_grads);

            // DDP Synchronizes gradients across the entire 4-node cluster using Ring All-Reduce Mean
            let mut ddp = DistributedDataParallel::new(transport, vec![weights.clone()], false).unwrap();
            ddp.sync_gradients().unwrap();

            // Average gradients:
            // sum = [10.0, 20.0, 30.0] / 4 = [2.5, 5.0, 7.5]
            weights.grad()
        });
        handles.push(handle);
    }

    for h in handles {
        let averaged_grads = h.join().unwrap();
        assert_eq!(averaged_grads, vec![2.5, 5.0, 7.5]);
    }
}

