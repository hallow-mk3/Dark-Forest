//! Distributed Cluster Transport Layer
//!
//! Provides point-to-point and peer-to-peer transport primitives for cluster nodes:
//! - `ChannelTransport`: In-memory multi-threaded message queues for local testing and multi-device nodes
//! - `TcpTransport`: Cross-node TCP network sockets for multi-machine physical clusters

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Abstract peer-to-peer transport between cluster ranks.
pub trait Transport: Send + Sync {
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
    fn send_to(&self, dest_rank: usize, data: &[f32]) -> Result<()>;
    fn recv_from(&self, src_rank: usize, expected_len: usize) -> Result<Vec<f32>>;
    fn barrier(&self) -> Result<()>;
}

// ============================================================================
// ChannelTransport (In-Memory Channel Hub for Fast Multi-Worker Simulation)
// ============================================================================

/// Shared message bus for local cluster simulation.
pub struct ChannelHub {
    pub world_size: usize,
    senders: HashMap<(usize, usize), Sender<Vec<f32>>>,
    receivers: HashMap<(usize, usize), Mutex<Receiver<Vec<f32>>>>,
    barrier_tx: HashMap<usize, Sender<()>>,
    barrier_rx: HashMap<usize, Mutex<Receiver<()>>>,
    barrier_rel_tx: HashMap<usize, Sender<()>>,
    barrier_rel_rx: HashMap<usize, Mutex<Receiver<()>>>,
}

impl ChannelHub {
    /// Creates a complete interconnected hub for `world_size` processes.
    pub fn new(world_size: usize) -> Arc<Self> {
        let mut senders = HashMap::new();
        let mut receivers = HashMap::new();
        let mut barrier_tx = HashMap::new();
        let mut barrier_rx = HashMap::new();
        let mut barrier_rel_tx = HashMap::new();
        let mut barrier_rel_rx = HashMap::new();

        for src in 0..world_size {
            for dst in 0..world_size {
                let (tx, rx) = channel();
                senders.insert((src, dst), tx);
                receivers.insert((src, dst), Mutex::new(rx));
            }
            let (b_tx, b_rx) = channel();
            barrier_tx.insert(src, b_tx);
            barrier_rx.insert(src, Mutex::new(b_rx));

            let (br_tx, br_rx) = channel();
            barrier_rel_tx.insert(src, br_tx);
            barrier_rel_rx.insert(src, Mutex::new(br_rx));
        }

        Arc::new(Self {
            world_size,
            senders,
            receivers,
            barrier_tx,
            barrier_rx,
            barrier_rel_tx,
            barrier_rel_rx,
        })
    }

    /// Connects a specific rank to this hub.
    pub fn connect_rank(hub: Arc<Self>, rank: usize) -> ChannelTransport {
        assert!(rank < hub.world_size, "Rank out of range");
        ChannelTransport { hub, rank }
    }
}

/// Transport implementation backed by `ChannelHub`.
pub struct ChannelTransport {
    hub: Arc<ChannelHub>,
    rank: usize,
}

impl Transport for ChannelTransport {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.hub.world_size
    }

    fn send_to(&self, dest_rank: usize, data: &[f32]) -> Result<()> {
        if dest_rank >= self.hub.world_size {
            bail!("dest_rank {} out of bounds", dest_rank);
        }
        let key = (self.rank, dest_rank);
        let tx = self
            .hub
            .senders
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("Sender not found for key {:?}", key))?;
        tx.send(data.to_vec())
            .map_err(|e| anyhow::anyhow!("Failed to send to rank {}: {}", dest_rank, e))
    }

    fn recv_from(&self, src_rank: usize, expected_len: usize) -> Result<Vec<f32>> {
        if src_rank >= self.hub.world_size {
            bail!("src_rank {} out of bounds", src_rank);
        }
        let key = (src_rank, self.rank);
        let rx_mutex = self
            .hub
            .receivers
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("Receiver not found for key {:?}", key))?;
        let rx = rx_mutex.lock().unwrap();
        let buf = rx
            .recv()
            .map_err(|e| anyhow::anyhow!("Failed to recv from rank {}: {}", src_rank, e))?;
        if buf.len() != expected_len {
            bail!("Expected {} floats from rank {}, got {}", expected_len, src_rank, buf.len());
        }
        Ok(buf)
    }

    fn barrier(&self) -> Result<()> {
        if self.hub.world_size <= 1 {
            return Ok(());
        }
        if self.rank == 0 {
            for r in 1..self.hub.world_size {
                if let Some(rx_mutex) = self.hub.barrier_rx.get(&r) {
                    let rx = rx_mutex.lock().unwrap();
                    rx.recv_timeout(Duration::from_secs(5))
                        .map_err(|e| anyhow::anyhow!("Barrier timeout waiting for rank {}: {}", r, e))?;
                }
            }
            // Release all waiting worker ranks
            for r in 1..self.hub.world_size {
                if let Some(tx) = self.hub.barrier_rel_tx.get(&r) {
                    let _ = tx.send(());
                }
            }
        } else {
            // Signal arrival to rank 0
            if let Some(tx) = self.hub.barrier_tx.get(&self.rank) {
                tx.send(()).map_err(|e| anyhow::anyhow!("Barrier send failed: {}", e))?;
            }
            // Await release notification from rank 0
            if let Some(rx_mutex) = self.hub.barrier_rel_rx.get(&self.rank) {
                let rx = rx_mutex.lock().unwrap();
                rx.recv_timeout(Duration::from_secs(5))
                    .map_err(|e| anyhow::anyhow!("Barrier timeout awaiting release from rank 0: {}", e))?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// TcpTransport (Physical Cross-Node Sockets)
// ============================================================================

/// Network transport connecting physically separate cluster machines via TCP sockets.
pub struct TcpTransport {
    rank: usize,
    world_size: usize,
    connections: Mutex<HashMap<usize, TcpStream>>,
    _listener: Option<TcpListener>,
}

impl TcpTransport {
    pub fn new(
        rank: usize,
        world_size: usize,
        bind_addr: &str,
        bind_port: u16,
        peers: &HashMap<usize, (String, u16)>,
    ) -> Result<Self> {
        let listener = TcpListener::bind(format!("{}:{}", bind_addr, bind_port))?;
        listener.set_nonblocking(false)?;

        let mut connections = HashMap::new();

        // Connect to peers with rank < self.rank
        for (&peer_rank, (peer_ip, peer_port)) in peers {
            if peer_rank < rank {
                let mut stream = TcpStream::connect(format!("{}:{}", peer_ip, peer_port))?;
                stream.set_nodelay(true)?;
                // Send our connecting rank as a 4-byte little-endian handshake
                stream.write_all(&(rank as u32).to_le_bytes())?;
                stream.flush()?;
                connections.insert(peer_rank, stream);
            }
        }

        // Accept incoming connections from peers with rank > self.rank
        for _ in (rank + 1)..world_size {
            let (mut stream, _) = listener.accept()?;
            stream.set_nodelay(true)?;
            // Read peer rank from handshake header
            let mut rank_bytes = [0u8; 4];
            stream.read_exact(&mut rank_bytes)?;
            let peer_rank = u32::from_le_bytes(rank_bytes) as usize;
            connections.insert(peer_rank, stream);
        }

        Ok(Self {
            rank,
            world_size,
            connections: Mutex::new(connections),
            _listener: Some(listener),
        })
    }
}

impl Transport for TcpTransport {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send_to(&self, dest_rank: usize, data: &[f32]) -> Result<()> {
        let mut conns = self.connections.lock().unwrap();
        let stream = conns
            .get_mut(&dest_rank)
            .ok_or_else(|| anyhow::anyhow!("No TCP connection to rank {}", dest_rank))?;

        // Send byte length followed by raw byte payload
        let byte_len = (data.len() * std::mem::size_of::<f32>()) as u32;
        stream.write_all(&byte_len.to_le_bytes())?;

        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f32>())
        };
        stream.write_all(byte_slice)?;
        stream.flush()?;
        Ok(())
    }

    fn recv_from(&self, src_rank: usize, expected_len: usize) -> Result<Vec<f32>> {
        let mut conns = self.connections.lock().unwrap();
        let stream = conns
            .get_mut(&src_rank)
            .ok_or_else(|| anyhow::anyhow!("No TCP connection to rank {}", src_rank))?;

        let mut len_bytes = [0u8; 4];
        stream.read_exact(&mut len_bytes)?;
        let byte_len = u32::from_le_bytes(len_bytes) as usize;

        let expected_bytes = expected_len * std::mem::size_of::<f32>();
        if byte_len != expected_bytes {
            // Drain socket buffer to prevent persistent framing corruption
            let mut discard = vec![0u8; byte_len.min(65536)];
            let mut remaining = byte_len;
            while remaining > 0 {
                let to_read = remaining.min(discard.len());
                let _ = stream.read_exact(&mut discard[..to_read]);
                remaining -= to_read;
            }
            bail!("TCP byte length mismatch: expected {} bytes, got {}", expected_bytes, byte_len);
        }

        let mut buffer = vec![0.0f32; expected_len];
        let byte_slice = unsafe {
            std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, expected_bytes)
        };
        stream.read_exact(byte_slice)?;
        Ok(buffer)
    }

    fn barrier(&self) -> Result<()> {
        if self.world_size <= 1 {
            return Ok(());
        }
        let mut conns = self.connections.lock().unwrap();
        if self.rank == 0 {
            // Master coordinates: collect arrival from all worker ranks
            for r in 1..self.world_size {
                if let Some(stream) = conns.get_mut(&r) {
                    let mut token = [0u8; 1];
                    stream.read_exact(&mut token)?;
                }
            }
            // Master releases all worker ranks
            for r in 1..self.world_size {
                if let Some(stream) = conns.get_mut(&r) {
                    stream.write_all(&[1u8])?;
                    stream.flush()?;
                }
            }
        } else {
            // Worker signals arrival to rank 0
            if let Some(stream) = conns.get_mut(&0) {
                stream.write_all(&[1u8])?;
                stream.flush()?;
                // Worker awaits release notification from rank 0
                let mut token = [0u8; 1];
                stream.read_exact(&mut token)?;
            }
        }
        Ok(())
    }
}

