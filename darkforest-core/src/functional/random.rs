//! Explicit Functional Randomness (JAX jax.random equivalent)
//!
//! Features:
//! - Pure, stateless Pseudo-Random Number Generation (PRNG)
//! - Explicit `Key` tracking eliminating global RNG hidden state
//! - `key.split()` and `key.split_n(n)` for thread-safe, reproducible parallel forks
//! - `key.fold_in(data)` to mix in iteration, step, or worker metadata
//! - High-avalanche bit mixer (SplitMix64 + Murmur3 permutation)
//! - Pure distribution samplers: Uniform, Normal (Box-Muller), Bernoulli, RandInt, Rademacher

use super::array::Array;

/// A 128-bit functional pseudo-random key.
/// Unlike standard libraries that maintain a hidden mutable global PRNG state,
/// all random operations in Dark Forest take an explicit `Key` and return an immutable `Array`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    state: [u64; 2],
}

impl Key {
    /// Creates a new root PRNG Key from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let s0 = splitmix64(seed);
        let s1 = splitmix64(s0);
        Self { state: [s0, s1] }
    }

    /// Splits a PRNG key into two independent subkeys: `(key1, key2)`.
    /// The original key should be discarded or kept for a separate branch of execution.
    pub fn split(&self) -> (Key, Key) {
        let (k1, k2) = split_two(self.state[0], self.state[1]);
        (Key { state: k1 }, Key { state: k2 })
    }

    /// Splits a PRNG key into `n` mutually independent subkeys.
    /// Useful for mapping across batches or parallel worker threads.
    pub fn split_n(&self, n: usize) -> Vec<Key> {
        let mut keys = Vec::with_capacity(n);
        let mut curr = *self;
        for i in 0..n {
            let (k_branch, k_next) = curr.split();
            keys.push(k_branch.fold_in(i as u64));
            curr = k_next;
        }
        keys
    }

    /// Folds an arbitrary integer (e.g. step index, layer index, rank) into the key,
    /// deterministically producing a new independent key.
    pub fn fold_in(&self, data: u64) -> Key {
        let s0 = splitmix64(self.state[0] ^ data);
        let s1 = splitmix64(self.state[1] ^ splitmix64(data));
        Key { state: [s0, s1] }
    }

    /// Raw state access
    pub fn raw_state(&self) -> [u64; 2] {
        self.state
    }
}

// ----------------------------------------------------------------------------
// Pure Distribution Samplers (Functional)
// ----------------------------------------------------------------------------

/// Generates an Array of uniformly distributed floats in `[low, high)`.
pub fn uniform(key: &Key, shape: Vec<usize>, low: f32, high: f32) -> Array {
    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);

    let mut state = key.state;
    for _ in 0..numel {
        let (next_state, u) = next_u01(state);
        state = next_state;
        data.push(low + u * (high - low));
    }

    Array::from_vec(data, shape).expect("Valid uniform array")
}

/// Generates an Array of standard normally distributed floats via Box-Muller transform.
pub fn normal(key: &Key, shape: Vec<usize>, mean: f32, std: f32) -> Array {
    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);

    let mut state = key.state;
    let pairs = (numel + 1) / 2;

    for _ in 0..pairs {
        let (s1, u1) = next_u01(state);
        let (s2, u2) = next_u01(s1);
        state = s2;

        let u1_safe = (u1 as f64).max(1e-15);
        let r = (-2.0 * u1_safe.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * (u2 as f64);

        let z0 = (r * theta.cos()) as f32;
        let z1 = (r * theta.sin()) as f32;

        data.push(mean + z0 * std);
        if data.len() < numel {
            data.push(mean + z1 * std);
        }
    }

    Array::from_vec(data, shape).expect("Valid normal array")
}

/// Generates a boolean mask or Bernoulli array with success probability `p` (1.0 with prob p, 0.0 otherwise).
pub fn bernoulli(key: &Key, shape: Vec<usize>, p: f32) -> Array {
    let u = uniform(key, shape.clone(), 0.0, 1.0);
    let data = u.to_vec().into_iter().map(|x| if x < p { 1.0 } else { 0.0 }).collect();
    Array::from_vec(data, shape).expect("Valid bernoulli array")
}

/// Generates an Array of uniform random integers in `[low, high)`.
pub fn randint(key: &Key, shape: Vec<usize>, low: i64, high: i64) -> Array {
    assert!(high > low, "high must be strictly greater than low");
    let range = (high - low) as f64;
    let u = uniform(key, shape.clone(), 0.0, 1.0);
    let data = u
        .to_vec()
        .into_iter()
        .map(|x| (low as f64 + (x as f64) * range).floor() as f32)
        .collect();
    Array::from_vec(data, shape).expect("Valid randint array")
}

/// Generates Rademacher random variables (-1.0 or +1.0 with equal 50% probability).
pub fn rademacher(key: &Key, shape: Vec<usize>) -> Array {
    let b = bernoulli(key, shape.clone(), 0.5);
    let data = b.to_vec().into_iter().map(|x| if x > 0.5 { 1.0 } else { -1.0 }).collect();
    Array::from_vec(data, shape).expect("Valid rademacher array")
}

// ----------------------------------------------------------------------------
// Internal High-Avalanche Pseudo-Random Bit Generators
// ----------------------------------------------------------------------------

#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[inline(always)]
fn split_two(s0: u64, s1: u64) -> ([u64; 2], [u64; 2]) {
    let m0 = splitmix64(s0 ^ 0xA0761D6478BD642F);
    let m1 = splitmix64(s1 ^ 0xE7037ED1A0B428DB);
    let m2 = splitmix64(s0.wrapping_add(0x8EB7650F7D7423D1));
    let m3 = splitmix64(s1.wrapping_add(0x587C5A33913D95D2));
    ([m0, m1], [m2, m3])
}

#[inline(always)]
fn next_u01(state: [u64; 2]) -> ([u64; 2], f32) {
    let s0 = state[0];
    let mut s1 = state[1];
    let result = s0.wrapping_add(s1);

    s1 ^= s0;
    let new_s0 = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
    let new_s1 = s1.rotate_left(37);

    // Convert upper 24 bits to uniform float in [0, 1)
    let u = ((result >> 40) as f32) / (16777216.0f32);
    ([new_s0, new_s1], u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_split_reproducibility() {
        let k1 = Key::new(42);
        let k2 = Key::new(42);

        let (sub1_a, sub1_b) = k1.split();
        let (sub2_a, sub2_b) = k2.split();

        assert_eq!(sub1_a, sub2_a);
        assert_eq!(sub1_b, sub2_b);
        assert_ne!(sub1_a, sub1_b);
    }

    #[test]
    fn test_key_fold_in() {
        let k = Key::new(100);
        let k_step0 = k.fold_in(0);
        let k_step1 = k.fold_in(1);
        assert_ne!(k_step0, k_step1);
    }

    #[test]
    fn test_uniform_and_normal_ranges() {
        let k = Key::new(1337);
        let u = uniform(&k, vec![1000], -2.0, 5.0);
        for &val in &u.to_vec() {
            assert!(val >= -2.0 && val <= 5.0);
        }

        let n = normal(&k, vec![2000], 0.0, 1.0);
        let mean = n.mean().unwrap().item().unwrap();
        assert!(mean.abs() < 0.1, "Sample mean should be close to 0, got {}", mean);
    }
}

