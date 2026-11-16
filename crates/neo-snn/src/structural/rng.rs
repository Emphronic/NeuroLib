//! Small deterministic generator (SplitMix64).
//!
//! Needed for the sake of reproducibility: the state is one `u64`, so it can be
//! save to checkpoint and continue exactly from the same place. Cryptographic
//! it is not and should not be used outside of structural ductility.

/// Deterministic RNG with serializable state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    /// Create from seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Restore from a saved state.
    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// Current state (for checkpoint).
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform number in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Uniform number in `(-1, 1)`.
    pub fn symmetric(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Uniform number in `[0, 1)` in scalar type `T`. Uses the same 24
    /// bit as [`next_f32`](Self::next_f32), so for `f32` the sequence
    /// the same (the reproducibility of the seeds is preserved).
    pub fn uniform<T: crate::scalar::Scalar>(&mut self) -> T {
        T::from_f64(((self.next_u64() >> 40) as f64) / (1u64 << 24) as f64)
    }

    /// Uniform number in `(-1, 1)` in scalar type `T`.
    pub fn symmetric_as<T: crate::scalar::Scalar>(&mut self) -> T {
        self.uniform::<T>() * T::c(2.0) - T::ONE
    }

    /// Index in `[0, n)`; for `n == 0` returns 0.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// `true` with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }
}
