use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;

/// A Fiat-Shamir transcript backed by Blake3.
///
/// Single random oracle (no per-round `fork`); the paper's FS uses
/// independent oracles `ρ_i` per round, but with negligible soundness loss
/// for a single-protocol setting (see `docs/future_revisions.md`).
#[derive(Clone)]
pub struct Blake3Transcript {
    hasher: blake3::Hasher,
}

impl Blake3Transcript {
    fn from_key(key: &[u8; 32]) -> Self {
        Self { hasher: blake3::Hasher::new_keyed(key) }
    }

    /// Initialize a fresh transcript with a domain-separation label. Two
    /// transcripts with different labels produce uncorrelated challenges.
    pub fn new(label: &'static [u8]) -> Self {
        let derived = blake3::derive_key("rok_m::transcript v0", label);
        let mut key = [0u8; 32];
        key.copy_from_slice(&derived);
        Self::from_key(&key)
    }

    /// Absorb raw bytes with a domain-separation label.
    pub fn absorb_bytes(&mut self, label: &'static [u8], bytes: &[u8]) {
        self.hasher.update(label);
        self.hasher.update(&(label.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
        self.hasher.update(&(bytes.len() as u64).to_le_bytes());
    }

    /// Absorb anything that has a canonical (compressed) byte representation.
    pub fn absorb<S: CanonicalSerialize>(&mut self, label: &'static [u8], val: &S) {
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf)
            .expect("serialization is infallible for well-formed types");
        self.absorb_bytes(label, &buf);
    }

    /// Absorb a `usize` portably (encoded as little-endian `u64`).
    pub fn absorb_usize(&mut self, label: &'static [u8], val: usize) {
        self.absorb_bytes(label, &(val as u64).to_le_bytes());
    }

    /// Squeeze pseudorandom bytes and re-key for the next operation.
    fn squeeze(&mut self, label: &'static [u8], out: &mut [u8]) {
        self.hasher.update(label);
        self.hasher.update(&(label.len() as u64).to_le_bytes());
        let mut reader = self.hasher.finalize_xof();
        reader.fill(out);
        // Re-key from the XOF so the next absorb starts fresh.
        let mut new_key = [0u8; 32];
        reader.fill(&mut new_key);
        self.hasher = blake3::Hasher::new_keyed(&new_key);
    }

    /// Squeeze a uniformly random field element (~2× field bytes for
    /// negligible bias under reduction mod p).
    pub fn squeeze_field<F: PrimeField>(&mut self, label: &'static [u8]) -> F {
        let byte_len = ((F::MODULUS_BIT_SIZE as usize) / 8 + 1) * 2;
        let mut buf = vec![0u8; byte_len];
        self.squeeze(label, &mut buf);
        F::from_le_bytes_mod_order(&buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;

    #[test]
    fn different_labels_produce_different_state() {
        let mut t1 = Blake3Transcript::new(b"a");
        let mut t2 = Blake3Transcript::new(b"b");
        let f1: Fr = t1.squeeze_field(b"x");
        let f2: Fr = t2.squeeze_field(b"x");
        assert_ne!(f1, f2);
    }

    #[test]
    fn absorb_changes_squeeze() {
        let mut t1 = Blake3Transcript::new(b"test");
        let mut t2 = Blake3Transcript::new(b"test");
        t1.absorb_bytes(b"msg", b"hello");
        let f1: Fr = t1.squeeze_field(b"x");
        let f2: Fr = t2.squeeze_field(b"x");
        assert_ne!(f1, f2);
    }

    #[test]
    fn deterministic() {
        let mut t1 = Blake3Transcript::new(b"test");
        let mut t2 = Blake3Transcript::new(b"test");
        t1.absorb_bytes(b"msg", b"hello");
        t2.absorb_bytes(b"msg", b"hello");
        let f1: Fr = t1.squeeze_field(b"x");
        let f2: Fr = t2.squeeze_field(b"x");
        assert_eq!(f1, f2);
    }
}
