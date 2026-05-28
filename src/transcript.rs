use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;

/// A Fiat-Shamir transcript backed by Blake3.
///
/// Provides a `fork(domain)` operation that derives an *independent* child
/// transcript — used by `merkle.rs` to bind leaf hashes to the protocol
/// context, and by future composed protocols to give each sub-proof its own
/// challenge stream (the paper's `ρ_i` oracles).
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

    /// Squeeze raw pseudorandom bytes (used e.g. for Merkle leaf hashes).
    pub fn squeeze_bytes(&mut self, label: &'static [u8], out: &mut [u8]) {
        self.squeeze(label, out);
    }

    /// Derive an independent child transcript whose challenges are
    /// uncorrelated with the parent's. Models the paper's `ρ_i` (an
    /// independent random oracle).
    pub fn fork(&self, domain: &'static [u8]) -> Self {
        let current = self.hasher.finalize();
        let mut key_material = Vec::with_capacity(32 + domain.len());
        key_material.extend_from_slice(current.as_bytes());
        key_material.extend_from_slice(domain);
        let derived = blake3::derive_key("ap-transcript fork v0", &key_material);
        let mut key = [0u8; 32];
        key.copy_from_slice(&derived);
        Self::from_key(&key)
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

    /// Two forks with different domains must yield uncorrelated challenges.
    #[test]
    fn fork_produces_independent_sequences() {
        let t = Blake3Transcript::new(b"test");
        let mut a = t.fork(b"oracle_a");
        let mut b = t.fork(b"oracle_b");
        let fa: Fr = a.squeeze_field(b"x");
        let fb: Fr = b.squeeze_field(b"x");
        assert_ne!(fa, fb);
    }

    /// Forking the same parent state with the same domain is deterministic.
    #[test]
    fn fork_deterministic() {
        let t = Blake3Transcript::new(b"test");
        let mut a1 = t.fork(b"oracle");
        let mut a2 = t.fork(b"oracle");
        let f1: Fr = a1.squeeze_field(b"x");
        let f2: Fr = a2.squeeze_field(b"x");
        assert_eq!(f1, f2);
    }

    /// Forking does not advance the parent transcript.
    #[test]
    fn fork_does_not_mutate_parent() {
        let mut a = Blake3Transcript::new(b"test");
        let mut b = Blake3Transcript::new(b"test");
        let _ = a.fork(b"side");
        let fa: Fr = a.squeeze_field(b"x");
        let fb: Fr = b.squeeze_field(b"x");
        assert_eq!(fa, fb);
    }
}
