use ark_serialize::CanonicalSerialize;

use crate::transcript::Blake3Transcript;

/// Complete binary Merkle tree of depth `κ` over `2^κ` serializable leaves.
///
/// Leaves are hashed via a fork of the supplied parent transcript (so the
/// tree is bound to whatever protocol context built it); internal nodes use
/// a fixed-key Blake3 derivation (no transcript state).
///
/// Internally the tree uses a 1-indexed level-order array of size `2K`
/// where `K = 2^κ`: `nodes[1]` is the root, leaves live in `nodes[K..2K]`.
/// Index `0` is unused.
pub struct MerkleTree {
    nodes: Vec<[u8; 32]>,
    pub kappa: usize,
}

/// A membership proof: the κ sibling hashes along the path from a leaf to
/// the root, ordered leaf → root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MembershipProof {
    pub sibling_hashes: Vec<[u8; 32]>,
}

impl MerkleTree {
    /// Build a tree from `2^κ` leaves (κ ≥ 1).
    pub fn build<L: CanonicalSerialize>(leaves: &[L], transcript: &Blake3Transcript) -> Self {
        let k = leaves.len();
        assert!(
            k.is_power_of_two() && k >= 2,
            "leaves must be a power-of-two ≥ 2"
        );
        let kappa = k.trailing_zeros() as usize;
        let mut nodes = vec![[0u8; 32]; 2 * k];

        // Leaf level: nodes[k..2k]
        for (i, leaf) in leaves.iter().enumerate() {
            nodes[k + i] = hash_leaf(leaf, transcript, i);
        }
        // Internal nodes, bottom-up.
        for i in (1..k).rev() {
            nodes[i] = hash_pair(&nodes[2 * i], &nodes[2 * i + 1]);
        }
        MerkleTree { nodes, kappa }
    }

    /// Build a tree directly from precomputed leaf hashes. Use this when the
    /// leaf type doesn't implement `CanonicalSerialize` and the caller hashes
    /// each leaf externally (binding the hash to the relevant context).
    pub fn from_leaf_hashes(leaf_hashes: &[[u8; 32]]) -> Self {
        let k = leaf_hashes.len();
        assert!(
            k.is_power_of_two() && k >= 2,
            "leaves must be a power-of-two ≥ 2"
        );
        let kappa = k.trailing_zeros() as usize;
        let mut nodes = vec![[0u8; 32]; 2 * k];
        for (i, h) in leaf_hashes.iter().enumerate() {
            nodes[k + i] = *h;
        }
        for i in (1..k).rev() {
            nodes[i] = hash_pair(&nodes[2 * i], &nodes[2 * i + 1]);
        }
        MerkleTree { nodes, kappa }
    }

    pub fn root(&self) -> [u8; 32] {
        self.nodes[1]
    }

    /// Membership proof for the leaf at `index` (0-indexed).
    pub fn membership_proof(&self, index: usize) -> MembershipProof {
        let k = 1 << self.kappa;
        assert!(index < k, "index out of range");
        let mut node = k + index;
        let mut sibs = Vec::with_capacity(self.kappa);
        while node > 1 {
            let sib = if node % 2 == 0 { node + 1 } else { node - 1 };
            sibs.push(self.nodes[sib]);
            node /= 2;
        }
        MembershipProof { sibling_hashes: sibs }
    }

    /// All `2^κ` membership proofs in index order.
    pub fn all_membership_proofs(&self) -> Vec<MembershipProof> {
        (0..(1 << self.kappa))
            .map(|i| self.membership_proof(i))
            .collect()
    }
}

/// Verify that `leaf` at `index` belongs to a tree of depth `kappa` rooted
/// at `root`. The `transcript` must be the same parent state that was used
/// to build the tree (its leaf-hash fork is deterministic from it).
pub fn verify_membership<L: CanonicalSerialize>(
    kappa: usize,
    index: usize,
    leaf: &L,
    root: &[u8; 32],
    proof: &MembershipProof,
    transcript: &Blake3Transcript,
) -> bool {
    if proof.sibling_hashes.len() != kappa {
        return false;
    }
    if index >= (1 << kappa) {
        return false;
    }
    let mut current = hash_leaf(leaf, transcript, index);
    let mut node = (1 << kappa) + index;
    for sib in &proof.sibling_hashes {
        let (l, r) = if node % 2 == 0 {
            (&current, sib)
        } else {
            (sib, &current)
        };
        current = hash_pair(l, r);
        node /= 2;
    }
    &current == root
}

/// Verify membership when the leaf has been hashed externally (paired with
/// `MerkleTree::from_leaf_hashes`).
pub fn verify_membership_from_hash(
    kappa: usize,
    index: usize,
    leaf_hash: &[u8; 32],
    root: &[u8; 32],
    proof: &MembershipProof,
) -> bool {
    if proof.sibling_hashes.len() != kappa {
        return false;
    }
    if index >= (1 << kappa) {
        return false;
    }
    let mut current = *leaf_hash;
    let mut node = (1 << kappa) + index;
    for sib in &proof.sibling_hashes {
        let (l, r) = if node % 2 == 0 {
            (&current, sib)
        } else {
            (sib, &current)
        };
        current = hash_pair(l, r);
        node /= 2;
    }
    &current == root
}

fn hash_leaf<L: CanonicalSerialize>(
    leaf: &L,
    transcript: &Blake3Transcript,
    index: usize,
) -> [u8; 32] {
    let mut t = transcript.fork(b"mt_leaf");
    t.absorb_usize(b"index", index);
    t.absorb(b"leaf", leaf);
    let mut out = [0u8; 32];
    t.squeeze_bytes(b"hash", &mut out);
    out
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key("ap-merkle internal v0");
    h.update(left);
    h.update(right);
    *h.finalize().as_bytes()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All `2^κ` membership proofs verify against the root.
    #[test]
    fn build_and_verify() {
        let t = Blake3Transcript::new(b"test-tree");
        let leaves: Vec<u64> = (0..8).collect();
        let tree = MerkleTree::build(&leaves, &t);
        let root = tree.root();

        for i in 0..8 {
            let proof = tree.membership_proof(i);
            assert!(
                verify_membership(3, i, &leaves[i], &root, &proof, &t),
                "proof for leaf {i} rejected"
            );
        }
    }

    /// `all_membership_proofs` agrees with index-by-index calls.
    #[test]
    fn all_proofs_match_individual() {
        let t = Blake3Transcript::new(b"test");
        let leaves: Vec<u64> = (0..4).collect();
        let tree = MerkleTree::build(&leaves, &t);
        let all = tree.all_membership_proofs();
        for (i, p) in all.iter().enumerate() {
            assert_eq!(*p, tree.membership_proof(i));
        }
    }

    /// Changing the leaf invalidates its proof.
    #[test]
    fn tampered_leaf_fails() {
        let t = Blake3Transcript::new(b"test");
        let leaves: Vec<u64> = (0..4).collect();
        let tree = MerkleTree::build(&leaves, &t);
        let proof = tree.membership_proof(0);
        let tampered: u64 = 99;
        assert!(!verify_membership(2, 0, &tampered, &tree.root(), &proof, &t));
    }

    /// Modifying any sibling in the proof breaks verification.
    #[test]
    fn tampered_sibling_fails() {
        let t = Blake3Transcript::new(b"test");
        let leaves: Vec<u64> = (0..4).collect();
        let tree = MerkleTree::build(&leaves, &t);
        let mut proof = tree.membership_proof(2);
        proof.sibling_hashes[0][0] ^= 0xFF;
        assert!(!verify_membership(2, 2, &leaves[2], &tree.root(), &proof, &t));
    }

    /// A valid leaf with the wrong claimed index fails (different leaf hash
    /// + path direction).
    #[test]
    fn wrong_index_fails() {
        let t = Blake3Transcript::new(b"test");
        let leaves: Vec<u64> = (0..4).collect();
        let tree = MerkleTree::build(&leaves, &t);
        let proof = tree.membership_proof(0);
        // Try to verify the leaf-0 proof as if it were leaf 1.
        assert!(!verify_membership(2, 1, &leaves[0], &tree.root(), &proof, &t));
    }

    /// Same leaves under a different parent transcript → different root.
    #[test]
    fn different_transcript_changes_root() {
        let leaves: Vec<u64> = (0..4).collect();
        let t1 = Blake3Transcript::new(b"label_a");
        let t2 = Blake3Transcript::new(b"label_b");
        let r1 = MerkleTree::build(&leaves, &t1).root();
        let r2 = MerkleTree::build(&leaves, &t2).root();
        assert_ne!(r1, r2);
    }
}
