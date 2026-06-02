use crate::merkle::{verify_membership_from_hash, MembershipProof, MerkleTree};
use crate::relations::pco::{PcoStatement, PcoWitness};
use crate::transcript::Blake3Transcript;

use super::rok_pco::RokPco;

/// `LfscPcc(RokPcoFold, κ)` — the plain (no-promise) K-leaf local folding
/// scheme for `R_PCO^x`. Given `K = 2^κ` PCO statements all at the same
/// evaluation point `x`, produces one root PCO instance plus per-leaf
/// path proofs that each verifier independently checks.
///
/// **Transcript discipline.** A Merkle root over the K input PCO statements
/// is absorbed into the (shared) transcript before any fold challenge is
/// squeezed, so the K leaves are bound to the prover's claim. Each fold
/// node then forks the transcript with `(level, pair)`, lets `RokPco`'s
/// internal absorb-and-squeeze derive `r`, and produces the folded PCO.
pub struct PcoFold;

/// One step of a leaf's fold path: just the sibling PCO statement
/// (`RokPco` produces no on-the-wire proof message).
pub struct PcoFoldPathStep {
    pub sibling: PcoStatement,
}

pub struct PcoFoldPathProof {
    /// Merkle root over the K input PCO statements.
    pub root: [u8; 32],
    /// Membership proof for this leaf in the Merkle tree above.
    pub membership: MembershipProof,
    /// κ entries, leaf → root order.
    pub steps: Vec<PcoFoldPathStep>,
}

impl PcoFold {
    /// Prover side. All input statements must share the same evaluation
    /// point `x` (precondition); panics otherwise.
    pub fn fold_amortized(
        leaves: Vec<(PcoStatement, PcoWitness)>,
        transcript: &mut Blake3Transcript,
    ) -> ((PcoStatement, PcoWitness), Vec<PcoFoldPathProof>) {
        let k = leaves.len();
        assert!(
            k.is_power_of_two() && k >= 2,
            "K must be a power of two ≥ 2"
        );
        let kappa = k.trailing_zeros() as usize;
        assert_shared_point(&leaves);

        // 1. Hash each input PCO statement, build the Merkle tree, absorb root.
        let leaf_hashes: Vec<[u8; 32]> = leaves
            .iter()
            .enumerate()
            .map(|(i, (s, _))| hash_leaf(transcript, i, s))
            .collect();
        let tree = MerkleTree::from_leaf_hashes(&leaf_hashes);
        let root = tree.root();
        transcript.absorb_bytes(b"pco_fold::root", &root);

        // 2. Build the fold tree, recording per-leaf siblings.
        let mut node_left_stmts: Vec<Vec<PcoStatement>> = Vec::with_capacity(kappa);
        let mut node_right_stmts: Vec<Vec<PcoStatement>> = Vec::with_capacity(kappa);
        let mut current: Vec<(PcoStatement, PcoWitness)> = leaves;

        for level in 0..kappa {
            let pairs = current.len() / 2;
            let mut next: Vec<(PcoStatement, PcoWitness)> = Vec::with_capacity(pairs);
            let mut lefts: Vec<PcoStatement> = Vec::with_capacity(pairs);
            let mut rights: Vec<PcoStatement> = Vec::with_capacity(pairs);

            let mut iter = current.into_iter();
            for pair_idx in 0..pairs {
                let (left_stmt, left_wit) = iter.next().unwrap();
                let (right_stmt, right_wit) = iter.next().unwrap();

                let mut t_node = transcript.fork(b"pco_fold");
                t_node.absorb_usize(b"level", level);
                t_node.absorb_usize(b"pair", pair_idx);

                // Binary RokPco fold over the pair.
                let stmts_pair = vec![left_stmt.clone(), right_stmt.clone()];
                let wits_pair = vec![left_wit, right_wit];
                let (folded_stmt, folded_wit) =
                    RokPco::reduce(&stmts_pair, &wits_pair, &mut t_node);

                lefts.push(left_stmt);
                rights.push(right_stmt);
                next.push((folded_stmt, folded_wit));
            }
            node_left_stmts.push(lefts);
            node_right_stmts.push(rights);
            current = next;
        }
        assert_eq!(current.len(), 1);
        let root_pair = current.into_iter().next().unwrap();

        // 3. Build per-leaf paths.
        let paths: Vec<PcoFoldPathProof> = (0..k)
            .map(|leaf_idx| {
                let mut steps: Vec<PcoFoldPathStep> = Vec::with_capacity(kappa);
                let mut cur = leaf_idx;
                for level in 0..kappa {
                    let pair_idx = cur / 2;
                    let position = cur & 1;
                    let sibling = if position == 0 {
                        node_right_stmts[level][pair_idx].clone()
                    } else {
                        node_left_stmts[level][pair_idx].clone()
                    };
                    steps.push(PcoFoldPathStep { sibling });
                    cur = pair_idx;
                }
                PcoFoldPathProof {
                    root,
                    membership: tree.membership_proof(leaf_idx),
                    steps,
                }
            })
            .collect();

        (root_pair, paths)
    }

    /// Per-leaf verifier. Returns `true` iff:
    /// 1. `leaf`'s hash is a Merkle leaf at index `index` under `proof.root`, and
    /// 2. the κ-step fold chain — using `RokPco::verify` at each node with the
    ///    same fork pattern as the prover — reaches `claimed_root`.
    pub fn verify(
        index: usize,
        leaf: &PcoStatement,
        proof: &PcoFoldPathProof,
        claimed_root: &PcoStatement,
        transcript: &mut Blake3Transcript,
    ) -> bool {
        let kappa = proof.steps.len();
        if proof.membership.sibling_hashes.len() != kappa {
            return false;
        }

        // 1. Recompute leaf hash and check Merkle membership.
        let leaf_hash = hash_leaf(transcript, index, leaf);
        if !verify_membership_from_hash(
            kappa,
            index,
            &leaf_hash,
            &proof.root,
            &proof.membership,
        ) {
            return false;
        }

        // 2. Absorb root, walk the fold tree.
        transcript.absorb_bytes(b"pco_fold::root", &proof.root);

        let mut current = leaf.clone();
        let mut cur_idx = index;
        for (level, step) in proof.steps.iter().enumerate() {
            let pair_idx = cur_idx / 2;
            let position = cur_idx & 1;
            let (left, right) = if position == 0 {
                (current.clone(), step.sibling.clone())
            } else {
                (step.sibling.clone(), current.clone())
            };

            let mut t_node = transcript.fork(b"pco_fold");
            t_node.absorb_usize(b"level", level);
            t_node.absorb_usize(b"pair", pair_idx);

            let stmts_pair = vec![left, right];
            current = RokPco::verify(&stmts_pair, &mut t_node);
            cur_idx = pair_idx;
        }

        &current == claimed_root
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_shared_point(leaves: &[(PcoStatement, PcoWitness)]) {
    let x = leaves[0].0.point;
    for (s, _) in &leaves[1..] {
        assert_eq!(
            s.point, x,
            "all inputs of PcoFold must share the same evaluation point"
        );
    }
}

fn hash_leaf(parent: &Blake3Transcript, index: usize, s: &PcoStatement) -> [u8; 32] {
    let mut t = parent.fork(b"pco_fold::leaf");
    t.absorb_usize(b"index", index);
    t.absorb(b"commitment", &s.commitment);
    t.absorb(b"point", &s.point);
    t.absorb(b"value", &s.value);
    let mut out = [0u8; 32];
    t.squeeze_bytes(b"hash", &mut out);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Relation;
    use crate::pc::{self, Poly, Srs};
    use crate::relations::pco::{PcoParams, PcoRelation};
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_poly::{univariate::SparsePolynomial, Polynomial};
    use ark_std::test_rng;

    /// Build K PCO leaves all at the same shared point `x`.
    fn k_pco_leaves(k: usize, x: Fr) -> (Srs, Vec<(PcoStatement, PcoWitness)>) {
        let rng = &mut test_rng();
        let srs = pc::setup(20, rng);
        let leaves: Vec<(PcoStatement, PcoWitness)> = (0..k)
            .map(|i| {
                // p_i(X) = (i+1) + (i+2)·X — distinct per leaf.
                let p = SparsePolynomial::from_coefficients_vec(vec![
                    (0, Fr::from((i + 1) as u64)),
                    (1, Fr::from((i + 2) as u64)),
                ]);
                let value = p.evaluate(&x);
                let commitment = pc::commit(&srs, &Poly::Sparse(p.clone()));
                let stmt = PcoStatement { commitment, point: x, value };
                let wit = PcoWitness { polynomial: p };
                (stmt, wit)
            })
            .collect();
        (srs, leaves)
    }

    /// Round-trip K=4: root satisfies `R_PCO` and every per-leaf path verifies.
    #[test]
    fn fold_roundtrip_k4() {
        let x = Fr::rand(&mut test_rng());
        let (srs, leaves) = k_pco_leaves(4, x);
        let leaf_stmts: Vec<PcoStatement> =
            leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let ((root_stmt, root_wit), paths) = PcoFold::fold_amortized(leaves, &mut t_p);

        let pco_params = PcoParams { srs };
        assert!(PcoRelation::is_satisfied(&pco_params, &root_stmt, &root_wit));

        for (i, path) in paths.iter().enumerate() {
            let mut t_v = Blake3Transcript::new(b"test");
            assert!(PcoFold::verify(i, &leaf_stmts[i], path, &root_stmt, &mut t_v));
        }
    }

    /// Tampered claimed root → reject.
    #[test]
    fn wrong_root_rejected() {
        let x = Fr::rand(&mut test_rng());
        let (_srs, leaves) = k_pco_leaves(4, x);
        let leaf_stmts: Vec<PcoStatement> =
            leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let ((mut root_stmt, _), paths) = PcoFold::fold_amortized(leaves, &mut t_p);
        root_stmt.value += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(!PcoFold::verify(0, &leaf_stmts[0], &paths[0], &root_stmt, &mut t_v));
    }

    /// Tampered sibling in path → fold chain misses the root → reject.
    #[test]
    fn tampered_sibling_rejected() {
        let x = Fr::rand(&mut test_rng());
        let (_srs, leaves) = k_pco_leaves(4, x);
        let leaf_stmts: Vec<PcoStatement> =
            leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let ((root_stmt, _), mut paths) = PcoFold::fold_amortized(leaves, &mut t_p);
        paths[0].steps[0].sibling.value += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(!PcoFold::verify(0, &leaf_stmts[0], &paths[0], &root_stmt, &mut t_v));
    }

    /// Tampered Merkle root → membership fails.
    #[test]
    fn tampered_merkle_root_rejected() {
        let x = Fr::rand(&mut test_rng());
        let (_srs, leaves) = k_pco_leaves(4, x);
        let leaf_stmts: Vec<PcoStatement> =
            leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let ((root_stmt, _), mut paths) = PcoFold::fold_amortized(leaves, &mut t_p);
        paths[0].root[0] ^= 0xFF;

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(!PcoFold::verify(0, &leaf_stmts[0], &paths[0], &root_stmt, &mut t_v));
    }

    /// Verify leaf 0 with the proof intended for leaf 1 → Merkle membership fails.
    #[test]
    fn wrong_index_rejected() {
        let x = Fr::rand(&mut test_rng());
        let (_srs, leaves) = k_pco_leaves(4, x);
        let leaf_stmts: Vec<PcoStatement> =
            leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let ((root_stmt, _), paths) = PcoFold::fold_amortized(leaves, &mut t_p);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(!PcoFold::verify(0, &leaf_stmts[0], &paths[1], &root_stmt, &mut t_v));
    }

    /// Inputs at different evaluation points violate the precondition.
    #[test]
    #[should_panic(expected = "all inputs of PcoFold must share the same evaluation point")]
    fn mismatched_points_panics() {
        let rng = &mut test_rng();
        let srs = pc::setup(10, rng);
        let p = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(1u64))]);
        let c = pc::commit(&srs, &Poly::Sparse(p.clone()));
        let leaves = vec![
            (
                PcoStatement { commitment: c, point: Fr::from(1u64), value: Fr::from(1u64) },
                PcoWitness { polynomial: p.clone() },
            ),
            (
                PcoStatement { commitment: c, point: Fr::from(2u64), value: Fr::from(1u64) },
                PcoWitness { polynomial: p },
            ),
        ];
        let mut t = Blake3Transcript::new(b"test");
        let _ = PcoFold::fold_amortized(leaves, &mut t);
    }
}
