use ark_bls12_381::Fr;
use ark_ff::Zero;
use ark_poly::Polynomial;

use crate::merkle::{verify_membership_from_hash, MembershipProof, MerkleTree};
use crate::relations::pcc::{PccParams, PccStatement, PccWitness};
use crate::relations::pco::{PcoStatement, PcoWitness};
use crate::transcript::Blake3Transcript;

use super::rok_pcc::{absorb_pcc_statement, evaluate_constraint, pow_fr};

/// `FsMt(RokPcc, κ)` — the Merkle-tree / Fiat-Shamir transform applied to
/// the `R_PCC → R_PCO^n` reduction.
///
/// The verifier's first message in the inner protocol — the evaluation
/// point `x` — is derived from the Merkle root of the `K = 2^κ` leaf
/// statements. Because every leaf binds to the same root, **all `K` leaves
/// share the same `x`**. The downstream PCO statements (one per polynomial
/// per leaf, so `K · n` total) all evaluate at this shared point and can be
/// batched in a single `RokPco` step.
///
/// Per-leaf transcript discipline: the verifier of leaf `b` must arrive at
/// `verify` with the same transcript state the prover had at the start of
/// `reduce_amortized`.
pub struct FsMtPcc;

/// Per-leaf proof: the prover's evaluations `y_i = p_i(x)`, the Merkle
/// membership proof for the leaf, and the tree root (so the verifier can
/// independently verify membership without reconstructing the tree).
pub struct FsMtPccProof {
    pub values: Vec<Fr>,
    pub membership: MembershipProof,
    pub root: [u8; 32],
}

impl FsMtPcc {
    /// Amortized prover. Returns one tuple per leaf.
    pub fn reduce_amortized(
        params: &PccParams,
        stmts: &[PccStatement],
        wits: &[PccWitness],
        transcript: &mut Blake3Transcript,
    ) -> Vec<(FsMtPccProof, Vec<PcoStatement>, Vec<PcoWitness>)> {
        assert_eq!(stmts.len(), wits.len(), "stmts and wits must agree in length");
        let k = stmts.len();
        assert!(
            k.is_power_of_two() && k >= 2,
            "K must be a power of two ≥ 2"
        );

        // 1. Hash each statement, binding to the current transcript state.
        let leaf_hashes: Vec<[u8; 32]> = stmts
            .iter()
            .enumerate()
            .map(|(i, s)| hash_leaf(transcript, i, s))
            .collect();

        // 2. Build tree, absorb root, derive shared x.
        let tree = MerkleTree::from_leaf_hashes(&leaf_hashes);
        let root = tree.root();
        transcript.absorb_bytes(b"fsmt_pcc::root", &root);
        let x: Fr = transcript.squeeze_field(b"fsmt_pcc::x");

        // 3. Per-leaf outputs.
        let big_d = params.srs.powers_g1.len();
        (0..k)
            .map(|i| {
                let stmt = &stmts[i];
                let wit = &wits[i];

                let values: Vec<Fr> =
                    wit.polynomials.iter().map(|p| p.evaluate(&x)).collect();

                let pco_stmts: Vec<PcoStatement> = stmt
                    .commitments
                    .iter()
                    .zip(&stmt.degrees)
                    .zip(&values)
                    .map(|((c, &d_i), &y)| {
                        let s = pow_fr(x, big_d - d_i);
                        PcoStatement {
                            commitment: *c * s,
                            point: x,
                            value: y * s,
                        }
                    })
                    .collect();
                let pco_wits: Vec<PcoWitness> = wit
                    .polynomials
                    .iter()
                    .zip(&stmt.degrees)
                    .map(|(p, &d_i)| {
                        let s = pow_fr(x, big_d - d_i);
                        PcoWitness { polynomial: p * s }
                    })
                    .collect();

                let proof = FsMtPccProof {
                    values,
                    membership: tree.membership_proof(i),
                    root,
                };
                (proof, pco_stmts, pco_wits)
            })
            .collect()
    }

    /// Per-leaf verifier. Returns `None` on either Merkle-membership failure
    /// or constraint-check failure (`Q_j(x, values) ≠ 0`).
    pub fn verify(
        params: &PccParams,
        index: usize,
        stmt: &PccStatement,
        proof: &FsMtPccProof,
        transcript: &mut Blake3Transcript,
    ) -> Option<Vec<PcoStatement>> {
        // 1. Recompute leaf hash and check Merkle membership.
        let leaf_hash = hash_leaf(transcript, index, stmt);
        let kappa = proof.membership.sibling_hashes.len();
        if !verify_membership_from_hash(
            kappa,
            index,
            &leaf_hash,
            &proof.root,
            &proof.membership,
        ) {
            return None;
        }

        // 2. Absorb root, derive shared x (same as the prover).
        transcript.absorb_bytes(b"fsmt_pcc::root", &proof.root);
        let x: Fr = transcript.squeeze_field(b"fsmt_pcc::x");

        // 3. Schwartz–Zippel constraint check.
        for q in &stmt.constraints {
            if !evaluate_constraint(q, x, &proof.values).is_zero() {
                return None;
            }
        }

        // 4. Build shifted PCO statements.
        let big_d = params.srs.powers_g1.len();
        Some(
            stmt.commitments
                .iter()
                .zip(&stmt.degrees)
                .zip(&proof.values)
                .map(|((c, &d_i), &y)| {
                    let s = pow_fr(x, big_d - d_i);
                    PcoStatement {
                        commitment: *c * s,
                        point: x,
                        value: y * s,
                    }
                })
                .collect(),
        )
    }
}

fn hash_leaf(parent: &Blake3Transcript, index: usize, s: &PccStatement) -> [u8; 32] {
    let mut t = parent.fork(b"fsmt_pcc::leaf");
    t.absorb_usize(b"index", index);
    absorb_pcc_statement(&mut t, s);
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
    use crate::relations::pcc::{Constraint, Monomial};
    use crate::relations::pco::{PcoParams, PcoRelation};
    use crate::reductions::rok_pco::RokPco;
    use ark_poly::univariate::SparsePolynomial;
    use ark_std::test_rng;

    /// Build one PCC instance with constraint `Y_0 − Y_1 = 0` over constant
    /// polynomials `p_0 = a, p_1 = a` (so the constraint holds polynomially).
    fn make_leaf(srs: &Srs, a: u64) -> (PccStatement, PccWitness) {
        let p0 = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(a))]);
        let p1 = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(a))]);
        let c0 = pc::commit(srs, &Poly::Sparse(p0.clone()));
        let c1 = pc::commit(srs, &Poly::Sparse(p1.clone()));
        let stmt = PccStatement {
            commitments: vec![c0, c1],
            degrees: vec![1, 1], // strict: deg < 1 → constants
            constraints: vec![Constraint {
                monomials: vec![
                    Monomial {
                        coeff: Fr::from(1u64),
                        x_deg: 0,
                        y_terms: vec![(0, 1)],
                    },
                    Monomial {
                        coeff: -Fr::from(1u64),
                        x_deg: 0,
                        y_terms: vec![(1, 1)],
                    },
                ],
            }],
        };
        let wit = PccWitness { polynomials: vec![p0, p1] };
        (stmt, wit)
    }

    /// Make K leaves and the params.
    fn k_leaves(k: usize) -> (PccParams, Vec<PccStatement>, Vec<PccWitness>) {
        let rng = &mut test_rng();
        let srs = pc::setup(20, rng);
        let mut stmts = Vec::with_capacity(k);
        let mut wits = Vec::with_capacity(k);
        for i in 0..k {
            let (s, w) = make_leaf(&srs, (i + 1) as u64);
            stmts.push(s);
            wits.push(w);
        }
        (PccParams { srs }, stmts, wits)
    }

    /// All `K` per-leaf outputs must verify under `PcoRelation`.
    #[test]
    fn amortized_roundtrip() {
        let (params, stmts, wits) = k_leaves(4);
        let mut t = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t);

        let pco_params = PcoParams { srs: params.srs.clone() };
        for (_, pco_stmts, pco_wits) in &outputs {
            for (s, w) in pco_stmts.iter().zip(pco_wits) {
                assert!(PcoRelation::is_satisfied(&pco_params, s, w));
            }
        }
    }

    /// Every PCO statement across every leaf shares the same evaluation point.
    /// This is the amortization invariant.
    #[test]
    fn all_leaves_share_x() {
        let (params, stmts, wits) = k_leaves(4);
        let mut t = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t);
        let x = outputs[0].1[0].point;
        for (_, pco_stmts, _) in &outputs {
            for s in pco_stmts {
                assert_eq!(s.point, x);
            }
        }
    }

    /// Per-leaf verify must reconstruct the same `Vec<PcoStatement>` the
    /// prover produced for that leaf.
    #[test]
    fn prover_verifier_agree() {
        let (params, stmts, wits) = k_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);

        for (i, (proof, prover_stmts, _)) in outputs.iter().enumerate() {
            let mut t_v = Blake3Transcript::new(b"test");
            let verifier_stmts = FsMtPcc::verify(&params, i, &stmts[i], proof, &mut t_v)
                .expect("honest verify must succeed");
            assert_eq!(*prover_stmts, verifier_stmts);
        }
    }

    /// Verifying leaf 0 against the proof for leaf 1 must fail (Merkle path
    /// hashes will not chain to the same root).
    #[test]
    fn wrong_leaf_index_rejected() {
        let (params, stmts, wits) = k_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);

        let proof_for_1 = &outputs[1].0;
        let mut t_v = Blake3Transcript::new(b"test");
        // Try to verify stmts[0] using the proof intended for index 1.
        assert!(FsMtPcc::verify(&params, 0, &stmts[0], proof_for_1, &mut t_v).is_none());
    }

    /// Tampering with the proof's claimed root must fail Merkle membership.
    #[test]
    fn tampered_root_rejected() {
        let (params, stmts, wits) = k_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);
        outputs[0].0.root[0] ^= 0xFF;

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(
            FsMtPcc::verify(&params, 0, &stmts[0], &outputs[0].0, &mut t_v).is_none()
        );
    }

    /// Tampering a `y` value flips the constraint check (with overwhelming
    /// probability) so verify returns `None`.
    #[test]
    fn tampered_y_rejected() {
        let (params, stmts, wits) = k_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);
        outputs[0].0.values[0] += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(
            FsMtPcc::verify(&params, 0, &stmts[0], &outputs[0].0, &mut t_v).is_none()
        );
    }

    /// The K · n PCO outputs across all leaves share the same evaluation
    /// point and can be batched into a single PCO instance via `RokPco`.
    /// This is the end-to-end amortization payoff.
    #[test]
    fn outputs_batchable_by_rok_pco() {
        let (params, stmts, wits) = k_leaves(4);
        let mut t = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t);

        let mut all_stmts = Vec::new();
        let mut all_wits = Vec::new();
        for (_, ss, ws) in outputs {
            all_stmts.extend(ss);
            all_wits.extend(ws);
        }

        let mut t_pco = Blake3Transcript::new(b"test_pco");
        let (batched_stmt, batched_wit) =
            RokPco::reduce(&all_stmts, &all_wits, &mut t_pco);

        let pco_params = PcoParams { srs: params.srs.clone() };
        assert!(PcoRelation::is_satisfied(
            &pco_params,
            &batched_stmt,
            &batched_wit
        ));
    }
}
