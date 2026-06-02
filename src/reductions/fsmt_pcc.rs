use ark_bls12_381::Fr;
use ark_ff::Zero;
use ark_poly::{univariate::SparsePolynomial, Polynomial};

use crate::merkle::{verify_membership_from_hash, MembershipProof, MerkleTree};
use crate::pc::Comm;
use crate::relations::pcc::{PccParams, PccStatement, PccWitness};
use crate::relations::pcc_d::PccDStatement;
use crate::relations::pco::{PcoStatement, PcoWitness};
use crate::transcript::Blake3Transcript;

use super::rok_dt::{RokDt, RokDtProof};
use super::rok_pcc::{absorb_pcc_d_statement, evaluate_constraint};

/// `FsMt(Π_PC ∘ Π_DT, κ)` — the Merkle-tree / Fiat-Shamir transform applied
/// to the composed `R_PCC --Π_DT--> R_PCC-D --Π_PC--> R_PCO` chain.
///
/// **Input shape.** Per leaf, a *bundle* of inner `R_PCC` statements
/// (`Vec<PccStatement>`): when `MFold` feeds this protocol, each leaf
/// carries the `κ` `R_PCC` promises gathered along its fold path. The
/// protocol treats the bundle as `R_PCC^κ_inner` — semantically the
/// Cartesian product — without ever concatenating into a single `R_PCC`.
///
/// **Output.** One `PcoStatement` per leaf, all sharing the same `x`
/// (derived from the Merkle root over the K per-leaf bundles).
pub struct FsMtPcc;

/// Per-leaf proof. All vecs are *outer-indexed by the inner statement*
/// in the leaf's bundle.
/// - `shifted_commitments[j]`: `Π_DT` shifted commitments for the j-th inner stmt.
/// - `values[j]`: evaluations at `x` of all 2·n_j polynomials of the j-th inner stmt.
pub struct FsMtPccProof {
    pub shifted_commitments: Vec<Vec<Comm>>,
    pub values: Vec<Vec<Fr>>,
    pub membership: MembershipProof,
    pub root: [u8; 32],
}

impl FsMtPcc {
    /// Amortized prover. Returns one tuple per leaf.
    pub fn reduce_amortized(
        params: &PccParams,
        stmts: &[Vec<PccStatement>],
        wits: &[Vec<PccWitness>],
        transcript: &mut Blake3Transcript,
    ) -> Vec<(FsMtPccProof, PcoStatement, PcoWitness)> {
        assert_eq!(stmts.len(), wits.len(), "K leaves on both sides");
        let k = stmts.len();
        assert!(
            k.is_power_of_two() && k >= 2,
            "K must be a power of two ≥ 2"
        );

        // 1. Per leaf, per inner stmt: run Π_DT.
        let mut dt_proofs: Vec<Vec<Vec<Comm>>> = Vec::with_capacity(k);
        let mut d_stmts: Vec<Vec<PccDStatement>> = Vec::with_capacity(k);
        let mut d_wits: Vec<Vec<crate::relations::pcc_d::PccDWitness>> = Vec::with_capacity(k);

        for (leaf_idx, (leaf_stmts, leaf_wits)) in stmts.iter().zip(wits).enumerate() {
            assert_eq!(
                leaf_stmts.len(),
                leaf_wits.len(),
                "inner bundle sizes must match"
            );
            let mut dt_for_leaf: Vec<Vec<Comm>> = Vec::with_capacity(leaf_stmts.len());
            let mut d_stmts_for_leaf: Vec<PccDStatement> = Vec::with_capacity(leaf_stmts.len());
            let mut d_wits_for_leaf: Vec<crate::relations::pcc_d::PccDWitness> =
                Vec::with_capacity(leaf_stmts.len());
            for (inner_idx, (s, w)) in leaf_stmts.iter().zip(leaf_wits).enumerate() {
                let mut t_dt = transcript.fork(b"fsmt_pcc::dt");
                t_dt.absorb_usize(b"leaf_idx", leaf_idx);
                t_dt.absorb_usize(b"inner_idx", inner_idx);
                let (dt_proof, d_stmt, d_wit) = RokDt::reduce(params, s, w, &mut t_dt);
                dt_for_leaf.push(dt_proof.shifted_commitments);
                d_stmts_for_leaf.push(d_stmt);
                d_wits_for_leaf.push(d_wit);
            }
            dt_proofs.push(dt_for_leaf);
            d_stmts.push(d_stmts_for_leaf);
            d_wits.push(d_wits_for_leaf);
        }

        // 2. Hash each leaf's bundle of PCC-D statements; build Merkle tree.
        let leaf_hashes: Vec<[u8; 32]> = d_stmts
            .iter()
            .enumerate()
            .map(|(i, bundle)| hash_leaf(transcript, i, bundle))
            .collect();
        let tree = MerkleTree::from_leaf_hashes(&leaf_hashes);
        let root = tree.root();
        transcript.absorb_bytes(b"fsmt_pcc::root", &root);

        // 3. Shared evaluation point.
        let x: Fr = transcript.squeeze_field(b"fsmt_pcc::x");

        // 4. Per leaf: evaluate every polynomial at `x`, squeeze leaf-local `r`,
        //    then RLC across the entire bundle (flat over inner stmts).
        (0..k)
            .map(|i| {
                let d_bundle_stmts = &d_stmts[i];
                let d_bundle_wits = &d_wits[i];

                // Per-inner-stmt evaluations.
                let values: Vec<Vec<Fr>> = d_bundle_wits
                    .iter()
                    .map(|w| w.polynomials.iter().map(|p| p.evaluate(&x)).collect())
                    .collect();

                let mut t_pc = transcript.fork(b"fsmt_pcc::pc");
                t_pc.absorb_usize(b"leaf_idx", i);
                for vs in &values {
                    for v in vs {
                        t_pc.absorb(b"y", v);
                    }
                }
                let r: Fr = t_pc.squeeze_field(b"r");

                // RLC across all (inner_stmt × poly) pairs.
                let mut r_pow = Fr::from(1u64);
                let mut combined_poly = SparsePolynomial::<Fr>::zero();
                let mut combined_comm = Comm::zero();
                let mut combined_value = Fr::from(0u64);
                for ((d_stmt, d_wit), vs) in
                    d_bundle_stmts.iter().zip(d_bundle_wits).zip(&values)
                {
                    for ((p, c), &y) in d_wit
                        .polynomials
                        .iter()
                        .zip(&d_stmt.commitments)
                        .zip(vs)
                    {
                        let scaled = p * r_pow;
                        combined_poly = &combined_poly + &scaled;
                        combined_comm += *c * r_pow;
                        combined_value += y * r_pow;
                        r_pow *= r;
                    }
                }

                let pco_stmt = PcoStatement {
                    commitment: combined_comm,
                    point: x,
                    value: combined_value,
                };
                let pco_wit = PcoWitness { polynomial: combined_poly };

                let proof = FsMtPccProof {
                    shifted_commitments: dt_proofs[i].clone(),
                    values,
                    membership: tree.membership_proof(i),
                    root,
                };
                (proof, pco_stmt, pco_wit)
            })
            .collect()
    }

    /// Per-leaf verifier. Takes the leaf's bundle of inner `R_PCC` stmts.
    /// Returns `None` on Merkle failure or on any `Q_j(x, y) ≠ 0`.
    pub fn verify(
        params: &PccParams,
        index: usize,
        stmts: &[PccStatement],
        proof: &FsMtPccProof,
        transcript: &mut Blake3Transcript,
    ) -> Option<PcoStatement> {
        let bundle_size = stmts.len();
        if proof.shifted_commitments.len() != bundle_size
            || proof.values.len() != bundle_size
        {
            return None;
        }

        // 1. Reconstruct the κ inner PCC-D statements via Π_DT.verify.
        let mut d_stmts: Vec<PccDStatement> = Vec::with_capacity(bundle_size);
        for (inner_idx, (s, sc)) in stmts.iter().zip(&proof.shifted_commitments).enumerate() {
            let mut t_dt = transcript.fork(b"fsmt_pcc::dt");
            t_dt.absorb_usize(b"leaf_idx", index);
            t_dt.absorb_usize(b"inner_idx", inner_idx);
            let dt_proof = RokDtProof { shifted_commitments: sc.clone() };
            let d_stmt = RokDt::verify(params, s, &dt_proof, &mut t_dt);
            d_stmts.push(d_stmt);
        }

        // 2. Verify Merkle membership of the bundle.
        let leaf_hash = hash_leaf(transcript, index, &d_stmts);
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

        // 3. Absorb root, derive shared x.
        transcript.absorb_bytes(b"fsmt_pcc::root", &proof.root);
        let x: Fr = transcript.squeeze_field(b"fsmt_pcc::x");

        // 4. Per inner stmt: Schwartz–Zippel constraint check against that
        //    statement's slice of y values.
        for (d_stmt, vs) in d_stmts.iter().zip(&proof.values) {
            if vs.len() != d_stmt.commitments.len() {
                return None;
            }
            for q in &d_stmt.constraints {
                if !evaluate_constraint(q, x, vs).is_zero() {
                    return None;
                }
            }
        }

        // 5. Per-leaf r squeeze.
        let mut t_pc = transcript.fork(b"fsmt_pcc::pc");
        t_pc.absorb_usize(b"leaf_idx", index);
        for vs in &proof.values {
            for v in vs {
                t_pc.absorb(b"y", v);
            }
        }
        let r: Fr = t_pc.squeeze_field(b"r");

        // 6. RLC of commitments and values across the bundle.
        let mut r_pow = Fr::from(1u64);
        let mut combined_comm = Comm::zero();
        let mut combined_value = Fr::from(0u64);
        for (d_stmt, vs) in d_stmts.iter().zip(&proof.values) {
            for (c, &y) in d_stmt.commitments.iter().zip(vs) {
                combined_comm += *c * r_pow;
                combined_value += y * r_pow;
                r_pow *= r;
            }
        }

        Some(PcoStatement {
            commitment: combined_comm,
            point: x,
            value: combined_value,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hash_leaf(parent: &Blake3Transcript, index: usize, bundle: &[PccDStatement]) -> [u8; 32] {
    let mut t = parent.fork(b"fsmt_pcc::leaf");
    t.absorb_usize(b"index", index);
    t.absorb_usize(b"bundle_size", bundle.len());
    for d_stmt in bundle {
        absorb_pcc_d_statement(&mut t, d_stmt);
    }
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
    use ark_poly::univariate::SparsePolynomial;
    use ark_std::test_rng;

    /// One PCC instance with constraint `Y_0 − Y_1 = 0` over constants
    /// `p_0 = a, p_1 = a`.
    fn make_inner(srs: &Srs, a: u64) -> (PccStatement, PccWitness) {
        let p0 = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(a))]);
        let p1 = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(a))]);
        let c0 = pc::commit(srs, &Poly::Sparse(p0.clone()));
        let c1 = pc::commit(srs, &Poly::Sparse(p1.clone()));
        let stmt = PccStatement {
            commitments: vec![c0, c1],
            degrees: vec![1, 1],
            constraints: vec![Constraint {
                monomials: vec![
                    Monomial { coeff:  Fr::from(1u64), x_deg: 0, y_terms: vec![(0, 1)] },
                    Monomial { coeff: -Fr::from(1u64), x_deg: 0, y_terms: vec![(1, 1)] },
                ],
            }],
        };
        let wit = PccWitness { polynomials: vec![p0, p1] };
        (stmt, wit)
    }

    /// Build K leaves, each carrying a bundle of `kappa_inner` inner PCC stmts.
    fn k_leaves_nested(
        k: usize,
        kappa_inner: usize,
    ) -> (PccParams, Vec<Vec<PccStatement>>, Vec<Vec<PccWitness>>) {
        let rng = &mut test_rng();
        let srs = pc::setup(20, rng);
        let mut stmts = Vec::with_capacity(k);
        let mut wits = Vec::with_capacity(k);
        for i in 0..k {
            let mut bundle_s = Vec::with_capacity(kappa_inner);
            let mut bundle_w = Vec::with_capacity(kappa_inner);
            for j in 0..kappa_inner {
                let (s, w) = make_inner(&srs, (i * kappa_inner + j + 1) as u64);
                bundle_s.push(s);
                bundle_w.push(w);
            }
            stmts.push(bundle_s);
            wits.push(bundle_w);
        }
        (PccParams { srs }, stmts, wits)
    }

    /// K=4, κ_inner=1: round-trip with the minimal bundle shape.
    #[test]
    fn amortized_roundtrip_inner1() {
        let (params, stmts, wits) = k_leaves_nested(4, 1);
        let mut t = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t);

        let pco_params = PcoParams { srs: params.srs.clone() };
        for (_, pco_stmt, pco_wit) in &outputs {
            assert!(PcoRelation::is_satisfied(&pco_params, pco_stmt, pco_wit));
        }
    }

    /// K=4, κ_inner=3: nested input (3 inner stmts per leaf) works end to end.
    #[test]
    fn amortized_roundtrip_inner3() {
        let (params, stmts, wits) = k_leaves_nested(4, 3);
        let mut t = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t);

        let pco_params = PcoParams { srs: params.srs.clone() };
        for (_, pco_stmt, pco_wit) in &outputs {
            assert!(PcoRelation::is_satisfied(&pco_params, pco_stmt, pco_wit));
        }
    }

    /// Every output's `point` equals the shared `x`.
    #[test]
    fn all_leaves_share_x() {
        let (params, stmts, wits) = k_leaves_nested(4, 2);
        let mut t = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t);
        let x = outputs[0].1.point;
        for (_, pco_stmt, _) in &outputs {
            assert_eq!(pco_stmt.point, x);
        }
    }

    /// Per-leaf verify reconstructs the prover's PCO statement.
    #[test]
    fn prover_verifier_agree() {
        let (params, stmts, wits) = k_leaves_nested(4, 2);
        let mut t_p = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);

        for (i, (proof, prover_stmt, _)) in outputs.iter().enumerate() {
            let mut t_v = Blake3Transcript::new(b"test");
            let verifier_stmt = FsMtPcc::verify(&params, i, &stmts[i], proof, &mut t_v)
                .expect("honest verify must succeed");
            assert_eq!(*prover_stmt, verifier_stmt);
        }
    }

    /// Wrong leaf index → membership fails.
    #[test]
    fn wrong_leaf_index_rejected() {
        let (params, stmts, wits) = k_leaves_nested(4, 1);
        let mut t_p = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);

        let proof_for_1 = &outputs[1].0;
        let mut t_v = Blake3Transcript::new(b"test");
        assert!(FsMtPcc::verify(&params, 0, &stmts[0], proof_for_1, &mut t_v).is_none());
    }

    /// Tampered root fails Merkle membership.
    #[test]
    fn tampered_root_rejected() {
        let (params, stmts, wits) = k_leaves_nested(4, 2);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);
        outputs[0].0.root[0] ^= 0xFF;

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(FsMtPcc::verify(&params, 0, &stmts[0], &outputs[0].0, &mut t_v).is_none());
    }

    /// Tampered y values fail the constraint check.
    #[test]
    fn tampered_y_rejected() {
        let (params, stmts, wits) = k_leaves_nested(4, 2);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);
        outputs[0].0.values[0][0] += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(FsMtPcc::verify(&params, 0, &stmts[0], &outputs[0].0, &mut t_v).is_none());
    }

    /// All outputs are valid PCO instances at a common point — ready to feed
    /// downstream `PcoFold`.
    #[test]
    fn outputs_ready_for_pco_fold() {
        let (params, stmts, wits) = k_leaves_nested(4, 2);
        let mut t = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t);

        let x = outputs[0].1.point;
        for (_, s, _) in &outputs {
            assert_eq!(s.point, x);
        }
        let pco_params = PcoParams { srs: params.srs.clone() };
        for (_, s, w) in &outputs {
            assert!(PcoRelation::is_satisfied(&pco_params, s, w));
        }
    }
}
