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

/// `FsMt(Π_DT ∘ Π_PC, κ)` — the Merkle-tree / Fiat-Shamir transform applied
/// to the composed `R_PCC → R_PCC-D → R_PCO` chain.
///
/// Per-leaf input is an `R_PCC` instance; per-leaf output is a single
/// `R_PCO` instance at a *shared* evaluation point `x` (derived from the
/// Merkle root over the K per-leaf `R_PCC-D` statements). All `K · 1 = K`
/// outputs share `x` and feed directly into `PcoFold`.
pub struct FsMtPcc;

/// Per-leaf proof:
/// - `shifted_commitments`: the `Π_DT` prover message (n commitments).
/// - `values`: the `Π_PC` prover message (2n evaluations at x).
/// - `membership`: Merkle membership of this leaf's PCC-D statement.
/// - `root`: the Merkle root (so the verifier can independently verify
///   membership without reconstructing the tree).
pub struct FsMtPccProof {
    pub shifted_commitments: Vec<Comm>,
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
    ) -> Vec<(FsMtPccProof, PcoStatement, PcoWitness)> {
        assert_eq!(stmts.len(), wits.len());
        let k = stmts.len();
        assert!(
            k.is_power_of_two() && k >= 2,
            "K must be a power of two ≥ 2"
        );

        // 1. Per leaf, run Π_DT in a leaf-tagged fork (the fork's final
        //    state is discarded — Π_DT contributes only its deterministic
        //    output, not any FS state).
        let mut dt_proofs: Vec<RokDtProof> = Vec::with_capacity(k);
        let mut d_stmts: Vec<PccDStatement> = Vec::with_capacity(k);
        // PccDWitness owns polynomials — we move into a Vec.
        let mut d_wits: Vec<crate::relations::pcc_d::PccDWitness> = Vec::with_capacity(k);

        for (i, (s, w)) in stmts.iter().zip(wits).enumerate() {
            let mut t_dt = transcript.fork(b"fsmt_pcc::dt");
            t_dt.absorb_usize(b"leaf_idx", i);
            let (dt_proof, d_stmt, d_wit) = RokDt::reduce(params, s, w, &mut t_dt);
            dt_proofs.push(dt_proof);
            d_stmts.push(d_stmt);
            d_wits.push(d_wit);
        }

        // 2. Hash each PCC-D statement, build the Merkle tree, absorb the root.
        let leaf_hashes: Vec<[u8; 32]> = d_stmts
            .iter()
            .enumerate()
            .map(|(i, d)| hash_leaf(transcript, i, d))
            .collect();
        let tree = MerkleTree::from_leaf_hashes(&leaf_hashes);
        let root = tree.root();
        transcript.absorb_bytes(b"fsmt_pcc::root", &root);

        // 3. Squeeze the shared evaluation point.
        let x: Fr = transcript.squeeze_field(b"fsmt_pcc::x");

        // 4. Per-leaf Π_PC: evaluate at x, derive per-leaf r, compute RLC.
        (0..k)
            .map(|i| {
                let d_stmt = &d_stmts[i];
                let d_wit = &d_wits[i];

                let values: Vec<Fr> = d_wit
                    .polynomials
                    .iter()
                    .map(|p| p.evaluate(&x))
                    .collect();

                let mut t_pc = transcript.fork(b"fsmt_pcc::pc");
                t_pc.absorb_usize(b"leaf_idx", i);
                for v in &values {
                    t_pc.absorb(b"y", v);
                }
                let r: Fr = t_pc.squeeze_field(b"r");

                let (combined_poly, combined_comm, combined_value) =
                    rlc(&d_wit.polynomials, &d_stmt.commitments, &values, r);

                let pco_stmt = PcoStatement {
                    commitment: combined_comm,
                    point: x,
                    value: combined_value,
                };
                let pco_wit = PcoWitness { polynomial: combined_poly };

                let proof = FsMtPccProof {
                    shifted_commitments: dt_proofs[i].shifted_commitments.clone(),
                    values,
                    membership: tree.membership_proof(i),
                    root,
                };
                (proof, pco_stmt, pco_wit)
            })
            .collect()
    }

    /// Per-leaf verifier. Returns `None` on Merkle-membership failure or on
    /// the Schwartz–Zippel constraint check (`Q_j(x, values) ≠ 0`).
    pub fn verify(
        params: &PccParams,
        index: usize,
        stmt: &PccStatement,
        proof: &FsMtPccProof,
        transcript: &mut Blake3Transcript,
    ) -> Option<PcoStatement> {
        // 1. Reconstruct the leaf's PCC-D statement via Π_DT.verify
        //    (using the same fork-and-tag the prover did).
        let mut t_dt = transcript.fork(b"fsmt_pcc::dt");
        t_dt.absorb_usize(b"leaf_idx", index);
        let dt_proof = RokDtProof {
            shifted_commitments: proof.shifted_commitments.clone(),
        };
        let d_stmt = RokDt::verify(params, stmt, &dt_proof, &mut t_dt);

        // 2. Verify Merkle membership.
        let leaf_hash = hash_leaf(transcript, index, &d_stmt);
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

        // 3. Absorb root, derive shared x (same as the prover).
        transcript.absorb_bytes(b"fsmt_pcc::root", &proof.root);
        let x: Fr = transcript.squeeze_field(b"fsmt_pcc::x");

        // 4. Schwartz–Zippel constraint check.
        for q in &d_stmt.constraints {
            if !evaluate_constraint(q, x, &proof.values).is_zero() {
                return None;
            }
        }

        // 5. Derive per-leaf r.
        let mut t_pc = transcript.fork(b"fsmt_pcc::pc");
        t_pc.absorb_usize(b"leaf_idx", index);
        for v in &proof.values {
            t_pc.absorb(b"y", v);
        }
        let r: Fr = t_pc.squeeze_field(b"r");

        // 6. RLC on commitments and values.
        let mut r_pow = Fr::from(1u64);
        let mut combined_comm = Comm::zero();
        let mut combined_value = Fr::from(0u64);
        for (c, &y) in d_stmt.commitments.iter().zip(&proof.values) {
            combined_comm += *c * r_pow;
            combined_value += y * r_pow;
            r_pow *= r;
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

fn hash_leaf(parent: &Blake3Transcript, index: usize, d_stmt: &PccDStatement) -> [u8; 32] {
    let mut t = parent.fork(b"fsmt_pcc::leaf");
    t.absorb_usize(b"index", index);
    absorb_pcc_d_statement(&mut t, d_stmt);
    let mut out = [0u8; 32];
    t.squeeze_bytes(b"hash", &mut out);
    out
}

/// Random linear combination of polynomials, commitments, and values with
/// bases `r^k`.
fn rlc(
    polys: &[SparsePolynomial<Fr>],
    comms: &[Comm],
    values: &[Fr],
    r: Fr,
) -> (SparsePolynomial<Fr>, Comm, Fr) {
    let mut r_pow = Fr::from(1u64);
    let mut combined_poly = SparsePolynomial::<Fr>::zero();
    let mut combined_comm = Comm::zero();
    let mut combined_value = Fr::from(0u64);
    for ((p, c), &y) in polys.iter().zip(comms).zip(values) {
        let scaled_poly = p * r_pow;
        combined_poly = &combined_poly + &scaled_poly;
        combined_comm += *c * r_pow;
        combined_value += y * r_pow;
        r_pow *= r;
    }
    (combined_poly, combined_comm, combined_value)
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

    /// Build one PCC instance with constraint `Y_0 − Y_1 = 0` over constant
    /// polynomials `p_0 = a, p_1 = a` (so the constraint holds polynomially).
    fn make_leaf(srs: &Srs, a: u64) -> (PccStatement, PccWitness) {
        let p0 = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(a))]);
        let p1 = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(a))]);
        let c0 = pc::commit(srs, &Poly::Sparse(p0.clone()));
        let c1 = pc::commit(srs, &Poly::Sparse(p1.clone()));
        let stmt = PccStatement {
            commitments: vec![c0, c1],
            degrees: vec![1, 1], // paper-strict: deg < 1 ⇒ constants
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
        for (_, pco_stmt, pco_wit) in &outputs {
            assert!(PcoRelation::is_satisfied(&pco_params, pco_stmt, pco_wit));
        }
    }

    /// Every output's `point` equals the shared `x` (amortization invariant).
    #[test]
    fn all_leaves_share_x() {
        let (params, stmts, wits) = k_leaves(4);
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
        let (params, stmts, wits) = k_leaves(4);
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
        let (params, stmts, wits) = k_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);

        let proof_for_1 = &outputs[1].0;
        let mut t_v = Blake3Transcript::new(b"test");
        assert!(FsMtPcc::verify(&params, 0, &stmts[0], proof_for_1, &mut t_v).is_none());
    }

    /// Tampered root fails Merkle membership.
    #[test]
    fn tampered_root_rejected() {
        let (params, stmts, wits) = k_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);
        outputs[0].0.root[0] ^= 0xFF;

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(FsMtPcc::verify(&params, 0, &stmts[0], &outputs[0].0, &mut t_v).is_none());
    }

    /// Tampered y values fail the constraint check.
    #[test]
    fn tampered_y_rejected() {
        let (params, stmts, wits) = k_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut outputs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_p);
        outputs[0].0.values[0] += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(FsMtPcc::verify(&params, 0, &stmts[0], &outputs[0].0, &mut t_v).is_none());
    }

    /// The K per-leaf PCO outputs all share `x` and can be folded across
    /// leaves via `PcoFold` (we'll verify this once PcoFold lands; here we
    /// just check they're suitable input — same point, well-formed).
    #[test]
    fn outputs_ready_for_pco_fold() {
        let (params, stmts, wits) = k_leaves(4);
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
