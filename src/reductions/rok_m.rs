use ark_bls12_381::Fr;

use crate::pc::{self, Comm, Poly};
use crate::relations::m::{MParams, MStatement, MWitness};
use crate::relations::pcc::{Constraint, Monomial, PccStatement, PccWitness};
use crate::transcript::Blake3Transcript;

/// The non-interactive reduction of knowledge `Π_M : R_M² → R_M × R_PCC`
/// (Fiat-Shamir transform of the public-coin protocol in Sec. 8 / Fig.
/// `rok_pcss` of the paper, applied per Sec. `fs_rok`).
///
/// Folds two `R_M` instances `(s_b, w_b)_{b∈{0,1}}` into one new `R_M`
/// instance plus an `R_PCC` "promise" that witnesses the algebraic
/// relationship between the new committed polynomials and the originals.
/// The challenge `γ` is derived from the transcript: prover and verifier
/// each absorb `(s_0, s_1, z)` and squeeze the same `γ`.
pub struct RokM;

/// What the prover sends to the verifier.
pub struct RokMProof {
    /// `z = u_0ᵀ M v_1 + u_1ᵀ M v_0` — the cross-term used in the new value.
    pub z: Fr,
    pub c_n_u_star: Comm,
    pub c_t_u_star: Comm,
    pub c_n_v_star: Comm,
    pub c_t_v_star: Comm,
}

impl RokM {
    /// Prover side. Returns the proof, the new `R_M` instance, and the
    /// `R_PCC` promise instance, each as a `(statement, witness)` pair.
    /// The challenge `γ` is derived from `transcript` after absorbing the
    /// public inputs and the prover's first message `z`.
    pub fn reduce(
        params: &MParams,
        s0: &MStatement,
        w0: &MWitness,
        s1: &MStatement,
        w1: &MWitness,
        transcript: &mut Blake3Transcript,
    ) -> (
        RokMProof,
        (MStatement, MWitness),
        (PccStatement, PccWitness),
    ) {
        // FS round 0: absorb the public inputs (the two M statements).
        absorb_m_statement(transcript, b"s0", s0);
        absorb_m_statement(transcript, b"s1", s1);

        // 1. Cross-term  z = u_0ᵀ M v_1 + u_1ᵀ M v_0  (prover's round-1 msg).
        let z = bilinear(&params.matrix, &w0.u, &w1.v)
            + bilinear(&params.matrix, &w1.u, &w0.v);
        transcript.absorb(b"z", &z);

        // FS challenge: γ depends on (s_0, s_1, z).
        let gamma: Fr = transcript.squeeze_field(b"gamma");

        // 2. New polynomials.
        //    N_u* = N_{u,0}·T_{u,1} + γ · N_{u,1}·T_{u,0}
        //    T_u* = T_{u,0}·T_{u,1}
        //
        // Note: we deliberately avoid arkworks' `AddAssign<(F, &Self)>` impl on
        // SparsePolynomial — in 0.4.2 it computes `f · (self + other)` rather
        // than `self + f · other` (the result coefficients are scaled
        // unconditionally, see ark-poly src/.../univariate/sparse.rs:169).
        let n_u_star = &w0.n_u.mul(&w1.t_u) + &(&w1.n_u.mul(&w0.t_u) * gamma);
        let t_u_star = w0.t_u.mul(&w1.t_u);

        let n_v_star = &w0.n_v.mul(&w1.t_v) + &(&w1.n_v.mul(&w0.t_v) * gamma);
        let t_v_star = w0.t_v.mul(&w1.t_v);

        // 3. New vectors:  u* = u_0 + γ u_1,  v* = v_0 + γ v_1.
        let u_star: Vec<Fr> =
            w0.u.iter().zip(&w1.u).map(|(a, b)| *a + gamma * *b).collect();
        let v_star: Vec<Fr> =
            w0.v.iter().zip(&w1.v).map(|(a, b)| *a + gamma * *b).collect();

        // 4. Commitments to the four new polynomials.
        let c_n_u_star = pc::commit(&params.srs, &Poly::Sparse(n_u_star.clone()));
        let c_t_u_star = pc::commit(&params.srs, &Poly::Sparse(t_u_star.clone()));
        let c_n_v_star = pc::commit(&params.srs, &Poly::Sparse(n_v_star.clone()));
        let c_t_v_star = pc::commit(&params.srs, &Poly::Sparse(t_v_star.clone()));

        // 5. New degree bounds.
        let (d_n_u_star, d_t_u_star, d_n_v_star, d_t_v_star) = star_degrees(s0, s1);

        // 6. y* = y_0 + γ z + γ² y_1.
        let y_star = s0.value + gamma * z + gamma * gamma * s1.value;

        // 7. Output `R_M` (statement, witness).
        let out_m_stmt = MStatement {
            c_n_u: c_n_u_star,
            c_t_u: c_t_u_star,
            c_n_v: c_n_v_star,
            c_t_v: c_t_v_star,
            d_n_u: d_n_u_star,
            d_t_u: d_t_u_star,
            d_n_v: d_n_v_star,
            d_t_v: d_t_v_star,
            value: y_star,
        };
        let out_m_wit = MWitness {
            n_u: n_u_star.clone(),
            t_u: t_u_star.clone(),
            u: u_star,
            n_v: n_v_star.clone(),
            t_v: t_v_star.clone(),
            v: v_star,
        };

        // 8. Output `R_PCC` (statement, witness).
        let out_pcc_stmt = build_pcc_statement(
            s0,
            s1,
            (c_n_u_star, c_t_u_star, c_n_v_star, c_t_v_star),
            (d_n_u_star, d_t_u_star, d_n_v_star, d_t_v_star),
            gamma,
        );
        let out_pcc_wit = PccWitness {
            polynomials: vec![
                n_u_star,
                t_u_star,
                n_v_star,
                t_v_star,
                w0.n_u.clone(),
                w0.t_u.clone(),
                w0.n_v.clone(),
                w0.t_v.clone(),
                w1.n_u.clone(),
                w1.t_u.clone(),
                w1.n_v.clone(),
                w1.t_v.clone(),
            ],
        };

        let proof = RokMProof {
            z,
            c_n_u_star,
            c_t_u_star,
            c_n_v_star,
            c_t_v_star,
        };

        // Absorb the prover's round-2 messages so the transcript is in a
        // well-defined state for any downstream composition.
        absorb_proof_round2(transcript, &proof);

        (proof, (out_m_stmt, out_m_wit), (out_pcc_stmt, out_pcc_wit))
    }

    /// Verifier side. Reconstructs the two output statements deterministically
    /// from the input statements and the proof. The challenge `γ` is derived
    /// from `transcript` (which must mirror the prover's transcript state) by
    /// absorbing `(s_0, s_1, z)` and squeezing.
    ///
    /// There is no boolean to return — soundness comes from the *next* layer
    /// (calling `is_satisfied` on the reconstructed `R_M` and `R_PCC`
    /// statements against whichever witnesses are eventually produced).
    pub fn verify(
        s0: &MStatement,
        s1: &MStatement,
        proof: &RokMProof,
        transcript: &mut Blake3Transcript,
    ) -> (MStatement, PccStatement) {
        // Mirror the prover's FS absorbs to derive the same γ.
        absorb_m_statement(transcript, b"s0", s0);
        absorb_m_statement(transcript, b"s1", s1);
        transcript.absorb(b"z", &proof.z);
        let gamma: Fr = transcript.squeeze_field(b"gamma");

        let (d_n_u_star, d_t_u_star, d_n_v_star, d_t_v_star) = star_degrees(s0, s1);
        let y_star = s0.value + gamma * proof.z + gamma * gamma * s1.value;

        let out_m_stmt = MStatement {
            c_n_u: proof.c_n_u_star,
            c_t_u: proof.c_t_u_star,
            c_n_v: proof.c_n_v_star,
            c_t_v: proof.c_t_v_star,
            d_n_u: d_n_u_star,
            d_t_u: d_t_u_star,
            d_n_v: d_n_v_star,
            d_t_v: d_t_v_star,
            value: y_star,
        };
        let out_pcc_stmt = build_pcc_statement(
            s0,
            s1,
            (
                proof.c_n_u_star,
                proof.c_t_u_star,
                proof.c_n_v_star,
                proof.c_t_v_star,
            ),
            (d_n_u_star, d_t_u_star, d_n_v_star, d_t_v_star),
            gamma,
        );

        // Mirror the prover's round-2 absorbs.
        absorb_proof_round2(transcript, proof);

        (out_m_stmt, out_pcc_stmt)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bilinear(matrix: &[(usize, usize, Fr)], u: &[Fr], v: &[Fr]) -> Fr {
    matrix
        .iter()
        .map(|&(i, j, m_ij)| m_ij * u[i] * v[j])
        .sum()
}

/// Absorb an `MStatement` into the FS transcript with a section tag (so
/// `s_0` and `s_1` are distinguishable in the absorbed bytes).
fn absorb_m_statement(t: &mut Blake3Transcript, tag: &'static [u8], s: &MStatement) {
    t.absorb_bytes(b"m_stmt::tag", tag);
    t.absorb(b"c_n_u", &s.c_n_u);
    t.absorb(b"c_t_u", &s.c_t_u);
    t.absorb(b"c_n_v", &s.c_n_v);
    t.absorb(b"c_t_v", &s.c_t_v);
    t.absorb_usize(b"d_n_u", s.d_n_u);
    t.absorb_usize(b"d_t_u", s.d_t_u);
    t.absorb_usize(b"d_n_v", s.d_n_v);
    t.absorb_usize(b"d_t_v", s.d_t_v);
    t.absorb(b"value", &s.value);
}

/// Absorb the prover's round-2 message (the four new commitments). Done
/// after `γ` is squeezed so the transcript carries the full transcript
/// state for downstream composition.
fn absorb_proof_round2(t: &mut Blake3Transcript, p: &RokMProof) {
    t.absorb(b"c_n_u_star", &p.c_n_u_star);
    t.absorb(b"c_t_u_star", &p.c_t_u_star);
    t.absorb(b"c_n_v_star", &p.c_n_v_star);
    t.absorb(b"c_t_v_star", &p.c_t_v_star);
}

fn star_degrees(s0: &MStatement, s1: &MStatement) -> (usize, usize, usize, usize) {
    let d_n_u_star = (s0.d_n_u + s1.d_t_u).max(s1.d_n_u + s0.d_t_u);
    let d_t_u_star = s0.d_t_u + s1.d_t_u;
    let d_n_v_star = (s0.d_n_v + s1.d_t_v).max(s1.d_n_v + s0.d_t_v);
    let d_t_v_star = s0.d_t_v + s1.d_t_v;
    (d_n_u_star, d_t_u_star, d_n_v_star, d_t_v_star)
}

/// Build the `R_PCC` statement that promises:
///   N_u* = N_{u,0}·T_{u,1} + γ · N_{u,1}·T_{u,0}
///   N_v* = N_{v,0}·T_{v,1} + γ · N_{v,1}·T_{v,0}
///   T_u* = T_{u,0}·T_{u,1}
///   T_v* = T_{v,0}·T_{v,1}
///
/// Polynomial-index convention used in `Q`:
///   0 = N_u*    1 = T_u*    2 = N_v*    3 = T_v*
///   4 = N_{u,0} 5 = T_{u,0} 6 = N_{v,0} 7 = T_{v,0}
///   8 = N_{u,1} 9 = T_{u,1} 10 = N_{v,1} 11 = T_{v,1}
fn build_pcc_statement(
    s0: &MStatement,
    s1: &MStatement,
    star_comms: (Comm, Comm, Comm, Comm),
    star_degs: (usize, usize, usize, usize),
    gamma: Fr,
) -> PccStatement {
    let (c_n_u_star, c_t_u_star, c_n_v_star, c_t_v_star) = star_comms;
    let (d_n_u_star, d_t_u_star, d_n_v_star, d_t_v_star) = star_degs;

    let commitments = vec![
        c_n_u_star, c_t_u_star, c_n_v_star, c_t_v_star,
        s0.c_n_u, s0.c_t_u, s0.c_n_v, s0.c_t_v,
        s1.c_n_u, s1.c_t_u, s1.c_n_v, s1.c_t_v,
    ];
    // Convention bridge: `R_M` uses inclusive degree bounds (`deg ≤ d`, matching
    // the paper's `lemma_degrees`), while `R_PCC` uses paper-strict exclusive
    // bounds (`deg < d_i`, from compiler.tex). Translate by +1 at the boundary.
    let degrees = vec![
        d_n_u_star + 1, d_t_u_star + 1, d_n_v_star + 1, d_t_v_star + 1,
        s0.d_n_u + 1, s0.d_t_u + 1, s0.d_n_v + 1, s0.d_t_v + 1,
        s1.d_n_u + 1, s1.d_t_u + 1, s1.d_n_v + 1, s1.d_t_v + 1,
    ];
    let constraints = build_constraints(gamma);
    PccStatement { commitments, degrees, constraints }
}

/// The four constraints `Q_1, ..., Q_4`. Note: the paper writes
///   Q_1 = N_u* − N_{u,0}·T_{u,1} **+** γ · N_{u,1}·T_{u,0}
/// which combined with N_u* = N_{u,0}·T_{u,1} **+** γ · N_{u,1}·T_{u,0}
/// does *not* reduce to zero. The correct sign is `−γ`, used here.
fn build_constraints(gamma: Fr) -> Vec<Constraint> {
    let one = Fr::from(1u64);
    let mone = -one;
    let mgamma = -gamma;

    vec![
        // Q_1: Y_0 − Y_4·Y_9 − γ · Y_8·Y_5
        Constraint {
            monomials: vec![
                Monomial { coeff: one,    x_deg: 0, y_terms: vec![(0, 1)] },
                Monomial { coeff: mone,   x_deg: 0, y_terms: vec![(4, 1), (9, 1)] },
                Monomial { coeff: mgamma, x_deg: 0, y_terms: vec![(8, 1), (5, 1)] },
            ],
        },
        // Q_2: Y_2 − Y_6·Y_11 − γ · Y_10·Y_7
        Constraint {
            monomials: vec![
                Monomial { coeff: one,    x_deg: 0, y_terms: vec![(2, 1)] },
                Monomial { coeff: mone,   x_deg: 0, y_terms: vec![(6, 1), (11, 1)] },
                Monomial { coeff: mgamma, x_deg: 0, y_terms: vec![(10, 1), (7, 1)] },
            ],
        },
        // Q_3: Y_1 − Y_5·Y_9
        Constraint {
            monomials: vec![
                Monomial { coeff: one,  x_deg: 0, y_terms: vec![(1, 1)] },
                Monomial { coeff: mone, x_deg: 0, y_terms: vec![(5, 1), (9, 1)] },
            ],
        },
        // Q_4: Y_3 − Y_7·Y_11
        Constraint {
            monomials: vec![
                Monomial { coeff: one,  x_deg: 0, y_terms: vec![(3, 1)] },
                Monomial { coeff: mone, x_deg: 0, y_terms: vec![(7, 1), (11, 1)] },
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Relation;
    use crate::relations::m::{leaf_instance, MRelation};
    use crate::relations::pcc::{PccParams, PccRelation};
    use ark_ff::UniformRand;
    use ark_poly::univariate::SparsePolynomial;
    use ark_std::test_rng;

    /// Build params + two leaf statements/witnesses for the identity matrix M_n.
    /// The SRS is sized for the *folded* polynomials (deg n+1, the largest
    /// produced by the reduction), which also covers the leaves (deg n).
    fn two_identity_leaves(n: usize) -> (MParams, (MStatement, MWitness), (MStatement, MWitness)) {
        let rng = &mut test_rng();
        let matrix: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        // After folding, N_u* has degree n+1 (each N has degree n, each T has degree 1).
        let srs = pc::setup(n + 1, rng);
        let params = MParams { srs, matrix, n };

        let (s0, w0) = leaf_instance(&params, Fr::rand(rng), Fr::rand(rng));
        let (s1, w1) = leaf_instance(&params, Fr::rand(rng), Fr::rand(rng));
        (params, (s0, w0), (s1, w1))
    }

    /// Reduce two valid leaves; the new R_M and R_PCC outputs must both verify.
    #[test]
    fn reduce_roundtrip() {
        let (params, (s0, w0), (s1, w1)) = two_identity_leaves(4);
        let mut transcript = Blake3Transcript::new(b"test");

        let (_proof, (out_m_s, out_m_w), (out_pcc_s, out_pcc_w)) =
            RokM::reduce(&params, &s0, &w0, &s1, &w1, &mut transcript);

        let pcc_params = PccParams { srs: params.srs.clone() };
        assert!(MRelation::is_satisfied(&params, &out_m_s, &out_m_w));
        assert!(PccRelation::is_satisfied(&pcc_params, &out_pcc_s, &out_pcc_w));
    }

    /// Prover and verifier, given fresh transcripts with the same label,
    /// must reconstruct identical output statements.
    #[test]
    fn prover_verifier_agree() {
        let (params, (s0, w0), (s1, w1)) = two_identity_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");

        let (proof, (m_p, _), (pcc_p, _)) =
            RokM::reduce(&params, &s0, &w0, &s1, &w1, &mut t_p);
        let (m_v, pcc_v) = RokM::verify(&s0, &s1, &proof, &mut t_v);

        assert_eq!(m_p, m_v);
        assert_eq!(pcc_p, pcc_v);
    }

    /// Tampering `z` after `reduce` makes the verifier derive a different `γ`
    /// (since `γ` depends on `z`), so the reconstructed `y*` differs from
    /// what the prover's witness was built against — `is_satisfied` rejects.
    #[test]
    fn tampered_z_breaks_m() {
        let (params, (s0, w0), (s1, w1)) = two_identity_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");

        let (mut proof, (_, m_wit), _) =
            RokM::reduce(&params, &s0, &w0, &s1, &w1, &mut t_p);
        proof.z += Fr::from(1u64);

        let (m_v, _) = RokM::verify(&s0, &s1, &proof, &mut t_v);
        assert!(!MRelation::is_satisfied(&params, &m_v, &m_wit));
    }

    /// Tampering a commitment is absorbed *after* `γ` is squeezed, so it
    /// doesn't change `γ` — but it breaks the PCC commit-validity check
    /// because the recomputed commit no longer matches the tampered one.
    #[test]
    fn tampered_commitment_breaks_pcc() {
        let (params, (s0, w0), (s1, w1)) = two_identity_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");

        let (mut proof, _, (_, pcc_wit)) =
            RokM::reduce(&params, &s0, &w0, &s1, &w1, &mut t_p);
        let bogus =
            SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(123u64))]);
        proof.c_n_u_star = pc::commit(&params.srs, &Poly::Sparse(bogus));

        let (_, pcc_v) = RokM::verify(&s0, &s1, &proof, &mut t_v);
        let pcc_params = PccParams { srs: params.srs.clone() };
        assert!(!PccRelation::is_satisfied(&pcc_params, &pcc_v, &pcc_wit));
    }

    /// Different transcript labels yield different `γ`s — and therefore
    /// different new commitments — even with identical inputs. `z` is
    /// computed before any squeeze, so it stays the same.
    #[test]
    fn different_label_different_challenge() {
        let (params, (s0, w0), (s1, w1)) = two_identity_leaves(4);
        let mut t_a = Blake3Transcript::new(b"label_a");
        let mut t_b = Blake3Transcript::new(b"label_b");

        let (proof_a, _, _) = RokM::reduce(&params, &s0, &w0, &s1, &w1, &mut t_a);
        let (proof_b, _, _) = RokM::reduce(&params, &s0, &w0, &s1, &w1, &mut t_b);

        assert_eq!(proof_a.z, proof_b.z);
        assert_ne!(proof_a.c_n_u_star, proof_b.c_n_u_star);
    }
}
