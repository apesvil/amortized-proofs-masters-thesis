use ark_bls12_381::Fr;

use crate::pc::{self, Comm, Poly};
use crate::relations::abc::{bilinear, AbcParams, AbcStatement, AbcWitness};
use crate::relations::pcc::{Constraint, Monomial, PccStatement, PccWitness};
use crate::transcript::Blake3Transcript;

/// `Π_{ABC} : R_{A,B,C}² → R_{A,B,C} × R_PCC` — the per-fold reduction
/// extended to three matrices `(A, B, C)`.
///
/// The N/T side (rational encoding of `u, v`) is identical to the single-
/// matrix `Π_M` reduction: matrix-independent products and a single γ.
/// What changes is the value side — instead of one cross-term `z` and one
/// folded `y*`, we emit three of each (`z_A, z_B, z_C` and `y_A*, y_B*,
/// y_C*`), with the same `γ` weighting them.
pub struct RokAbc;

/// What the prover sends to the verifier.
#[derive(Clone)]
pub struct RokAbcProof {
    /// Cross-terms `z_X = u_0ᵀ·X·v_1 + u_1ᵀ·X·v_0` for `X ∈ {A, B, C}`.
    pub z_a: Fr,
    pub z_b: Fr,
    pub z_c: Fr,
    pub c_n_u_star: Comm,
    pub c_t_u_star: Comm,
    pub c_n_v_star: Comm,
    pub c_t_v_star: Comm,
}

impl RokAbc {
    /// Prover side. Returns the proof, the folded R_{A,B,C} instance, and
    /// the R_PCC promise.
    pub fn reduce(
        params: &AbcParams,
        s0: &AbcStatement,
        w0: &AbcWitness,
        s1: &AbcStatement,
        w1: &AbcWitness,
        transcript: &mut Blake3Transcript,
    ) -> (
        RokAbcProof,
        (AbcStatement, AbcWitness),
        (PccStatement, PccWitness),
    ) {
        absorb_abc_statement(transcript, b"s0", s0);
        absorb_abc_statement(transcript, b"s1", s1);

        // 1. Cross-terms, one per matrix.
        let z_a = bilinear(&params.matrix_a, &w0.u, &w1.v)
            + bilinear(&params.matrix_a, &w1.u, &w0.v);
        let z_b = bilinear(&params.matrix_b, &w0.u, &w1.v)
            + bilinear(&params.matrix_b, &w1.u, &w0.v);
        let z_c = bilinear(&params.matrix_c, &w0.u, &w1.v)
            + bilinear(&params.matrix_c, &w1.u, &w0.v);
        transcript.absorb(b"z_a", &z_a);
        transcript.absorb(b"z_b", &z_b);
        transcript.absorb(b"z_c", &z_c);

        // FS challenge.
        let gamma: Fr = transcript.squeeze_field(b"gamma");

        // 2. N/T folding — identical to Π_M (matrix-independent).
        // Workaround for arkworks 0.4.2 AddAssign<(F, &SparsePolynomial)> bug:
        // multiply first, then add.
        let n_u_star = &w0.n_u.mul(&w1.t_u) + &(&w1.n_u.mul(&w0.t_u) * gamma);
        let t_u_star = w0.t_u.mul(&w1.t_u);
        let n_v_star = &w0.n_v.mul(&w1.t_v) + &(&w1.n_v.mul(&w0.t_v) * gamma);
        let t_v_star = w0.t_v.mul(&w1.t_v);

        // 3. Folded u*, v*.
        let u_star: Vec<Fr> =
            w0.u.iter().zip(&w1.u).map(|(a, b)| *a + gamma * *b).collect();
        let v_star: Vec<Fr> =
            w0.v.iter().zip(&w1.v).map(|(a, b)| *a + gamma * *b).collect();

        // 4. Commitments to the four new polynomials.
        let c_n_u_star = pc::commit(&params.srs, &Poly::Sparse(n_u_star.clone()));
        let c_t_u_star = pc::commit(&params.srs, &Poly::Sparse(t_u_star.clone()));
        let c_n_v_star = pc::commit(&params.srs, &Poly::Sparse(n_v_star.clone()));
        let c_t_v_star = pc::commit(&params.srs, &Poly::Sparse(t_v_star.clone()));

        // 5. Folded degrees (same as Π_M).
        let (d_n_u_star, d_t_u_star, d_n_v_star, d_t_v_star) = star_degrees(s0, s1);

        // 6. Folded values, one per matrix.
        let gamma_sq = gamma * gamma;
        let y_a_star = s0.y_a + gamma * z_a + gamma_sq * s1.y_a;
        let y_b_star = s0.y_b + gamma * z_b + gamma_sq * s1.y_b;
        let y_c_star = s0.y_c + gamma * z_c + gamma_sq * s1.y_c;

        let out_abc_stmt = AbcStatement {
            c_n_u: c_n_u_star,
            c_t_u: c_t_u_star,
            c_n_v: c_n_v_star,
            c_t_v: c_t_v_star,
            d_n_u: d_n_u_star,
            d_t_u: d_t_u_star,
            d_n_v: d_n_v_star,
            d_t_v: d_t_v_star,
            y_a: y_a_star,
            y_b: y_b_star,
            y_c: y_c_star,
        };
        let out_abc_wit = AbcWitness {
            n_u: n_u_star.clone(),
            t_u: t_u_star.clone(),
            u: u_star,
            n_v: n_v_star.clone(),
            t_v: t_v_star.clone(),
            v: v_star,
        };

        // 7. R_PCC promise (matrix-independent — same shape as Π_M).
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

        let proof = RokAbcProof {
            z_a,
            z_b,
            z_c,
            c_n_u_star,
            c_t_u_star,
            c_n_v_star,
            c_t_v_star,
        };

        absorb_proof_round2(transcript, &proof);

        (proof, (out_abc_stmt, out_abc_wit), (out_pcc_stmt, out_pcc_wit))
    }

    /// Verifier side. Reconstructs the folded R_{A,B,C} statement and the
    /// R_PCC promise from the proof.
    pub fn verify(
        s0: &AbcStatement,
        s1: &AbcStatement,
        proof: &RokAbcProof,
        transcript: &mut Blake3Transcript,
    ) -> (AbcStatement, PccStatement) {
        absorb_abc_statement(transcript, b"s0", s0);
        absorb_abc_statement(transcript, b"s1", s1);
        transcript.absorb(b"z_a", &proof.z_a);
        transcript.absorb(b"z_b", &proof.z_b);
        transcript.absorb(b"z_c", &proof.z_c);
        let gamma: Fr = transcript.squeeze_field(b"gamma");

        let (d_n_u_star, d_t_u_star, d_n_v_star, d_t_v_star) = star_degrees(s0, s1);
        let gamma_sq = gamma * gamma;
        let y_a_star = s0.y_a + gamma * proof.z_a + gamma_sq * s1.y_a;
        let y_b_star = s0.y_b + gamma * proof.z_b + gamma_sq * s1.y_b;
        let y_c_star = s0.y_c + gamma * proof.z_c + gamma_sq * s1.y_c;

        let out_abc_stmt = AbcStatement {
            c_n_u: proof.c_n_u_star,
            c_t_u: proof.c_t_u_star,
            c_n_v: proof.c_n_v_star,
            c_t_v: proof.c_t_v_star,
            d_n_u: d_n_u_star,
            d_t_u: d_t_u_star,
            d_n_v: d_n_v_star,
            d_t_v: d_t_v_star,
            y_a: y_a_star,
            y_b: y_b_star,
            y_c: y_c_star,
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

        absorb_proof_round2(transcript, proof);

        (out_abc_stmt, out_pcc_stmt)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn absorb_abc_statement(t: &mut Blake3Transcript, tag: &'static [u8], s: &AbcStatement) {
    t.absorb_bytes(b"abc_stmt::tag", tag);
    t.absorb(b"c_n_u", &s.c_n_u);
    t.absorb(b"c_t_u", &s.c_t_u);
    t.absorb(b"c_n_v", &s.c_n_v);
    t.absorb(b"c_t_v", &s.c_t_v);
    t.absorb_usize(b"d_n_u", s.d_n_u);
    t.absorb_usize(b"d_t_u", s.d_t_u);
    t.absorb_usize(b"d_n_v", s.d_n_v);
    t.absorb_usize(b"d_t_v", s.d_t_v);
    t.absorb(b"y_a", &s.y_a);
    t.absorb(b"y_b", &s.y_b);
    t.absorb(b"y_c", &s.y_c);
}

fn absorb_proof_round2(t: &mut Blake3Transcript, p: &RokAbcProof) {
    t.absorb(b"c_n_u_star", &p.c_n_u_star);
    t.absorb(b"c_t_u_star", &p.c_t_u_star);
    t.absorb(b"c_n_v_star", &p.c_n_v_star);
    t.absorb(b"c_t_v_star", &p.c_t_v_star);
}

fn star_degrees(s0: &AbcStatement, s1: &AbcStatement) -> (usize, usize, usize, usize) {
    let d_n_u_star = (s0.d_n_u + s1.d_t_u).max(s1.d_n_u + s0.d_t_u);
    let d_t_u_star = s0.d_t_u + s1.d_t_u;
    let d_n_v_star = (s0.d_n_v + s1.d_t_v).max(s1.d_n_v + s0.d_t_v);
    let d_t_v_star = s0.d_t_v + s1.d_t_v;
    (d_n_u_star, d_t_u_star, d_n_v_star, d_t_v_star)
}

/// Build the R_PCC statement that promises the folded N/T polynomials
/// match the per-matrix combination of the inputs. Same shape as in `Π_M`
/// — index convention:
///   0 = N_u*    1 = T_u*    2 = N_v*    3 = T_v*
///   4 = N_{u,0} 5 = T_{u,0} 6 = N_{v,0} 7 = T_{v,0}
///   8 = N_{u,1} 9 = T_{u,1} 10 = N_{v,1} 11 = T_{v,1}
fn build_pcc_statement(
    s0: &AbcStatement,
    s1: &AbcStatement,
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
    use crate::relations::abc::{leaf_instance, AbcRelation};
    use crate::relations::pcc::{PccParams, PccRelation};
    use ark_ff::UniformRand;
    use ark_poly::univariate::SparsePolynomial;
    use ark_std::test_rng;

    fn two_leaves(n: usize) -> (AbcParams, (AbcStatement, AbcWitness), (AbcStatement, AbcWitness)) {
        let rng = &mut test_rng();
        let a: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        let b: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, (i + 1) % n, Fr::from(1u64))).collect();
        let c: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
        let srs = pc::setup(n + 1, rng);
        let params = AbcParams { srs, matrix_a: a, matrix_b: b, matrix_c: c, n };

        let (s0, w0) = leaf_instance(&params, Fr::rand(rng), Fr::rand(rng));
        let (s1, w1) = leaf_instance(&params, Fr::rand(rng), Fr::rand(rng));
        (params, (s0, w0), (s1, w1))
    }

    #[test]
    fn reduce_roundtrip() {
        let (params, (s0, w0), (s1, w1)) = two_leaves(4);
        let mut transcript = Blake3Transcript::new(b"test");

        let (_proof, (out_s, out_w), (out_pcc_s, out_pcc_w)) =
            RokAbc::reduce(&params, &s0, &w0, &s1, &w1, &mut transcript);

        let pcc_params = PccParams { srs: params.srs.clone() };
        assert!(AbcRelation::is_satisfied(&params, &out_s, &out_w));
        assert!(PccRelation::is_satisfied(&pcc_params, &out_pcc_s, &out_pcc_w));
    }

    #[test]
    fn prover_verifier_agree() {
        let (params, (s0, w0), (s1, w1)) = two_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");

        let (proof, (s_p, _), (pcc_p, _)) =
            RokAbc::reduce(&params, &s0, &w0, &s1, &w1, &mut t_p);
        let (s_v, pcc_v) = RokAbc::verify(&s0, &s1, &proof, &mut t_v);

        assert_eq!(s_p, s_v);
        assert_eq!(pcc_p, pcc_v);
    }

    #[test]
    fn tampered_z_a_breaks_abc() {
        let (params, (s0, w0), (s1, w1)) = two_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");

        let (mut proof, (_, wit), _) =
            RokAbc::reduce(&params, &s0, &w0, &s1, &w1, &mut t_p);
        proof.z_a += Fr::from(1u64);

        let (s_v, _) = RokAbc::verify(&s0, &s1, &proof, &mut t_v);
        assert!(!AbcRelation::is_satisfied(&params, &s_v, &wit));
    }

    #[test]
    fn tampered_z_b_breaks_abc() {
        let (params, (s0, w0), (s1, w1)) = two_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");

        let (mut proof, (_, wit), _) =
            RokAbc::reduce(&params, &s0, &w0, &s1, &w1, &mut t_p);
        proof.z_b += Fr::from(1u64);

        let (s_v, _) = RokAbc::verify(&s0, &s1, &proof, &mut t_v);
        assert!(!AbcRelation::is_satisfied(&params, &s_v, &wit));
    }

    #[test]
    fn tampered_z_c_breaks_abc() {
        let (params, (s0, w0), (s1, w1)) = two_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");

        let (mut proof, (_, wit), _) =
            RokAbc::reduce(&params, &s0, &w0, &s1, &w1, &mut t_p);
        proof.z_c += Fr::from(1u64);

        let (s_v, _) = RokAbc::verify(&s0, &s1, &proof, &mut t_v);
        assert!(!AbcRelation::is_satisfied(&params, &s_v, &wit));
    }

    #[test]
    fn tampered_commitment_breaks_pcc() {
        let (params, (s0, w0), (s1, w1)) = two_leaves(4);
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");

        let (mut proof, _, (_, pcc_wit)) =
            RokAbc::reduce(&params, &s0, &w0, &s1, &w1, &mut t_p);
        let bogus =
            SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(123u64))]);
        proof.c_n_u_star = pc::commit(&params.srs, &Poly::Sparse(bogus));

        let (_, pcc_v) = RokAbc::verify(&s0, &s1, &proof, &mut t_v);
        let pcc_params = PccParams { srs: params.srs.clone() };
        assert!(!PccRelation::is_satisfied(&pcc_params, &pcc_v, &pcc_wit));
    }

    #[test]
    fn different_label_different_challenge() {
        let (params, (s0, w0), (s1, w1)) = two_leaves(4);
        let mut t_a = Blake3Transcript::new(b"label_a");
        let mut t_b = Blake3Transcript::new(b"label_b");

        let (proof_a, _, _) = RokAbc::reduce(&params, &s0, &w0, &s1, &w1, &mut t_a);
        let (proof_b, _, _) = RokAbc::reduce(&params, &s0, &w0, &s1, &w1, &mut t_b);

        assert_eq!(proof_a.z_a, proof_b.z_a);
        assert_eq!(proof_a.z_b, proof_b.z_b);
        assert_eq!(proof_a.z_c, proof_b.z_c);
        assert_ne!(proof_a.c_n_u_star, proof_b.c_n_u_star);
    }
}
