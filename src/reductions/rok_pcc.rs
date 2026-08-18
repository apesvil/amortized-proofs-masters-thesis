use ark_bls12_381::Fr;
use ark_ff::{Field, Zero};
use ark_poly::Polynomial;

use crate::relations::pcc::Constraint;
use crate::relations::pcc_d::{PccDParams, PccDStatement, PccDWitness};
use crate::relations::pco::{PcoStatement, PcoWitness};
use crate::transcript::Blake3Transcript;

/// `Π_PC : R_PCC-D → R_PCO` (updated `figure_pccrok.tex`).
///
/// Two-challenge protocol:
/// ```text
///   x ← FS          (after absorbing the input statement)
///   y_k = p_k(x)    for every k in 0..2n  (sent by prover)
///   r ← FS          (after absorbing y)
///   p(X) = Σ r^k · p_k(X),  C = Σ r^k · C_k,  y = Σ r^k · y_k
/// ```
///
/// The verifier additionally checks `Q_j(x, y) = 0` for every constraint
/// (Schwartz–Zippel). The output is a **single** PCO instance `(C, x, y)`
/// with witness `p(X)`.
///
/// **Reference-only** — this standalone reduction is not reached by the
/// benchmarks. On the hot path `FsMtPcc` inlines Π_PC, reusing only the free
/// helpers `evaluate_constraint` / `absorb_pcc_d_statement` (below); `reduce`
/// and `verify` here exist for the unit tests and as the readable reference.
pub struct RokPcc;

/// Prover's wire message: the unshifted evaluation vector.
pub struct RokPccProof {
    pub values: Vec<Fr>,
}

impl RokPcc {
    /// Prover side.
    pub fn reduce(
        _params: &PccDParams,
        stmt: &PccDStatement,
        wit: &PccDWitness,
        transcript: &mut Blake3Transcript,
    ) -> (RokPccProof, PcoStatement, PcoWitness) {
        absorb_pcc_d_statement(transcript, stmt);
        let x: Fr = transcript.squeeze_field(b"rok_pcc::x");

        // y_k = p_k(x)
        let values: Vec<Fr> = wit.polynomials.iter().map(|p| p.evaluate(&x)).collect();
        for v in &values {
            transcript.absorb(b"rok_pcc::y", v);
        }

        let r: Fr = transcript.squeeze_field(b"rok_pcc::r");

        // RLC with bases `r^k`.
        let mut r_pow = Fr::from(1u64);
        let mut combined_poly = ark_poly::univariate::SparsePolynomial::<Fr>::zero();
        let mut combined_comm = crate::pc::Comm::zero();
        let mut combined_value = Fr::from(0u64);
        for ((p, c), &y) in wit
            .polynomials
            .iter()
            .zip(&stmt.commitments)
            .zip(&values)
        {
            let scaled_poly = p * r_pow;
            combined_poly = &combined_poly + &scaled_poly;
            combined_comm += *c * r_pow;
            combined_value += y * r_pow;
            r_pow *= r;
        }

        let out_stmt = PcoStatement {
            commitment: combined_comm,
            point: x,
            value: combined_value,
        };
        let out_wit = PcoWitness { polynomial: combined_poly };
        (RokPccProof { values }, out_stmt, out_wit)
    }

    /// Verifier side. Returns `None` iff some constraint fails the
    /// Schwartz–Zippel check `Q_j(x, values) = 0`.
    pub fn verify(
        _params: &PccDParams,
        stmt: &PccDStatement,
        proof: &RokPccProof,
        transcript: &mut Blake3Transcript,
    ) -> Option<PcoStatement> {
        absorb_pcc_d_statement(transcript, stmt);
        let x: Fr = transcript.squeeze_field(b"rok_pcc::x");

        // Soundness: each constraint must vanish at (x, values).
        for q in &stmt.constraints {
            if !evaluate_constraint(q, x, &proof.values).is_zero() {
                return None;
            }
        }

        for v in &proof.values {
            transcript.absorb(b"rok_pcc::y", v);
        }
        let r: Fr = transcript.squeeze_field(b"rok_pcc::r");

        let mut r_pow = Fr::from(1u64);
        let mut combined_comm = crate::pc::Comm::zero();
        let mut combined_value = Fr::from(0u64);
        for (c, &y) in stmt.commitments.iter().zip(&proof.values) {
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
// Helpers (exposed to siblings — used by `fsmt_pcc.rs` for the FsMt variant).
// ---------------------------------------------------------------------------

/// Numeric `Q(x, y_1, ..., y_n)`.
///
/// `x_deg` reaches `D − d_i` for the degree-shift constraints `Π_DT` appends
/// (see `rok_dt::shift_constraint`), i.e. it is on the order of the SRS size.
/// The exponentiation is therefore done by square-and-multiply — the same way
/// `rok_p::pow_usize` handles the identical shift exponents — not by a linear
/// multiply loop, which would make the verifier `O(D)` per shifted commitment.
pub(super) fn evaluate_constraint(q: &Constraint, x: Fr, y: &[Fr]) -> Fr {
    q.monomials
        .iter()
        .map(|m| {
            let mut term = m.coeff * x.pow([m.x_deg as u64]);
            for &(idx, exp) in &m.y_terms {
                for _ in 0..exp {
                    term *= y[idx];
                }
            }
            term
        })
        .sum()
}

pub(super) fn absorb_pcc_d_statement(t: &mut Blake3Transcript, s: &PccDStatement) {
    t.absorb_usize(b"n_commitments", s.commitments.len());
    for c in &s.commitments {
        t.absorb(b"commitment", c);
    }
    t.absorb_usize(b"n_constraints", s.constraints.len());
    for q in &s.constraints {
        absorb_constraint(t, q);
    }
}

fn absorb_constraint(t: &mut Blake3Transcript, q: &Constraint) {
    t.absorb_usize(b"n_monomials", q.monomials.len());
    for m in &q.monomials {
        t.absorb(b"coeff", &m.coeff);
        t.absorb_usize(b"x_deg", m.x_deg);
        t.absorb_usize(b"n_y_terms", m.y_terms.len());
        for &(idx, exp) in &m.y_terms {
            t.absorb_usize(b"y_idx", idx);
            t.absorb_usize(b"y_exp", exp);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Relation;
    use crate::pc::{self, Poly};
    use crate::relations::pcc::Monomial;
    use crate::relations::pco::{PcoParams, PcoRelation};
    use ark_poly::univariate::SparsePolynomial;
    use ark_std::test_rng;

    /// Build a small valid `R_PCC-D` instance: three polynomials with
    /// the constraint `Y_0 · Y_1 − Y_2 = 0`.
    fn product_pcc_d() -> (PccDParams, PccDStatement, PccDWitness) {
        let rng = &mut test_rng();
        let srs = pc::setup(20, rng);
        let p0 = SparsePolynomial::from_coefficients_vec(vec![
            (0, Fr::from(1u64)),
            (1, Fr::from(1u64)),
        ]);
        let p1 = SparsePolynomial::from_coefficients_vec(vec![
            (0, -Fr::from(1u64)),
            (2, Fr::from(1u64)),
        ]);
        let p2 = p0.mul(&p1);
        let commitments = vec![
            pc::commit(&srs, &Poly::Sparse(p0.clone())),
            pc::commit(&srs, &Poly::Sparse(p1.clone())),
            pc::commit(&srs, &Poly::Sparse(p2.clone())),
        ];
        let q = Constraint {
            monomials: vec![
                Monomial { coeff:  Fr::from(1u64), x_deg: 0, y_terms: vec![(0, 1), (1, 1)] },
                Monomial { coeff: -Fr::from(1u64), x_deg: 0, y_terms: vec![(2, 1)] },
            ],
        };
        let stmt = PccDStatement { commitments, constraints: vec![q] };
        let wit = PccDWitness { polynomials: vec![p0, p1, p2] };
        (PccDParams { srs }, stmt, wit)
    }

    /// Reduce a valid `R_PCC-D` instance; the output PCO `(stmt, wit)` must satisfy `R_PCO`.
    #[test]
    fn reduce_roundtrip() {
        let (params, stmt, wit) = product_pcc_d();
        let mut t = Blake3Transcript::new(b"test");
        let (_proof, out_stmt, out_wit) = RokPcc::reduce(&params, &stmt, &wit, &mut t);

        let pco_params = PcoParams { srs: params.srs.clone() };
        assert!(PcoRelation::is_satisfied(&pco_params, &out_stmt, &out_wit));
    }

    /// Prover and verifier reconstruct the same PCO statement.
    #[test]
    fn prover_verifier_agree() {
        let (params, stmt, wit) = product_pcc_d();
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");
        let (proof, out_p, _) = RokPcc::reduce(&params, &stmt, &wit, &mut t_p);
        let out_v = RokPcc::verify(&params, &stmt, &proof, &mut t_v)
            .expect("verify must succeed");
        assert_eq!(out_p, out_v);
    }

    /// Bumping a `y` value flips the constraint check.
    #[test]
    fn verify_rejects_tampered_y() {
        let (params, stmt, wit) = product_pcc_d();
        let mut t_p = Blake3Transcript::new(b"test");
        let (mut proof, _, _) = RokPcc::reduce(&params, &stmt, &wit, &mut t_p);
        proof.values[0] += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(RokPcc::verify(&params, &stmt, &proof, &mut t_v).is_none());
    }

    /// Different transcript labels yield different reductions.
    #[test]
    fn different_label_different_output() {
        let (params, stmt, wit) = product_pcc_d();
        let mut t_a = Blake3Transcript::new(b"label_a");
        let mut t_b = Blake3Transcript::new(b"label_b");
        let (_, out_a, _) = RokPcc::reduce(&params, &stmt, &wit, &mut t_a);
        let (_, out_b, _) = RokPcc::reduce(&params, &stmt, &wit, &mut t_b);
        assert_ne!(out_a, out_b);
    }

    /// A PCC-D with no constraints — verify always returns `Some(...)`.
    #[test]
    fn empty_constraints_works() {
        let rng = &mut test_rng();
        let srs = pc::setup(10, rng);
        let p = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(7u64))]);
        let c = pc::commit(&srs, &Poly::Sparse(p.clone()));
        let stmt = PccDStatement { commitments: vec![c], constraints: vec![] };
        let wit = PccDWitness { polynomials: vec![p] };
        let params = PccDParams { srs };

        let mut t_p = Blake3Transcript::new(b"test");
        let (proof, out_p, out_wit) = RokPcc::reduce(&params, &stmt, &wit, &mut t_p);
        let mut t_v = Blake3Transcript::new(b"test");
        let out_v = RokPcc::verify(&params, &stmt, &proof, &mut t_v).unwrap();
        assert_eq!(out_p, out_v);

        let pco_params = PcoParams { srs: params.srs };
        assert!(PcoRelation::is_satisfied(&pco_params, &out_p, &out_wit));
    }
}
