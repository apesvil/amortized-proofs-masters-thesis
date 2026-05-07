use ark_bls12_381::Fr;
use ark_ff::{Field, Zero};
use ark_poly::Polynomial;

use crate::pc::Comm;
use crate::relations::pcc::{Constraint, PccParams, PccStatement, PccWitness};
use crate::relations::pco::{PcoStatement, PcoWitness};
use crate::transcript::Blake3Transcript;

/// Non-interactive `Π_PCC : R_PCC → R_PCO^n`.
///
/// Given a PCC instance with `n` committed polynomials `p_i` (each
/// `deg p_i < d_i`) and constraints `Q_j` over them, the reduction outputs
/// `n` PCO instances all sharing a single random point `x`:
///
/// ```text
///   x      ← FS challenge after absorbing the input statement
///   y_i    = p_i(x)                          (sent by prover)
///   shift  = x^{D − d_i},  D = srs.powers_g1.len()
///   q_i(X) = shift · p_i(X)                  (PCO witness)
///   C'_i   = shift · C_i                     (PCO commitment, in G1)
///   y'_i   = shift · y_i                     (PCO value)
/// ```
///
/// The verifier additionally checks `Q_j(x, y) = 0` for every constraint
/// (Schwartz–Zippel) — see `verify` returning `Option<...>`.
///
/// Note on the figure (`figure_pccrok.tex`). The paper writes the shifted
/// witness as `p'_i(X) := X^{D−d_i} · p_i(X)` (an *X*-shift), which would
/// need a degree-bound-aware PC (e.g. shifted KZG with extra G2 powers) for
/// the verifier to derive the corresponding shifted commitment. Our `pc/`
/// is plain KZG, so we use the *scalar* shift `q_i := x^{D−d_i} · p_i`
/// instead — the only interpretation consistent with the verifier's
/// `C'_i = x^{D−d_i} · C_i`. See `docs/future_revisions.md`.
pub struct RokPcc;

/// What the prover sends across the wire: the unshifted evaluations
/// `y_i = p_i(x)`. The verifier shifts them itself when building the PCO
/// statements.
pub struct RokPccProof {
    pub values: Vec<Fr>,
}

impl RokPcc {
    /// Prover side. Output: one PCO `(statement, witness)` per input polynomial.
    pub fn reduce(
        params: &PccParams,
        stmt: &PccStatement,
        wit: &PccWitness,
        transcript: &mut Blake3Transcript,
    ) -> (RokPccProof, Vec<PcoStatement>, Vec<PcoWitness>) {
        absorb_pcc_statement(transcript, stmt);
        let x: Fr = transcript.squeeze_field(b"rok_pcc::x");

        // Unshifted evaluations, sent in the proof.
        let values: Vec<Fr> = wit.polynomials.iter().map(|p| p.evaluate(&x)).collect();
        for v in &values {
            transcript.absorb(b"rok_pcc::y", v);
        }

        let big_d = params.srs.powers_g1.len();
        let stmts: Vec<PcoStatement> = stmt
            .commitments
            .iter()
            .zip(&stmt.degrees)
            .zip(&values)
            .map(|((c, &d_i), &y)| {
                let shift = pow_fr(x, big_d - d_i);
                PcoStatement {
                    commitment: *c * shift,
                    point: x,
                    value: y * shift,
                }
            })
            .collect();
        let wits: Vec<PcoWitness> = wit
            .polynomials
            .iter()
            .zip(&stmt.degrees)
            .map(|(p, &d_i)| {
                let shift = pow_fr(x, big_d - d_i);
                PcoWitness { polynomial: p * shift }
            })
            .collect();

        (RokPccProof { values }, stmts, wits)
    }

    /// Verifier side. Returns `None` iff some constraint fails the numerical
    /// Schwartz–Zippel check `Q_j(x, values) = 0`.
    pub fn verify(
        params: &PccParams,
        stmt: &PccStatement,
        proof: &RokPccProof,
        transcript: &mut Blake3Transcript,
    ) -> Option<Vec<PcoStatement>> {
        absorb_pcc_statement(transcript, stmt);
        let x: Fr = transcript.squeeze_field(b"rok_pcc::x");

        // Soundness check: each constraint must vanish at (x, values).
        for q in &stmt.constraints {
            if !evaluate_constraint(q, x, &proof.values).is_zero() {
                return None;
            }
        }

        for v in &proof.values {
            transcript.absorb(b"rok_pcc::y", v);
        }

        let big_d = params.srs.powers_g1.len();
        Some(
            stmt.commitments
                .iter()
                .zip(&stmt.degrees)
                .zip(&proof.values)
                .map(|((c, &d_i), &y)| {
                    let shift = pow_fr(x, big_d - d_i);
                    PcoStatement {
                        commitment: *c * shift,
                        point: x,
                        value: y * shift,
                    }
                })
                .collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn pow_fr(base: Fr, exp: usize) -> Fr {
    base.pow([exp as u64])
}

/// Numeric `Q(x, y_1, ..., y_n)`.
fn evaluate_constraint(q: &Constraint, x: Fr, y: &[Fr]) -> Fr {
    q.monomials
        .iter()
        .map(|m| {
            let mut term = m.coeff;
            for _ in 0..m.x_deg {
                term *= x;
            }
            for &(idx, exp) in &m.y_terms {
                for _ in 0..exp {
                    term *= y[idx];
                }
            }
            term
        })
        .sum()
}

fn absorb_pcc_statement(t: &mut Blake3Transcript, s: &PccStatement) {
    t.absorb_usize(b"n_commitments", s.commitments.len());
    for c in &s.commitments {
        t.absorb(b"commitment", c);
    }
    t.absorb_usize(b"n_degrees", s.degrees.len());
    for d in &s.degrees {
        t.absorb_usize(b"degree", *d);
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

// Workaround: Comm = G1Projective. We need `Comm * Fr` -> Comm. The
// `&G1Projective * Fr` is... actually let's just check by writing it.
// (No code here — just stating the design.)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Relation;
    use crate::pc::{self, Poly};
    use crate::relations::pcc::{Monomial, PccRelation};
    use crate::relations::pco::{PcoParams, PcoRelation};
    use ark_poly::univariate::SparsePolynomial;
    use ark_std::test_rng;

    /// Build a small valid PCC instance with one constraint `Y_0·Y_1 − Y_2 = 0`
    /// over `p_0 = X+1, p_1 = X²−1, p_2 = (X+1)(X²−1)`.
    fn product_pcc() -> (PccParams, PccStatement, PccWitness) {
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
        let polys = vec![p0, p1, p2];
        let commitments: Vec<Comm> = polys
            .iter()
            .map(|p| pc::commit(&srs, &Poly::Sparse((*p).clone())))
            .collect();
        // Paper-strict bounds: deg < d_i.
        let degrees = vec![2, 3, 4];
        let q = Constraint {
            monomials: vec![
                Monomial {
                    coeff: Fr::from(1u64),
                    x_deg: 0,
                    y_terms: vec![(0, 1), (1, 1)],
                },
                Monomial {
                    coeff: -Fr::from(1u64),
                    x_deg: 0,
                    y_terms: vec![(2, 1)],
                },
            ],
        };
        let stmt = PccStatement {
            commitments,
            degrees,
            constraints: vec![q],
        };
        let wit = PccWitness { polynomials: polys };
        (PccParams { srs }, stmt, wit)
    }

    /// Reduce a valid PCC; each output PCO `(statement, witness)` must verify.
    #[test]
    fn reduce_roundtrip() {
        let (params, stmt, wit) = product_pcc();
        let mut transcript = Blake3Transcript::new(b"test");
        let (_proof, out_stmts, out_wits) =
            RokPcc::reduce(&params, &stmt, &wit, &mut transcript);

        // Sanity: also check the input PCC was valid.
        assert!(PccRelation::is_satisfied(&params, &stmt, &wit));

        let pco_params = PcoParams { srs: params.srs.clone() };
        for (pco_stmt, pco_wit) in out_stmts.iter().zip(&out_wits) {
            assert!(PcoRelation::is_satisfied(&pco_params, pco_stmt, pco_wit));
        }
    }

    /// Prover and verifier with identical labels must reconstruct the same
    /// list of PCO statements.
    #[test]
    fn prover_verifier_agree() {
        let (params, stmt, wit) = product_pcc();
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");
        let (proof, stmts_p, _) = RokPcc::reduce(&params, &stmt, &wit, &mut t_p);
        let stmts_v = RokPcc::verify(&params, &stmt, &proof, &mut t_v)
            .expect("verify must succeed for an honest proof");
        assert_eq!(stmts_p, stmts_v);
    }

    /// Bumping a `y` value will (with overwhelming probability) cause the
    /// constraint check `Q(x, values) = 0` to fail.
    #[test]
    fn verify_rejects_tampered_y() {
        let (params, stmt, wit) = product_pcc();
        let mut t_p = Blake3Transcript::new(b"test");
        let (mut proof, _, _) = RokPcc::reduce(&params, &stmt, &wit, &mut t_p);
        proof.values[0] += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(RokPcc::verify(&params, &stmt, &proof, &mut t_v).is_none());
    }

    /// Different transcript labels yield different `x` and therefore
    /// different shifted PCO statements.
    #[test]
    fn different_label_different_outputs() {
        let (params, stmt, wit) = product_pcc();
        let mut t_a = Blake3Transcript::new(b"label_a");
        let mut t_b = Blake3Transcript::new(b"label_b");
        let (_, stmts_a, _) = RokPcc::reduce(&params, &stmt, &wit, &mut t_a);
        let (_, stmts_b, _) = RokPcc::reduce(&params, &stmt, &wit, &mut t_b);
        assert_ne!(stmts_a[0].point, stmts_b[0].point);
        assert_ne!(stmts_a, stmts_b);
    }

    /// A PCC with no constraints — verify always returns `Some(stmts)`.
    #[test]
    fn empty_constraints_works() {
        let rng = &mut test_rng();
        let srs = pc::setup(10, rng);
        let p = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(7u64))]);
        let c = pc::commit(&srs, &Poly::Sparse(p.clone()));
        let stmt = PccStatement {
            commitments: vec![c],
            degrees: vec![1],
            constraints: vec![],
        };
        let wit = PccWitness { polynomials: vec![p] };
        let params = PccParams { srs };

        let mut t_p = Blake3Transcript::new(b"test");
        let (proof, stmts_p, _) = RokPcc::reduce(&params, &stmt, &wit, &mut t_p);
        let mut t_v = Blake3Transcript::new(b"test");
        let stmts_v = RokPcc::verify(&params, &stmt, &proof, &mut t_v)
            .expect("no constraints means no check can fail");
        assert_eq!(stmts_p, stmts_v);
    }
}
