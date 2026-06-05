use ark_bls12_381::Fr;
use ark_poly::univariate::SparsePolynomial;

use crate::pc::{self, Comm, Poly};
use crate::reductions::poly_util::x_shift;
use crate::relations::pcc::{Constraint, Monomial, PccParams, PccStatement, PccWitness};
use crate::relations::pcc_d::{PccDStatement, PccDWitness};
use crate::transcript::Blake3Transcript;

/// The `Π_DT : R_PCC → R_PCC-D` "degree test" reduction
/// (`figure_dtrok.tex` in the updated paper).
///
/// The prover commits to the X-shifted polynomials `p'_i(X) = X^{D−d_i} · p_i(X)`
/// and ships those commitments. The output `R_PCC-D` statement bundles
/// both halves (commitments and constraints) and adds `n` extra
/// constraints `Q'_i(X, Y) = X^{D−d_i} · Y_i − Y_{n+i}` tying each shifted
/// polynomial to its original.
///
/// Soundness of the per-polynomial degree bound now follows from the SRS
/// itself: the prover can only commit `p'_i` if `deg p'_i < D`, which
/// forces `deg p_i < d_i`.
pub struct RokDt;

/// The prover's message: commitments to the `n` shifted polynomials.
pub struct RokDtProof {
    pub shifted_commitments: Vec<Comm>,
}

impl RokDt {
    /// Prover side.
    pub fn reduce(
        params: &PccParams,
        stmt: &PccStatement,
        wit: &PccWitness,
        transcript: &mut Blake3Transcript,
    ) -> (RokDtProof, PccDStatement, PccDWitness) {
        let n = stmt.commitments.len();
        assert_eq!(stmt.degrees.len(), n);
        assert_eq!(wit.polynomials.len(), n);

        let big_d = params.srs.powers_g1.len();

        // 1. Build the shifted polynomials and commit each.
        let shifted_polys: Vec<SparsePolynomial<Fr>> = wit
            .polynomials
            .iter()
            .zip(&stmt.degrees)
            .map(|(p, &d_i)| {
                assert!(d_i <= big_d, "d_i must satisfy d_i ≤ D");
                x_shift(p, big_d - d_i)
            })
            .collect();
        let shifted_commitments: Vec<Comm> = shifted_polys
            .iter()
            .map(|p| pc::commit(&params.srs, &Poly::Sparse(p.clone())))
            .collect();

        // 2. Absorb the input statement and the new commitments into the
        //    transcript so downstream challenges depend on them.
        absorb_pcc_statement(transcript, stmt);
        for c in &shifted_commitments {
            transcript.absorb(b"rok_dt::shifted_c", c);
        }

        // 3. Assemble the combined PCC-D statement and witness.
        let combined_commitments: Vec<Comm> = stmt
            .commitments
            .iter()
            .copied()
            .chain(shifted_commitments.iter().copied())
            .collect();

        let mut combined_constraints = stmt.constraints.clone();
        for (i, &d_i) in stmt.degrees.iter().enumerate() {
            combined_constraints.push(shift_constraint(big_d - d_i, i, n + i));
        }
        let out_stmt = PccDStatement {
            commitments: combined_commitments,
            constraints: combined_constraints,
        };

        let combined_polys: Vec<SparsePolynomial<Fr>> = wit
            .polynomials
            .iter()
            .cloned()
            .chain(shifted_polys.into_iter())
            .collect();
        let out_wit = PccDWitness { polynomials: combined_polys };

        let proof = RokDtProof { shifted_commitments };
        (proof, out_stmt, out_wit)
    }

    /// Verifier side. Reconstructs the `R_PCC-D` statement deterministically.
    pub fn verify(
        params: &PccParams,
        stmt: &PccStatement,
        proof: &RokDtProof,
        transcript: &mut Blake3Transcript,
    ) -> PccDStatement {
        let n = stmt.commitments.len();
        assert_eq!(stmt.degrees.len(), n);
        assert_eq!(proof.shifted_commitments.len(), n);

        let big_d = params.srs.powers_g1.len();

        // Mirror the prover's absorbs.
        absorb_pcc_statement(transcript, stmt);
        for c in &proof.shifted_commitments {
            transcript.absorb(b"rok_dt::shifted_c", c);
        }

        let combined_commitments: Vec<Comm> = stmt
            .commitments
            .iter()
            .copied()
            .chain(proof.shifted_commitments.iter().copied())
            .collect();

        let mut combined_constraints = stmt.constraints.clone();
        for (i, &d_i) in stmt.degrees.iter().enumerate() {
            assert!(d_i <= big_d, "d_i must satisfy d_i ≤ D");
            combined_constraints.push(shift_constraint(big_d - d_i, i, n + i));
        }

        PccDStatement {
            commitments: combined_commitments,
            constraints: combined_constraints,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the constraint `X^k · Y_i − Y_j = 0`.
fn shift_constraint(k: usize, i: usize, j: usize) -> Constraint {
    Constraint {
        monomials: vec![
            Monomial { coeff:  Fr::from(1u64), x_deg: k, y_terms: vec![(i, 1)] },
            Monomial { coeff: -Fr::from(1u64), x_deg: 0, y_terms: vec![(j, 1)] },
        ],
    }
}

/// Absorb a `PccStatement` into the transcript (same shape as `rok_pcc`).
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Relation;
    use crate::pc::{self, Poly};
    use crate::relations::pcc::{PccParams, PccStatement, PccWitness};
    use crate::relations::pcc_d::{PccDParams, PccDRelation};
    use ark_std::test_rng;

    /// Build a valid `R_PCC` instance: `p_0 = 1 + X`, `p_1 = X² − 1`,
    /// `p_2 = p_0 · p_1`, with constraint `Y_0·Y_1 − Y_2 = 0` and degrees
    /// `(2, 3, 4)` (paper-strict, i.e. `deg p_i < d_i`).
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
        let commitments = vec![
            pc::commit(&srs, &Poly::Sparse(p0.clone())),
            pc::commit(&srs, &Poly::Sparse(p1.clone())),
            pc::commit(&srs, &Poly::Sparse(p2.clone())),
        ];
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
        let wit = PccWitness { polynomials: vec![p0, p1, p2] };
        (PccParams { srs }, stmt, wit)
    }

    /// `Π_DT` on an honest PCC instance yields a PCC-D instance satisfying its relation.
    #[test]
    fn dt_roundtrip() {
        let (params, stmt, wit) = product_pcc();
        let mut t = Blake3Transcript::new(b"test");
        let (_proof, d_stmt, d_wit) = RokDt::reduce(&params, &stmt, &wit, &mut t);
        let d_params = PccDParams { srs: params.srs.clone() };
        assert!(PccDRelation::is_satisfied(&d_params, &d_stmt, &d_wit));
    }

    /// Prover and verifier reconstruct the same `R_PCC-D` statement.
    #[test]
    fn prover_verifier_agree() {
        let (params, stmt, wit) = product_pcc();
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");
        let (proof, d_stmt_p, _) = RokDt::reduce(&params, &stmt, &wit, &mut t_p);
        let d_stmt_v = RokDt::verify(&params, &stmt, &proof, &mut t_v);
        assert_eq!(d_stmt_p, d_stmt_v);
    }

    /// The combined statement carries `2n` commitments and `ℓ + n` constraints.
    #[test]
    fn shape_check() {
        let (params, stmt, wit) = product_pcc();
        let mut t = Blake3Transcript::new(b"test");
        let (_, d_stmt, d_wit) = RokDt::reduce(&params, &stmt, &wit, &mut t);
        assert_eq!(d_stmt.commitments.len(), 6); // 2 * 3
        assert_eq!(d_stmt.constraints.len(), 4); // 1 + 3
        assert_eq!(d_wit.polynomials.len(), 6);
    }

    /// A witness with `deg p_i > d_i − 1` cannot be committed under our SRS
    /// because the shifted polynomial overflows `D`. We model "the prover
    /// would have to break the SRS" by checking that committing such a
    /// shifted polynomial directly is rejected.
    #[test]
    #[should_panic(expected = "polynomial degree")]
    fn oversized_polynomial_panics_at_commit() {
        let (params, mut stmt, mut wit) = product_pcc();
        // Replace p_0 (deg 1) with a polynomial of degree 2 while keeping
        // d_0 = 2 (which says deg p_0 < 2). The Π_DT prover tries to commit
        // p'_0 = X^{D-2} · p_0, which has degree D, and `pc::commit` panics.
        let bad_p0 = SparsePolynomial::from_coefficients_vec(vec![
            (0, Fr::from(1u64)),
            (2, Fr::from(1u64)),
        ]);
        wit.polynomials[0] = bad_p0.clone();
        stmt.commitments[0] = pc::commit(&params.srs, &Poly::Sparse(bad_p0));
        let mut t = Blake3Transcript::new(b"test");
        let _ = RokDt::reduce(&params, &stmt, &wit, &mut t);
    }
}
