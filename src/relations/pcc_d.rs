use ark_bls12_381::Fr;
use ark_ff::Zero;
use ark_poly::{univariate::SparsePolynomial, Polynomial};

use crate::core::relation::Relation;
use crate::pc::{self, Comm, Poly, Srs};
use crate::relations::pcc::Constraint;

/// The `R_PCC-D` relation (updated `compiler.tex`). Same as `R_PCC` but the
/// per-polynomial degree bound `d_i` is dropped — every polynomial is only
/// required to have degree strictly less than the global `D` set by the SRS.
///
/// Callers typically arrive here via `Π_DT`, which shifts each original
/// polynomial by `X^{D − d_i}` so that the *paper-strict* per-polynomial
/// bound `deg p_i < d_i` becomes enforced **implicitly** by the SRS-level
/// `< D` check on the shifted polynomial.
pub struct PccDRelation;

pub struct PccDParams {
    pub srs: Srs,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PccDStatement {
    /// Length 2n in typical use: `[C_0, ..., C_{n-1}, C'_0, ..., C'_{n-1}]`
    /// where `C'_i` is a commitment to `p'_i(X) = X^{D-d_i} · p_i(X)`.
    pub commitments: Vec<Comm>,
    /// Length ℓ + n in typical use: original `Q_j` plus the shift-relation
    /// constraints `Q'_i(X, Y) = X^{D-d_i} · Y_i − Y_{n+i}`.
    pub constraints: Vec<Constraint>,
}

#[derive(Clone)]
pub struct PccDWitness {
    /// Length 2n in typical use: `[p_0, ..., p_{n-1}, p'_0, ..., p'_{n-1}]`.
    pub polynomials: Vec<SparsePolynomial<Fr>>,
}

impl Relation for PccDRelation {
    type Params = PccDParams;
    type Statement = PccDStatement;
    type Witness = PccDWitness;

    fn is_satisfied(p: &PccDParams, s: &PccDStatement, w: &PccDWitness) -> bool {
        if s.commitments.len() != w.polynomials.len() {
            return false;
        }

        let big_d = p.srs.powers_g1.len();
        for (poly, c) in w.polynomials.iter().zip(&s.commitments) {
            // Global (paper-strict) degree bound.
            if poly.degree() >= big_d {
                return false;
            }
            // Commit validity.
            if pc::commit(&p.srs, &Poly::Sparse(poly.clone())) != *c {
                return false;
            }
        }

        // Constraints.
        for q in &s.constraints {
            if !substitute(q, &w.polynomials).is_zero() {
                return false;
            }
        }
        true
    }
}

/// Symbolic substitution `Y_i := p_i(X)` into `Q`.
/// (Identical to the helper in `pcc.rs`; kept local to keep the two relations
/// each self-contained.)
fn substitute(q: &Constraint, polys: &[SparsePolynomial<Fr>]) -> SparsePolynomial<Fr> {
    let mut total = SparsePolynomial::<Fr>::zero();
    for mono in &q.monomials {
        let mut term =
            SparsePolynomial::from_coefficients_vec(vec![(mono.x_deg, mono.coeff)]);
        for &(idx, exp) in &mono.y_terms {
            for _ in 0..exp {
                term = term.mul(&polys[idx]);
            }
        }
        total = &total + &term;
    }
    total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relations::pcc::Monomial;
    use ark_std::test_rng;

    fn srs() -> Srs {
        pc::setup(20, &mut test_rng())
    }

    fn commit(srs: &Srs, p: &SparsePolynomial<Fr>) -> Comm {
        pc::commit(srs, &Poly::Sparse(p.clone()))
    }

    /// Two polys with the relation `p_1 = X · p_0`. Constraint: `X·Y_0 − Y_1 = 0`.
    #[test]
    fn shift_relation_satisfied() {
        let srs = srs();
        let p0 = SparsePolynomial::from_coefficients_vec(vec![
            (0, Fr::from(3u64)),
            (1, Fr::from(5u64)),
        ]);
        // p_1 = X · p_0 = 3X + 5X²
        let p1 = SparsePolynomial::from_coefficients_vec(vec![
            (1, Fr::from(3u64)),
            (2, Fr::from(5u64)),
        ]);
        let c0 = commit(&srs, &p0);
        let c1 = commit(&srs, &p1);
        let q = Constraint {
            monomials: vec![
                Monomial { coeff:  Fr::from(1u64), x_deg: 1, y_terms: vec![(0, 1)] },
                Monomial { coeff: -Fr::from(1u64), x_deg: 0, y_terms: vec![(1, 1)] },
            ],
        };
        let stmt = PccDStatement { commitments: vec![c0, c1], constraints: vec![q] };
        let wit = PccDWitness { polynomials: vec![p0, p1] };
        assert!(PccDRelation::is_satisfied(&PccDParams { srs }, &stmt, &wit));
    }

    /// Tampered commitment must reject.
    #[test]
    fn tampered_commitment_rejected() {
        let srs = srs();
        let p = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(1u64))]);
        let mut stmt = PccDStatement {
            commitments: vec![commit(&srs, &p)],
            constraints: vec![],
        };
        let bogus = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(99u64))]);
        stmt.commitments[0] = commit(&srs, &bogus);
        let wit = PccDWitness { polynomials: vec![p] };
        assert!(!PccDRelation::is_satisfied(&PccDParams { srs }, &stmt, &wit));
    }

    /// Polynomial exceeding the SRS bound `D` must reject. Hard to test
    /// directly since `pc::commit` panics; instead we construct a statement
    /// claiming D-degree polynomials but provide a witness whose recomputed
    /// commit doesn't match (the second poly has high degree).
    /// Here we just check the length-mismatch guard.
    #[test]
    fn length_mismatch_rejected() {
        let srs = srs();
        let stmt = PccDStatement { commitments: vec![], constraints: vec![] };
        let wit = PccDWitness {
            polynomials: vec![SparsePolynomial::from_coefficients_vec(vec![(
                0,
                Fr::from(1u64),
            )])],
        };
        assert!(!PccDRelation::is_satisfied(&PccDParams { srs }, &stmt, &wit));
    }
}
