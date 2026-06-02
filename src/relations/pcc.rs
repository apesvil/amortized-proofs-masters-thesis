use ark_bls12_381::Fr;
use ark_ff::Zero;
use ark_poly::{univariate::SparsePolynomial, Polynomial};

use crate::core::relation::Relation;
use crate::pc::{self, Comm, Poly, Srs};

/// The R_PCC relation (Section 7 / page 22 of the paper). A statement consists
/// of `n` polynomial commitments `c_i` with claimed degree bounds `d_i`, and a
/// list of polynomial constraints `Q_j ∈ F[X, Y_1, ..., Y_n]` that must vanish
/// when each `Y_i` is replaced by the witnessed polynomial `p_i(X)`.
pub struct PccRelation;

pub struct PccParams {
    /// SRS for the polynomial commitment scheme. Bundles ck and vk.
    pub srs: Srs,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PccStatement {
    /// `c_i = pc.commit(ck, p_i(X), d_i)` for each i.
    pub commitments: Vec<Comm>,
    /// `d_i`: a (strict) upper bound on the degree of `p_i(X)`.
    pub degrees: Vec<usize>,
    /// `Q_j ∈ F[X, Y_1, ..., Y_n]`. Each must satisfy `Q_j(X, p(X)) = 0`.
    pub constraints: Vec<Constraint>,
}

#[derive(Clone)]
pub struct PccWitness {
    /// `p_i(X)` — the polynomials behind the commitments.
    pub polynomials: Vec<SparsePolynomial<Fr>>,
}

/// A constraint `Q ∈ F[X, Y_1, ..., Y_n]` as a sum of monomials
/// `coeff · X^{x_deg} · ∏_{(i, e) ∈ y_terms} Y_i^e`.
#[derive(Clone, Debug, PartialEq)]
pub struct Constraint {
    pub monomials: Vec<Monomial>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Monomial {
    pub coeff: Fr,
    pub x_deg: usize,
    /// `(Y-variable index, exponent)` pairs.
    pub y_terms: Vec<(usize, usize)>,
}

impl Relation for PccRelation {
    type Params = PccParams;
    type Statement = PccStatement;
    type Witness = PccWitness;

    fn is_satisfied(p: &PccParams, s: &PccStatement, w: &PccWitness) -> bool {
        // Length sanity: commitments, degrees, witness polys must agree.
        if s.commitments.len() != s.degrees.len()
            || s.commitments.len() != w.polynomials.len()
        {
            return false;
        }

        // Per-polynomial: paper-strict degree bound (deg p_i < d_i) and commit validity.
        for ((poly, &d_i), c_i) in
            w.polynomials.iter().zip(&s.degrees).zip(&s.commitments)
        {
            if poly.degree() >= d_i {
                return false;
            }
            if pc::commit(&p.srs, &Poly::Sparse((*poly).clone())) != *c_i {
                return false;
            }
        }

        // Each Q_j(X, p(X)) must be the zero polynomial in X.
        for q in &s.constraints {
            if !substitute(q, &w.polynomials).is_zero() {
                return false;
            }
        }
        true
    }
}

/// Symbolic substitution `Y_i := p_i(X)` into `Q`, returning the resulting
/// univariate polynomial in `X`.
fn substitute(q: &Constraint, polys: &[SparsePolynomial<Fr>]) -> SparsePolynomial<Fr> {
    let mut total = SparsePolynomial::<Fr>::zero();
    for mono in &q.monomials {
        // Start with `coeff · X^{x_deg}`.
        let mut term =
            SparsePolynomial::from_coefficients_vec(vec![(mono.x_deg, mono.coeff)]);
        // Multiply in `p_idx(X)^exp` for each y-term.
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
    use ark_std::test_rng;

    fn srs() -> Srs {
        pc::setup(20, &mut test_rng())
    }

    fn commit_polys(srs: &Srs, polys: &[SparsePolynomial<Fr>]) -> Vec<Comm> {
        polys
            .iter()
            .map(|p| pc::commit(srs, &Poly::Sparse((*p).clone())))
            .collect()
    }

    /// Y_0 · Y_1 − Y_2 = 0  with  p_0 = X+1, p_1 = X²−1, p_2 = (X+1)(X²−1).
    #[test]
    fn product_constraint_satisfied() {
        let srs = srs();
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
        let commitments = commit_polys(&srs, &polys);
        let degrees = vec![2, 3, 4]; // d_i strictly greater than deg p_i
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
        let stmt = PccStatement { commitments, degrees, constraints: vec![q] };
        let wit = PccWitness { polynomials: polys };
        assert!(PccRelation::is_satisfied(&PccParams { srs }, &stmt, &wit));
    }

    /// 2·Y_0 − Y_1 = 0  with  p_0 = X, p_1 = 2X.
    #[test]
    fn linear_constraint_satisfied() {
        let srs = srs();
        let p0 = SparsePolynomial::from_coefficients_vec(vec![(1, Fr::from(1u64))]);
        let p1 = SparsePolynomial::from_coefficients_vec(vec![(1, Fr::from(2u64))]);
        let polys = vec![p0, p1];
        let commitments = commit_polys(&srs, &polys);
        let degrees = vec![2, 2];
        let q = Constraint {
            monomials: vec![
                Monomial {
                    coeff: Fr::from(2u64),
                    x_deg: 0,
                    y_terms: vec![(0, 1)],
                },
                Monomial {
                    coeff: -Fr::from(1u64),
                    x_deg: 0,
                    y_terms: vec![(1, 1)],
                },
            ],
        };
        let stmt = PccStatement { commitments, degrees, constraints: vec![q] };
        let wit = PccWitness { polynomials: polys };
        assert!(PccRelation::is_satisfied(&PccParams { srs }, &stmt, &wit));
    }

    /// X^3 − X · Y_0 = 0  with  p_0 = X^2  (exercises the `x_deg` field).
    #[test]
    fn x_power_constraint_satisfied() {
        let srs = srs();
        let p0 = SparsePolynomial::from_coefficients_vec(vec![(2, Fr::from(1u64))]);
        let polys = vec![p0];
        let commitments = commit_polys(&srs, &polys);
        let degrees = vec![3];
        let q = Constraint {
            monomials: vec![
                Monomial {
                    coeff: Fr::from(1u64),
                    x_deg: 3,
                    y_terms: vec![],
                },
                Monomial {
                    coeff: -Fr::from(1u64),
                    x_deg: 1,
                    y_terms: vec![(0, 1)],
                },
            ],
        };
        let stmt = PccStatement { commitments, degrees, constraints: vec![q] };
        let wit = PccWitness { polynomials: polys };
        assert!(PccRelation::is_satisfied(&PccParams { srs }, &stmt, &wit));
    }

    /// Y_0 − Y_1 = 0  with  p_0 = X+1, p_1 = X — must reject.
    #[test]
    fn violated_constraint_rejected() {
        let srs = srs();
        let p0 = SparsePolynomial::from_coefficients_vec(vec![
            (0, Fr::from(1u64)),
            (1, Fr::from(1u64)),
        ]);
        let p1 = SparsePolynomial::from_coefficients_vec(vec![(1, Fr::from(1u64))]);
        let polys = vec![p0, p1];
        let commitments = commit_polys(&srs, &polys);
        let degrees = vec![2, 2];
        let q = Constraint {
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
        };
        let stmt = PccStatement { commitments, degrees, constraints: vec![q] };
        let wit = PccWitness { polynomials: polys };
        assert!(!PccRelation::is_satisfied(&PccParams { srs }, &stmt, &wit));
    }

    /// Replacing a commitment with one for a different polynomial must reject.
    #[test]
    fn tampered_commitment_rejected() {
        let srs = srs();
        let p0 = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(1u64))]);
        let polys = vec![p0];
        let mut commitments = commit_polys(&srs, &polys);
        let bogus =
            SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(99u64))]);
        commitments[0] = pc::commit(&srs, &Poly::Sparse(bogus));
        let stmt = PccStatement {
            commitments,
            degrees: vec![1],
            constraints: vec![],
        };
        let wit = PccWitness { polynomials: polys };
        assert!(!PccRelation::is_satisfied(&PccParams { srs }, &stmt, &wit));
    }

    /// Claiming `d_0 = 2` for a degree-2 polynomial must reject (paper: deg < d_i).
    #[test]
    fn degree_bound_violated_rejected() {
        let srs = srs();
        let p0 = SparsePolynomial::from_coefficients_vec(vec![(2, Fr::from(1u64))]);
        let polys = vec![p0];
        let commitments = commit_polys(&srs, &polys);
        let stmt = PccStatement {
            commitments,
            degrees: vec![2],
            constraints: vec![],
        };
        let wit = PccWitness { polynomials: polys };
        assert!(!PccRelation::is_satisfied(&PccParams { srs }, &stmt, &wit));
    }
}
