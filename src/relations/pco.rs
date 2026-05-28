use ark_bls12_381::Fr;
use ark_poly::{univariate::SparsePolynomial, Polynomial};

use crate::core::relation::Relation;
use crate::pc::{self, Comm, Poly, Srs};

/// The R_PCO relation (compiler.tex, page 22 of the paper).
///
/// A statement asserts that `commitment` is a polynomial commitment to some
/// `p(X) ∈ F^{<D}[X]` (with `D = srs.powers_g1.len()`) such that `p(point) = value`.
///
/// The paper also names a variant `R_PCO^x` which fixes `x` ahead of time;
/// at the type level that's identical here — it's just a usage convention
/// over multiple `PcoStatement`s sharing the same `point`.
pub struct PcoRelation;

pub struct PcoParams {
    /// SRS for the polynomial commitment scheme. Bundles ck and vk and
    /// implicitly defines `D = srs.powers_g1.len()`, the strict degree bound.
    pub srs: Srs,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PcoStatement {
    pub commitment: Comm,
    /// Evaluation point `x`.
    pub point: Fr,
    /// Claimed evaluation `y = p(x)`.
    pub value: Fr,
}

#[derive(Clone)]
pub struct PcoWitness {
    pub polynomial: SparsePolynomial<Fr>,
}

impl Relation for PcoRelation {
    type Params = PcoParams;
    type Statement = PcoStatement;
    type Witness = PcoWitness;

    fn is_satisfied(p: &PcoParams, s: &PcoStatement, w: &PcoWitness) -> bool {
        // Paper-strict degree bound: `deg p < D`.
        if w.polynomial.degree() >= p.srs.powers_g1.len() {
            return false;
        }
        // Commit validity: re-commit and compare.
        if pc::commit(&p.srs, &Poly::Sparse(w.polynomial.clone())) != s.commitment {
            return false;
        }
        // Evaluation: p(x) == y.
        w.polynomial.evaluate(&s.point) == s.value
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    /// Build (params, statement, witness) for a polynomial `p` evaluated at `x`.
    fn make(poly: SparsePolynomial<Fr>, x: Fr) -> (PcoParams, PcoStatement, PcoWitness) {
        let rng = &mut test_rng();
        let srs = pc::setup(10, rng);
        let value = poly.evaluate(&x);
        let commitment = pc::commit(&srs, &Poly::Sparse(poly.clone()));
        let stmt = PcoStatement { commitment, point: x, value };
        let wit = PcoWitness { polynomial: poly };
        (PcoParams { srs }, stmt, wit)
    }

    /// Round-trip: a fresh PCO statement constructed from `p(x)` must verify.
    #[test]
    fn roundtrip() {
        // p(X) = 1 + 2X + 3X²
        let p = SparsePolynomial::from_coefficients_vec(vec![
            (0, Fr::from(1u64)),
            (1, Fr::from(2u64)),
            (2, Fr::from(3u64)),
        ]);
        let (params, stmt, wit) = make(p, Fr::from(5u64));
        // Sanity: 1 + 2·5 + 3·25 = 86.
        assert_eq!(stmt.value, Fr::from(86u64));
        assert!(PcoRelation::is_satisfied(&params, &stmt, &wit));
    }

    /// Random-x round-trip across a few iterations.
    #[test]
    fn roundtrip_random_points() {
        let p = SparsePolynomial::from_coefficients_vec(vec![
            (0, Fr::from(7u64)),
            (3, Fr::from(11u64)),
        ]);
        let rng = &mut test_rng();
        for _ in 0..5 {
            let x = Fr::rand(rng);
            let (params, stmt, wit) = make(p.clone(), x);
            assert!(PcoRelation::is_satisfied(&params, &stmt, &wit));
        }
    }

    /// Tampering `value` must reject.
    #[test]
    fn tampered_value_rejected() {
        let p = SparsePolynomial::from_coefficients_vec(vec![
            (0, Fr::from(1u64)),
            (1, Fr::from(2u64)),
        ]);
        let (params, mut stmt, wit) = make(p, Fr::from(3u64));
        stmt.value += Fr::from(1u64);
        assert!(!PcoRelation::is_satisfied(&params, &stmt, &wit));
    }

    /// Tampering `commitment` (replacing it with a commit to a different
    /// polynomial) must reject.
    #[test]
    fn tampered_commitment_rejected() {
        let p = SparsePolynomial::from_coefficients_vec(vec![(1, Fr::from(1u64))]);
        let (params, mut stmt, wit) = make(p, Fr::from(2u64));
        let bogus = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(99u64))]);
        stmt.commitment = pc::commit(&params.srs, &Poly::Sparse(bogus));
        assert!(!PcoRelation::is_satisfied(&params, &stmt, &wit));
    }

    /// Changing the point without recomputing `value` must reject (since
    /// `p` evaluated at the new point is generally not `value`).
    #[test]
    fn wrong_point_rejected() {
        // p(X) = X (so p(x) = x).
        let p = SparsePolynomial::from_coefficients_vec(vec![(1, Fr::from(1u64))]);
        let (params, mut stmt, wit) = make(p, Fr::from(7u64));
        stmt.point = Fr::from(8u64); // value is still 7
        assert!(!PcoRelation::is_satisfied(&params, &stmt, &wit));
    }

    /// Edge case: the zero polynomial evaluates to 0 at every point, and its
    /// commitment is the group identity.
    #[test]
    fn zero_polynomial_works() {
        let zero = SparsePolynomial::from_coefficients_vec(Vec::<(usize, Fr)>::new());
        let (params, stmt, wit) = make(zero, Fr::from(42u64));
        assert_eq!(stmt.value, Fr::from(0u64));
        assert!(PcoRelation::is_satisfied(&params, &stmt, &wit));
    }
}
