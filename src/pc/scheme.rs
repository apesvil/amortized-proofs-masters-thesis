use ark_bls12_381::{Bls12_381, Fr, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{One, PrimeField, Zero};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};

use super::msm::{msm_dense, msm_sparse};
use super::poly::Poly;
use super::srs::Srs;

/// A KZG commitment in projective form. `G1Projective` already implements
/// `Add`, `Sub`, `Mul<Fr>`, so commitments compose homomorphically.
pub type Comm = G1Projective;

/// An evaluation proof, also in projective form for the same reason.
pub type Opening = G1Projective;

/// Commits to `poly`. For `Poly::Sparse`, runs an MSM whose size is the number
/// of nonzero terms; for `Poly::Dense`, runs an MSM over the whole coefficient slice.
pub fn commit(srs: &Srs, poly: &Poly) -> Comm {
    assert!(
        poly.degree() < srs.powers_g1.len(),
        "polynomial degree exceeds SRS max_degree"
    );
    match poly {
        // DensePolynomial.coeffs is a public Vec<Fr>.
        Poly::Dense(p) => msm_dense(&srs.powers_g1, &p.coeffs),
        // SparsePolynomial: Deref<Target = [(usize, Fr)]>; &**p materializes the slice.
        Poly::Sparse(p) => msm_sparse(&srs.powers_g1, &**p),
    }
}

/// Proves that the committed polynomial evaluates to `f(point)`.
/// Returns `(π, f(point))` where `π = q(τ)·g` and `q(x) = (f(x) - f(z)) / (x - z)`.
///
/// The quotient is generally dense, so the proof MSM is unavoidably O(d).
pub fn prove(srs: &Srs, poly: &Poly, point: Fr) -> (Opening, Fr) {
    let value = poly.evaluate(&point);
    let dense = poly.to_dense();

    // Build f(x) - f(z), then divide by (x - z).
    let mut num_coeffs = dense.coeffs.clone();
    if num_coeffs.is_empty() {
        num_coeffs.push(Fr::zero());
    }
    num_coeffs[0] -= value;
    let numerator = DensePolynomial::from_coefficients_vec(num_coeffs);
    let divisor = DensePolynomial::from_coefficients_vec(vec![-point, Fr::one()]);
    let quotient = &numerator / &divisor;

    let proof = msm_dense(&srs.powers_g1, &quotient.coeffs);
    (proof, value)
}

/// Verifies that `comm` commits to a polynomial with `f(point) = value`.
/// Checks the pairing equation: e(C − v·g, h) == e(π, τ·h − z·h).
pub fn verify(srs: &Srs, comm: &Comm, point: Fr, value: Fr, proof: &Opening) -> bool {
    let g = srs.powers_g1[0];
    // Convert to affine only at the pairing boundary.
    let lhs_g1 = (*comm - g.mul_bigint(value.into_bigint())).into_affine();
    let rhs_g2 = (G2Projective::from(srs.tau_g2)
        - srs.g2.mul_bigint(point.into_bigint()))
    .into_affine();

    Bls12_381::pairing(lhs_g1, srs.g2) == Bls12_381::pairing(proof.into_affine(), rhs_g2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::pc::*;
    use ark_bls12_381::Fr;
    use ark_std::test_rng;

    fn srs() -> Srs {
        setup(20, &mut test_rng())
    }

    // Round-trip with sparse input.
    #[test]
    fn test_roundtrip_sparse() {
        let srs = srs();
        // f(x) = 1 + 3x² + 5x⁵
        let poly = Poly::sparse(vec![
            (0, Fr::from(1u64)),
            (2, Fr::from(3u64)),
            (5, Fr::from(5u64)),
        ]);
        let comm = commit(&srs, &poly);
        let point = Fr::from(7u64);
        let (proof, value) = prove(&srs, &poly, point);
        assert!(verify(&srs, &comm, point, value, &proof));
    }

    // Round-trip with dense input.
    #[test]
    fn test_roundtrip_dense() {
        let srs = srs();
        let poly = Poly::dense([1u64, 0, 3, 0, 0, 5].into_iter().map(Fr::from).collect());
        let comm = commit(&srs, &poly);
        let point = Fr::from(7u64);
        let (proof, value) = prove(&srs, &poly, point);
        assert!(verify(&srs, &comm, point, value, &proof));
    }

    // Sparse and dense representations of the same polynomial give the same commitment.
    #[test]
    fn test_sparse_dense_equivalent() {
        let srs = srs();
        let sparse = Poly::sparse(vec![
            (0, Fr::from(1u64)),
            (2, Fr::from(3u64)),
            (5, Fr::from(5u64)),
        ]);
        let dense = Poly::dense([1u64, 0, 3, 0, 0, 5].into_iter().map(Fr::from).collect());
        assert_eq!(commit(&srs, &sparse), commit(&srs, &dense));
    }

    // Evaluation value matches manual computation: f(x) = 2 + 3x, f(4) = 14.
    #[test]
    fn test_evaluation_correctness() {
        let srs = srs();
        let poly = Poly::sparse(vec![(0, Fr::from(2u64)), (1, Fr::from(3u64))]);
        let (_, value) = prove(&srs, &poly, Fr::from(4u64));
        assert_eq!(value, Fr::from(14u64));
    }

    // A tampered claimed value must not pass verification.
    #[test]
    fn test_wrong_value_rejected() {
        let srs = srs();
        let poly = Poly::sparse(vec![(0, Fr::from(1u64)), (1, Fr::from(1u64))]);
        let comm = commit(&srs, &poly);
        let point = Fr::from(5u64);
        let (proof, value) = prove(&srs, &poly, point);
        assert!(!verify(&srs, &comm, point, value + Fr::from(1u64), &proof));
    }

    // A constant polynomial (degree 0) works end-to-end.
    #[test]
    fn test_constant_polynomial() {
        let srs = setup(1, &mut test_rng());
        let poly = Poly::sparse(vec![(0, Fr::from(42u64))]);
        let comm = commit(&srs, &poly);
        let point = Fr::from(999u64);
        let (proof, value) = prove(&srs, &poly, point);
        assert_eq!(value, Fr::from(42u64));
        assert!(verify(&srs, &comm, point, value, &proof));
    }

    // The same commitment verifies at multiple evaluation points.
    #[test]
    fn test_multiple_points_same_commitment() {
        let srs = srs();
        let poly = Poly::sparse(vec![(2, Fr::from(1u64))]); // f(x) = x²
        let comm = commit(&srs, &poly);
        for i in 0u64..6 {
            let point = Fr::from(i);
            let (proof, value) = prove(&srs, &poly, point);
            assert_eq!(value, Fr::from(i * i), "f({i}) should be {}", i * i);
            assert!(verify(&srs, &comm, point, value, &proof));
        }
    }

    // A proof for polynomial B must not verify against a commitment to polynomial A.
    #[test]
    fn test_wrong_commitment_rejected() {
        let srs = srs();
        let poly_a = Poly::sparse(vec![(0, Fr::from(1u64)), (1, Fr::from(2u64))]);
        let poly_b = Poly::sparse(vec![(0, Fr::from(9u64)), (1, Fr::from(8u64))]);

        let comm_a = commit(&srs, &poly_a);
        let point = Fr::from(3u64);
        let (proof_b, value_b) = prove(&srs, &poly_b, point);

        assert!(!verify(&srs, &comm_a, point, value_b, &proof_b));
    }

    // Duplicate degrees no longer panic — they sum during the MSM.
    // f(x) = (3 + 4)x² = 7x² should match a clean (2, 7) input.
    #[test]
    fn test_sparse_duplicate_degrees_sum() {
        let srs = srs();
        let dup = Poly::sparse(vec![(2, Fr::from(3u64)), (2, Fr::from(4u64))]);
        let merged = Poly::sparse(vec![(2, Fr::from(7u64))]);
        assert_eq!(commit(&srs, &dup), commit(&srs, &merged));
    }

    // Commitment homomorphism: commit(α·f + β·g) == α·commit(f) + β·commit(g).
    #[test]
    fn test_commitment_homomorphism() {
        let srs = srs();
        let f = Poly::sparse(vec![(0, Fr::from(1u64)), (3, Fr::from(2u64))]);
        let g = Poly::dense([5u64, 0, 7].into_iter().map(Fr::from).collect());
        let alpha = Fr::from(11u64);
        let beta = Fr::from(13u64);

        // Direct: commit(α·f + β·g) computed as a single polynomial.
        let f_dense = f.to_dense();
        let g_dense = g.to_dense();
        let scaled_f = &f_dense * alpha;
        let scaled_g = &g_dense * beta;
        let combined = &scaled_f + &scaled_g;
        let c_direct = commit(&srs, &Poly::Dense(combined));

        // Via homomorphism: combine the commitments as group elements.
        let c_f = commit(&srs, &f);
        let c_g = commit(&srs, &g);
        let c_homom = c_f * alpha + c_g * beta;

        assert_eq!(c_direct, c_homom);
    }
}
