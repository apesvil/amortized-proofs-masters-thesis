use ark_bls12_381::Fr;
use ark_ff::Zero;
use ark_poly::{
    univariate::{DensePolynomial, SparsePolynomial},
    DenseUVPolynomial,
};

/// Multiply a sparse polynomial by `X^k` (i.e. shift every exponent by `k`).
///
/// Used by the X-shift degree-binding trick in `rok_dt` (Π_DT) and `rok_p`
/// (Π_P), so it lives at module scope rather than being duplicated in each
/// reduction.
pub(super) fn x_shift(p: &SparsePolynomial<Fr>, k: usize) -> SparsePolynomial<Fr> {
    SparsePolynomial::from_coefficients_vec(
        p.iter().map(|&(deg, c)| (deg + k, c)).collect(),
    )
}

/// X-shift on a dense polynomial: prepend `k` zero coefficients.
/// Used by `rok_p`'s w/q shifts after the dense-poly refactor.
pub(super) fn x_shift_dense(p: &DensePolynomial<Fr>, k: usize) -> DensePolynomial<Fr> {
    let mut coeffs = vec![Fr::zero(); k];
    coeffs.extend_from_slice(&p.coeffs);
    DensePolynomial::from_coefficients_vec(coeffs)
}
