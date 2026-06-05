use ark_bls12_381::Fr;
use ark_poly::univariate::SparsePolynomial;

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
