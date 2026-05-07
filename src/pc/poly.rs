use ark_bls12_381::Fr;
use ark_poly::{
    univariate::{DensePolynomial, SparsePolynomial},
    DenseUVPolynomial, Polynomial,
};

/// Polynomial input. Owns arkworks types so `+`, scalar `*`, evaluation, and
/// degree work via the standard arkworks impls.
pub enum Poly {
    Dense(DensePolynomial<Fr>),
    Sparse(SparsePolynomial<Fr>),
}

impl Poly {
    /// Build from `(degree, coefficient)` pairs. Zeros are stripped and terms
    /// are sorted by degree; duplicate degrees are tolerated and their
    /// contributions sum during the commit MSM.
    pub fn sparse(terms: Vec<(usize, Fr)>) -> Self {
        Poly::Sparse(SparsePolynomial::from_coefficients_vec(terms))
    }

    /// Build from dense coefficients (`coeffs[i]` = coefficient of `xⁱ`).
    pub fn dense(coeffs: Vec<Fr>) -> Self {
        Poly::Dense(DensePolynomial::from_coefficients_vec(coeffs))
    }

    pub fn degree(&self) -> usize {
        match self {
            Poly::Dense(p) => p.degree(),
            Poly::Sparse(p) => p.degree(),
        }
    }

    pub fn evaluate(&self, point: &Fr) -> Fr {
        match self {
            Poly::Dense(p) => p.evaluate(point),
            Poly::Sparse(p) => p.evaluate(point),
        }
    }

    /// Materialize a dense coefficient form (needed for the proof's quotient).
    pub fn to_dense(&self) -> DensePolynomial<Fr> {
        match self {
            // (*p).clone() forces the Clone impl on DensePolynomial<Fr>;
            // p.clone() would resolve to <&T as Clone>::clone in some tooling.
            Poly::Dense(p) => (*p).clone(),
            Poly::Sparse(p) => DensePolynomial::from((*p).clone()),
        }
    }
}
