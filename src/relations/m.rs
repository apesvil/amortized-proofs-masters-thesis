use ark_bls12_381::Fr;
use ark_poly::{
    univariate::SparsePolynomial,
    EvaluationDomain, Polynomial, Radix2EvaluationDomain,
};

use crate::core::relation::Relation;
use crate::pc::{self, Comm, Poly, Srs};

/// The R_M relation (Section 8 of the paper). Captures
/// `y = uᵀ M v` for a sparse matrix M, with u and v committed as rational
/// polynomial pairs (N_u, T_u) and (N_v, T_v) over a multiplicative subgroup H.
pub struct MRelation;

pub struct MParams {
    /// SRS for the polynomial commitment scheme. Bundles ck and vk.
    pub srs: Srs,
    /// The matrix M as sparse triples (row, col, value).
    pub matrix: Vec<(usize, usize, Fr)>,
    /// |H|, must be a power of two. Defines the Lagrange basis.
    pub n: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MStatement {
    pub c_n_u: Comm,
    pub c_t_u: Comm,
    pub c_n_v: Comm,
    pub c_t_v: Comm,
    pub d_n_u: usize,
    pub d_t_u: usize,
    pub d_n_v: usize,
    pub d_t_v: usize,
    /// y
    pub value: Fr,
}

pub struct MWitness {
    pub n_u: SparsePolynomial<Fr>,
    pub t_u: SparsePolynomial<Fr>,
    pub u: Vec<Fr>,
    pub n_v: SparsePolynomial<Fr>,
    pub t_v: SparsePolynomial<Fr>,
    pub v: Vec<Fr>,
}

impl Relation for MRelation {
    type Params = MParams;
    type Statement = MStatement;
    type Witness = MWitness;

    fn is_satisfied(p: &MParams, s: &MStatement, w: &MWitness) -> bool {
        // Degree bounds.
        if w.n_u.degree() > s.d_n_u {
            return false;
        }
        if w.t_u.degree() > s.d_t_u {
            return false;
        }
        if w.n_v.degree() > s.d_n_v {
            return false;
        }
        if w.t_v.degree() > s.d_t_v {
            return false;
        }

        // Commitment validity, paper-strict: each statement commitment must
        // equal a fresh commit of the witness polynomial.
        if pc::commit(&p.srs, &Poly::Sparse(w.n_u.clone())) != s.c_n_u {
            return false;
        }
        if pc::commit(&p.srs, &Poly::Sparse(w.t_u.clone())) != s.c_t_u {
            return false;
        }
        if pc::commit(&p.srs, &Poly::Sparse(w.n_v.clone())) != s.c_n_v {
            return false;
        }
        if pc::commit(&p.srs, &Poly::Sparse(w.t_v.clone())) != s.c_t_v {
            return false;
        }

        // Encoding identity in paper form, multiplied to dodge division
        // and the T(h_i) = 0 edge case:
        //   paper: N(X) / (n · T(X)) = wᵀ λ(X)
        //   here:  N(h_i) = n · T(h_i) · w[i]   for every h_i ∈ H
        let dom =
            Radix2EvaluationDomain::<Fr>::new(p.n).expect("n must be a power of two");
        let n_field = Fr::from(p.n as u64);
        for (i, h_i) in dom.elements().enumerate() {
            if w.n_u.evaluate(&h_i) != n_field * w.t_u.evaluate(&h_i) * w.u[i] {
                return false;
            }
            if w.n_v.evaluate(&h_i) != n_field * w.t_v.evaluate(&h_i) * w.v[i] {
                return false;
            }
        }

        // y = uᵀ M v.
        let bilinear: Fr = p
            .matrix
            .iter()
            .map(|&(i, j, m_ij)| m_ij * w.u[i] * w.v[j])
            .sum();
        bilinear == s.value
    }
}

/// Build a leaf-form `R_M` instance for the bivariate evaluation
/// `P(α, β) = y`, where `P` is the bivariate polynomial encoded by
/// `params.matrix` (per the leaf construction in `delegation.tex`).
///
/// Polynomials follow the paper's leaf shape:
/// ```text
///   N_u(X) = (X^n − 1)·α − (α^n − 1)·X       T_u(X) = X − α
///   N_v(X) = (X^n − 1)·β − (β^n − 1)·X       T_v(X) = X − β
/// ```
/// with `u = λ(α)`, `v = λ(β)`, `y = uᵀ M v`. The returned
/// `(s, w)` satisfies `MRelation::is_satisfied(params, &s, &w)`.
pub fn leaf_instance(params: &MParams, alpha: Fr, beta: Fr) -> (MStatement, MWitness) {
    let dom = Radix2EvaluationDomain::<Fr>::new(params.n)
        .expect("n must be a power of two");
    let alpha_n_minus_1 = dom.evaluate_vanishing_polynomial(alpha);
    let beta_n_minus_1 = dom.evaluate_vanishing_polynomial(beta);

    let n_u = SparsePolynomial::from_coefficients_vec(vec![
        (0, -alpha),
        (1, -alpha_n_minus_1),
        (params.n, alpha),
    ]);
    let t_u = SparsePolynomial::from_coefficients_vec(vec![
        (0, -alpha),
        (1, Fr::from(1u64)),
    ]);
    let n_v = SparsePolynomial::from_coefficients_vec(vec![
        (0, -beta),
        (1, -beta_n_minus_1),
        (params.n, beta),
    ]);
    let t_v = SparsePolynomial::from_coefficients_vec(vec![
        (0, -beta),
        (1, Fr::from(1u64)),
    ]);

    let u = dom.evaluate_all_lagrange_coefficients(alpha);
    let v = dom.evaluate_all_lagrange_coefficients(beta);

    let c_n_u = pc::commit(&params.srs, &Poly::Sparse(n_u.clone()));
    let c_t_u = pc::commit(&params.srs, &Poly::Sparse(t_u.clone()));
    let c_n_v = pc::commit(&params.srs, &Poly::Sparse(n_v.clone()));
    let c_t_v = pc::commit(&params.srs, &Poly::Sparse(t_v.clone()));

    let y: Fr = params
        .matrix
        .iter()
        .map(|&(i, j, m_ij)| m_ij * u[i] * v[j])
        .sum();

    let stmt = MStatement {
        c_n_u,
        c_t_u,
        c_n_v,
        c_t_v,
        d_n_u: params.n,
        d_t_u: 1,
        d_n_v: params.n,
        d_t_v: 1,
        value: y,
    };
    let wit = MWitness { n_u, t_u, u, n_v, t_v, v };
    (stmt, wit)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    /// Build params (identity matrix `M_n`) plus a single leaf instance from
    /// random `(α, β)`.
    fn leaf_identity(n: usize) -> (MParams, MStatement, MWitness) {
        let rng = &mut test_rng();
        let matrix: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        let srs = pc::setup(n + 1, rng);
        let params = MParams { srs, matrix, n };
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (stmt, wit) = leaf_instance(&params, alpha, beta);
        (params, stmt, wit)
    }

    #[test]
    fn leaf_roundtrip_identity() {
        let (p, s, w) = leaf_identity(4);
        assert!(MRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn leaf_roundtrip_identity_n8() {
        let (p, s, w) = leaf_identity(8);
        assert!(MRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn tampered_y_rejected() {
        let (p, mut s, w) = leaf_identity(4);
        s.value += Fr::from(1u64);
        assert!(!MRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn tampered_commitment_rejected() {
        let (p, mut s, w) = leaf_identity(4);
        // Replace c_n_u with a commitment to a different (valid) polynomial.
        let bogus = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(7u64))]);
        s.c_n_u = pc::commit(&p.srs, &Poly::Sparse(bogus));
        assert!(!MRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn degree_bound_too_small_rejected() {
        let (p, mut s, w) = leaf_identity(4);
        // n_u has degree 4; claiming d_n_u = 3 must fail the degree check.
        s.d_n_u = 3;
        assert!(!MRelation::is_satisfied(&p, &s, &w));
    }

    /// Build K = 4 leaves from random `(α_i, β_i)` pairs (sharing one params)
    /// and check each one satisfies `R_M`. This is the multi-verifier setup
    /// the protocol consumes.
    #[test]
    fn k_leaves_all_satisfy() {
        let rng = &mut test_rng();
        let n = 4;
        let k = 4;
        let matrix: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        let srs = pc::setup(n + 1, rng);
        let params = MParams { srs, matrix, n };

        let points: Vec<(Fr, Fr)> =
            (0..k).map(|_| (Fr::rand(rng), Fr::rand(rng))).collect();
        let leaves: Vec<(MStatement, MWitness)> = points
            .iter()
            .map(|&(a, b)| leaf_instance(&params, a, b))
            .collect();

        assert_eq!(leaves.len(), k);
        for (s, w) in &leaves {
            assert!(MRelation::is_satisfied(&params, s, w));
        }
    }
}
