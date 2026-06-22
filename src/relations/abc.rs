use ark_bls12_381::Fr;
use ark_poly::{
    univariate::SparsePolynomial,
    EvaluationDomain, Polynomial, Radix2EvaluationDomain,
};

use crate::core::relation::Relation;
use crate::pc::{self, Comm, Poly, Srs};

/// The R_{A,B,C} relation (extension of R_M, Section 8 of the paper, to the
/// R1CS-style triple of matrices Marlin operates on). The witnesses `u, v`
/// are encoded by the same rational pairs `(N_u, T_u)` and `(N_v, T_v)` as
/// in R_M; the relation pins three bilinear forms simultaneously:
/// `y_X = uᵀ·X·v` for `X ∈ {A, B, C}`.
pub struct AbcRelation;

pub struct AbcParams {
    /// SRS for the polynomial commitment scheme. Bundles ck and vk.
    pub srs: Srs,
    /// The three matrices, each as sparse triples `(row, col, value)`.
    pub matrix_a: Vec<(usize, usize, Fr)>,
    pub matrix_b: Vec<(usize, usize, Fr)>,
    pub matrix_c: Vec<(usize, usize, Fr)>,
    /// `|H|`, must be a power of two. Defines the Lagrange basis.
    pub n: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AbcStatement {
    pub c_n_u: Comm,
    pub c_t_u: Comm,
    pub c_n_v: Comm,
    pub c_t_v: Comm,
    pub d_n_u: usize,
    pub d_t_u: usize,
    pub d_n_v: usize,
    pub d_t_v: usize,
    pub y_a: Fr,
    pub y_b: Fr,
    pub y_c: Fr,
}

#[derive(Clone)]
pub struct AbcWitness {
    pub n_u: SparsePolynomial<Fr>,
    pub t_u: SparsePolynomial<Fr>,
    pub u: Vec<Fr>,
    pub n_v: SparsePolynomial<Fr>,
    pub t_v: SparsePolynomial<Fr>,
    pub v: Vec<Fr>,
}

impl Relation for AbcRelation {
    type Params = AbcParams;
    type Statement = AbcStatement;
    type Witness = AbcWitness;

    fn is_satisfied(p: &AbcParams, s: &AbcStatement, w: &AbcWitness) -> bool {
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

        // Commitment validity, paper-strict.
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

        // Encoding identity at each h_i ∈ H, both sides.
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

        // Three bilinear forms.
        bilinear(&p.matrix_a, &w.u, &w.v) == s.y_a
            && bilinear(&p.matrix_b, &w.u, &w.v) == s.y_b
            && bilinear(&p.matrix_c, &w.u, &w.v) == s.y_c
    }
}

/// `uᵀ·M·v` for a sparse `M`.
pub fn bilinear(matrix: &[(usize, usize, Fr)], u: &[Fr], v: &[Fr]) -> Fr {
    matrix
        .iter()
        .map(|&(i, j, m_ij)| m_ij * u[i] * v[j])
        .sum()
}

/// Build a leaf-form `R_{A,B,C}` instance for the bivariate evaluations
/// `P_X(α, β) = y_X` for each `X ∈ {A, B, C}`. Polynomials follow the
/// paper's leaf shape:
/// ```text
///   N_u(X) = (X^n − 1)·α − (α^n − 1)·X       T_u(X) = X − α
///   N_v(X) = (X^n − 1)·β − (β^n − 1)·X       T_v(X) = X − β
/// ```
/// with `u = λ(α)`, `v = λ(β)`, and `y_X = λ(α)ᵀ·X·λ(β)`.
pub fn leaf_instance(params: &AbcParams, alpha: Fr, beta: Fr) -> (AbcStatement, AbcWitness) {
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

    let y_a = bilinear(&params.matrix_a, &u, &v);
    let y_b = bilinear(&params.matrix_b, &u, &v);
    let y_c = bilinear(&params.matrix_c, &u, &v);

    let stmt = AbcStatement {
        c_n_u,
        c_t_u,
        c_n_v,
        c_t_v,
        d_n_u: params.n,
        d_t_u: 1,
        d_n_v: params.n,
        d_t_v: 1,
        y_a,
        y_b,
        y_c,
    };
    let wit = AbcWitness { n_u, t_u, u, n_v, t_v, v };
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

    /// Three different matrices: identity (A), a left-shift permutation (B),
    /// and a diagonal of small scalars (C).
    fn abc_matrices<R: ark_std::rand::Rng>(n: usize, rng: &mut R) -> (
        Vec<(usize, usize, Fr)>,
        Vec<(usize, usize, Fr)>,
        Vec<(usize, usize, Fr)>,
    ) {
        let a: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        let b: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, (i + 1) % n, Fr::from(1u64))).collect();
        let c: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
        (a, b, c)
    }

    fn leaf_setup(n: usize) -> (AbcParams, AbcStatement, AbcWitness) {
        let rng = &mut test_rng();
        let (a, b, c) = abc_matrices(n, rng);
        let srs = pc::setup(n + 1, rng);
        let params = AbcParams { srs, matrix_a: a, matrix_b: b, matrix_c: c, n };
        let (s, w) = leaf_instance(&params, Fr::rand(rng), Fr::rand(rng));
        (params, s, w)
    }

    #[test]
    fn leaf_roundtrip_n4() {
        let (p, s, w) = leaf_setup(4);
        assert!(AbcRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn leaf_roundtrip_n8() {
        let (p, s, w) = leaf_setup(8);
        assert!(AbcRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn tampered_y_a_rejected() {
        let (p, mut s, w) = leaf_setup(4);
        s.y_a += Fr::from(1u64);
        assert!(!AbcRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn tampered_y_b_rejected() {
        let (p, mut s, w) = leaf_setup(4);
        s.y_b += Fr::from(1u64);
        assert!(!AbcRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn tampered_y_c_rejected() {
        let (p, mut s, w) = leaf_setup(4);
        s.y_c += Fr::from(1u64);
        assert!(!AbcRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn tampered_commitment_rejected() {
        let (p, mut s, w) = leaf_setup(4);
        let bogus = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(7u64))]);
        s.c_n_u = pc::commit(&p.srs, &Poly::Sparse(bogus));
        assert!(!AbcRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn degree_bound_too_small_rejected() {
        let (p, mut s, w) = leaf_setup(4);
        s.d_n_u = 3;
        assert!(!AbcRelation::is_satisfied(&p, &s, &w));
    }

    /// K = 4 leaves: each must satisfy the relation.
    #[test]
    fn k_leaves_all_satisfy() {
        let rng = &mut test_rng();
        let n = 4;
        let k = 4;
        let (a, b, c) = abc_matrices(n, rng);
        let srs = pc::setup(n + 1, rng);
        let params = AbcParams { srs, matrix_a: a, matrix_b: b, matrix_c: c, n };

        let leaves: Vec<(AbcStatement, AbcWitness)> = (0..k)
            .map(|_| leaf_instance(&params, Fr::rand(rng), Fr::rand(rng)))
            .collect();
        assert_eq!(leaves.len(), k);
        for (s, w) in &leaves {
            assert!(AbcRelation::is_satisfied(&params, s, w));
        }
    }
}
