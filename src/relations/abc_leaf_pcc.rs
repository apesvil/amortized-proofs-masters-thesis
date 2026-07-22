use ark_bls12_381::Fr;
use ark_poly::{
    univariate::SparsePolynomial, EvaluationDomain, Radix2EvaluationDomain,
};

use crate::pc::{self, Poly};
use crate::relations::pcc::{
    Constraint, Monomial, PccParams, PccStatement, PccWitness,
};

/// Build the per-verifier R_PCC instance that proves the leaf-encoding
/// correctness of `(N_u, T_u, N_v, T_v)` for a leaf `(α, β)` of R_{A,B,C}.
///
/// Witness polynomials (must match `abc::leaf_instance`):
/// ```text
///   N_u(X) = (X^n − 1)·α − (α^n − 1)·X       T_u(X) = X − α
///   N_v(X) = (X^n − 1)·β − (β^n − 1)·X       T_v(X) = X − β
/// ```
///
/// Constraints encode the rational identity `N_u/T_u = M_α(X)/(X − α)`
/// (and likewise for `v`), which together with the strict degree bounds
/// pins each pair to the leaf shape up to the (paper-allowed) joint scaling:
/// ```text
///   Q_u : (X − α)·N_u(X) − ((X^n − 1)·α − (α^n − 1)·X)·T_u(X) = 0
///   Q_v : (X − β)·N_v(X) − ((X^n − 1)·β − (β^n − 1)·X)·T_v(X) = 0
/// ```
///
/// Witness indices in the constraints: `Y_0 = N_u, Y_1 = T_u, Y_2 = N_v, Y_3 = T_v`.
///
/// Note: matrix-independent — same shape works for every leaf regardless of
/// which `(A, B, C)` the R_{A,B,C} instance is over.
pub fn leaf_correctness_pcc(
    params: &PccParams,
    n: usize,
    alpha: Fr,
    beta: Fr,
) -> (PccStatement, PccWitness) {
    let dom =
        Radix2EvaluationDomain::<Fr>::new(n).expect("n must be a power of two");
    let alpha_n_minus_1 = dom.evaluate_vanishing_polynomial(alpha);
    let beta_n_minus_1 = dom.evaluate_vanishing_polynomial(beta);

    let n_u = SparsePolynomial::from_coefficients_vec(vec![
        (0, -alpha),
        (1, -alpha_n_minus_1),
        (n, alpha),
    ]);
    let t_u = SparsePolynomial::from_coefficients_vec(vec![
        (0, -alpha),
        (1, Fr::from(1u64)),
    ]);
    let n_v = SparsePolynomial::from_coefficients_vec(vec![
        (0, -beta),
        (1, -beta_n_minus_1),
        (n, beta),
    ]);
    let t_v = SparsePolynomial::from_coefficients_vec(vec![
        (0, -beta),
        (1, Fr::from(1u64)),
    ]);

    let c_n_u = pc::commit(&params.srs, &Poly::Sparse(n_u.clone()));
    let c_t_u = pc::commit(&params.srs, &Poly::Sparse(t_u.clone()));
    let c_n_v = pc::commit(&params.srs, &Poly::Sparse(n_v.clone()));
    let c_t_v = pc::commit(&params.srs, &Poly::Sparse(t_v.clone()));

    let q_u = Constraint {
        monomials: vec![
            Monomial { coeff: Fr::from(1u64), x_deg: 1, y_terms: vec![(0, 1)] },
            Monomial { coeff: -alpha,         x_deg: 0, y_terms: vec![(0, 1)] },
            Monomial { coeff: -alpha,         x_deg: n, y_terms: vec![(1, 1)] },
            Monomial { coeff:  alpha,         x_deg: 0, y_terms: vec![(1, 1)] },
            Monomial { coeff: alpha_n_minus_1, x_deg: 1, y_terms: vec![(1, 1)] },
        ],
    };
    let q_v = Constraint {
        monomials: vec![
            Monomial { coeff: Fr::from(1u64), x_deg: 1, y_terms: vec![(2, 1)] },
            Monomial { coeff: -beta,          x_deg: 0, y_terms: vec![(2, 1)] },
            Monomial { coeff: -beta,          x_deg: n, y_terms: vec![(3, 1)] },
            Monomial { coeff:  beta,          x_deg: 0, y_terms: vec![(3, 1)] },
            Monomial { coeff: beta_n_minus_1,  x_deg: 1, y_terms: vec![(3, 1)] },
        ],
    };

    let stmt = PccStatement {
        commitments: vec![c_n_u, c_t_u, c_n_v, c_t_v],
        degrees: vec![n + 1, 2, n + 1, 2],
        constraints: vec![q_u, q_v],
    };
    let wit = PccWitness { polynomials: vec![n_u, t_u, n_v, t_v] };
    (stmt, wit)
}

/// Glue helper: given the K × κ R_PCC bundle `AbcFold` emitted and the K
/// per-leaf challenges, append each leaf's correctness R_PCC instance so
/// the result is K × (κ+1). Correctness goes last in each inner vec.
pub fn full_pcc_bundles(
    params: &PccParams,
    n: usize,
    leaf_challenges: &[(Fr, Fr)],
    fold_pcc_bundles: Vec<Vec<(PccStatement, PccWitness)>>,
) -> Vec<Vec<(PccStatement, PccWitness)>> {
    assert_eq!(
        leaf_challenges.len(),
        fold_pcc_bundles.len(),
        "leaf_challenges and fold_pcc_bundles must have matching outer length K",
    );
    fold_pcc_bundles
        .into_iter()
        .zip(leaf_challenges)
        .map(|(mut bundle, &(alpha, beta))| {
            bundle.push(leaf_correctness_pcc(params, n, alpha, beta));
            bundle
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::relation::Relation;
    use crate::relations::pcc::PccRelation;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    fn params(max_deg: usize) -> PccParams {
        PccParams { srs: pc::setup(max_deg, &mut test_rng()) }
    }

    #[test]
    fn leaf_correctness_satisfied_n4() {
        let n = 4;
        let p = params(n + 1);
        let rng = &mut test_rng();
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (s, w) = leaf_correctness_pcc(&p, n, alpha, beta);
        assert!(PccRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn leaf_correctness_satisfied_n8() {
        let n = 8;
        let p = params(n + 1);
        let rng = &mut test_rng();
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (s, w) = leaf_correctness_pcc(&p, n, alpha, beta);
        assert!(PccRelation::is_satisfied(&p, &s, &w));
    }

    /// A leaf whose N_u differs from the honest one by a multiple of the
    /// vanishing polynomial Z_H(X) = X^n − 1 still agrees with N_u on all of H
    /// (Z_H vanishes there) and stays within the degree bound (deg Z_H = n =
    /// d_n_u), so it survives the spot-check + degree checks AbcRelation uses.
    /// But it corrupts N_u(β)/(n·T_u(β)) at the random β RokP reads, and
    /// leaf_correctness_pcc's Q_u — a full polynomial identity — rejects it.
    #[test]
    fn zh_multiple_forgery_rejected() {
        use ark_poly::Polynomial;
        let n = 4;
        let p = params(n + 1);
        let rng = &mut test_rng();
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (mut s, mut w) = leaf_correctness_pcc(&p, n, alpha, beta);

        // Forge N_u' = N_u + c·Z_H (c ≠ 0). Z_H = X^n − 1.
        let z_h = SparsePolynomial::from_coefficients_vec(vec![
            (0, -Fr::from(1u64)),
            (n, Fr::from(1u64)),
        ]);
        let forged = &w.polynomials[0] + &(&z_h * Fr::from(7u64));

        // Passes the checks AbcRelation uses: agrees with honest N_u on all of
        // H, and stays within the degree bound d_n_u = n.
        let dom = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        for h in dom.elements() {
            assert_eq!(forged.evaluate(&h), w.polynomials[0].evaluate(&h));
        }
        assert!(forged.degree() <= n);

        // Self-consistent malicious witness: commitment matches the forged poly.
        s.commitments[0] = pc::commit(&p.srs, &Poly::Sparse(forged.clone()));
        w.polynomials[0] = forged;

        // Only Q_u catches it.
        assert!(!PccRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn wrong_alpha_in_constraint_rejected() {
        let n = 4;
        let p = params(n + 1);
        let rng = &mut test_rng();
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (mut s, w) = leaf_correctness_pcc(&p, n, alpha, beta);
        let alpha_prime = alpha + Fr::from(1u64);
        let (s_bad, _) = leaf_correctness_pcc(&p, n, alpha_prime, beta);
        s.constraints[0] = s_bad.constraints[0].clone();
        assert!(!PccRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn tampered_t_u_rejected() {
        let n = 4;
        let p = params(n + 1);
        let rng = &mut test_rng();
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (mut s, mut w) = leaf_correctness_pcc(&p, n, alpha, beta);
        let bad_t_u = SparsePolynomial::from_coefficients_vec(vec![
            (0, -alpha + Fr::from(1u64)),
            (1, Fr::from(1u64)),
        ]);
        w.polynomials[1] = bad_t_u.clone();
        s.commitments[1] = pc::commit(&p.srs, &Poly::Sparse(bad_t_u));
        assert!(!PccRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn tight_degree_bound_rejected_when_loosened() {
        let n = 4;
        let p = params(n + 1);
        let rng = &mut test_rng();
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (mut s, w) = leaf_correctness_pcc(&p, n, alpha, beta);
        s.degrees[0] = n;
        assert!(!PccRelation::is_satisfied(&p, &s, &w));
    }

    #[test]
    fn full_pcc_bundles_appends_in_order() {
        let n = 4;
        let p = params(n + 1);
        let rng = &mut test_rng();
        let k = 3;
        let challenges: Vec<(Fr, Fr)> =
            (0..k).map(|_| (Fr::rand(rng), Fr::rand(rng))).collect();

        let fold: Vec<Vec<(PccStatement, PccWitness)>> = (0..k)
            .map(|leaf| {
                vec![(
                    PccStatement {
                        commitments: vec![],
                        degrees: vec![leaf],
                        constraints: vec![],
                    },
                    PccWitness { polynomials: vec![] },
                )]
            })
            .collect();

        let out = full_pcc_bundles(&p, n, &challenges, fold);
        assert_eq!(out.len(), k);
        for (leaf, bundle) in out.iter().enumerate() {
            assert_eq!(bundle.len(), 2);
            assert_eq!(bundle[0].0.degrees, vec![leaf]);
            assert!(PccRelation::is_satisfied(&p, &bundle[1].0, &bundle[1].1));
        }
    }
}
