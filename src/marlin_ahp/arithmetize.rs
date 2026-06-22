// Vendored (with adaptations) from `arkworks-rs/marlin` `src/ahp/constraint_systems.rs`
// (MIT/Apache-2.0). We drop `LabeledPolynomial` (we don't use ark-poly-commit's
// batched-opening API) and ship raw `DensePolynomial<Fr>`. We also accept the
// three matrices as sparse `(row, col, value)` triples rather than going through
// `ConstraintSynthesizer`, since our matrices live in `AbcParams`.
//
// Algebraic identity being arithmetized (Marlin §5.3):
//   M(α, β) = v_H(α)·v_H(β) · Σ_{k ∈ K}  val_M(k) / ( (α − row(k))·(β − col(k)) )
//
// where the indexer assigns, for each non-zero (i, j) in `A ∪ B ∪ C`,
//   row(k) = h_i,  col(k) = h_j,  val_M(k) = M[i, j] · h_i · h_j / |H|².
//
// `row_col(k) = row(k)·col(k)` is interpolated separately (degree ≤ |K|−1)
// so the verifier's denom polynomial b̂(X) := αβ − α·col(X) − β·row(X) + row_col(X)
// agrees with the genuine `(α − row(X))·(β − col(X))` *on K* without paying
// the degree-2(|K|−1) of the actual product.

use std::collections::BTreeSet;

use ark_bls12_381::Fr;
use ark_ff::{Field, Zero};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};

#[derive(Clone, Debug)]
pub struct MatrixEvals {
    pub row: Vec<Fr>,
    pub col: Vec<Fr>,
    pub val_a: Vec<Fr>,
    pub val_b: Vec<Fr>,
    pub val_c: Vec<Fr>,
    pub row_col: Vec<Fr>,
}

#[derive(Clone, Debug)]
pub struct MatrixArithmetization {
    pub row: DensePolynomial<Fr>,
    pub col: DensePolynomial<Fr>,
    pub val_a: DensePolynomial<Fr>,
    pub val_b: DensePolynomial<Fr>,
    pub val_c: DensePolynomial<Fr>,
    pub row_col: DensePolynomial<Fr>,
    pub evals_on_k: MatrixEvals,
}

/// Build the row/col/val_M/row_col polynomials over the indexer domain `K`.
///
/// `domain_h` is the n-th-roots-of-unity subgroup that the matrices' row/col
/// indices live on. `domain_k` is the indexer's domain — its size must be
/// a power of two `≥` the joint support size of `(A, B, C)`.
pub fn arithmetize(
    matrix_a: &[(usize, usize, Fr)],
    matrix_b: &[(usize, usize, Fr)],
    matrix_c: &[(usize, usize, Fr)],
    domain_h: &Radix2EvaluationDomain<Fr>,
    domain_k: &Radix2EvaluationDomain<Fr>,
) -> MatrixArithmetization {
    let k_size = domain_k.size();
    let n = domain_h.size();
    let n_fr = Fr::from(n as u64);
    let n_sq_inv = (n_fr * n_fr).inverse().expect("|H|² is non-zero");

    let h_elems: Vec<Fr> = domain_h.elements().collect();

    // Joint support: every (i, j) that's non-zero in at least one of A, B, C.
    // Sorted for deterministic indexing — k_idx = position in this sorted list.
    let mut support: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(i, j, _) in matrix_a {
        support.insert((i, j));
    }
    for &(i, j, _) in matrix_b {
        support.insert((i, j));
    }
    for &(i, j, _) in matrix_c {
        support.insert((i, j));
    }
    let support: Vec<(usize, usize)> = support.into_iter().collect();
    assert!(
        support.len() <= k_size,
        "|K| = {k_size} must cover the joint support of size {}",
        support.len()
    );

    let mut row = vec![Fr::zero(); k_size];
    let mut col = vec![Fr::zero(); k_size];
    let mut val_a = vec![Fr::zero(); k_size];
    let mut val_b = vec![Fr::zero(); k_size];
    let mut val_c = vec![Fr::zero(); k_size];
    let mut row_col = vec![Fr::zero(); k_size];

    let h_one = h_elems[0]; // dummy slot value
    let h_one_sq = h_one * h_one;

    for (k_idx, &(i, j)) in support.iter().enumerate() {
        let h_i = h_elems[i];
        let h_j = h_elems[j];
        let norm = h_i * h_j * n_sq_inv;
        row[k_idx] = h_i;
        col[k_idx] = h_j;
        val_a[k_idx] = lookup(matrix_a, i, j) * norm;
        val_b[k_idx] = lookup(matrix_b, i, j) * norm;
        val_c[k_idx] = lookup(matrix_c, i, j) * norm;
        row_col[k_idx] = h_i * h_j;
    }
    // Pad: dummies have val_M = 0 (no contribution to the sum) and row = col = h_0.
    for slot in support.len()..k_size {
        row[slot] = h_one;
        col[slot] = h_one;
        row_col[slot] = h_one_sq;
    }

    let row_poly = DensePolynomial::from_coefficients_vec(domain_k.ifft(&row));
    let col_poly = DensePolynomial::from_coefficients_vec(domain_k.ifft(&col));
    let val_a_poly = DensePolynomial::from_coefficients_vec(domain_k.ifft(&val_a));
    let val_b_poly = DensePolynomial::from_coefficients_vec(domain_k.ifft(&val_b));
    let val_c_poly = DensePolynomial::from_coefficients_vec(domain_k.ifft(&val_c));
    let row_col_poly = DensePolynomial::from_coefficients_vec(domain_k.ifft(&row_col));

    MatrixArithmetization {
        row: row_poly,
        col: col_poly,
        val_a: val_a_poly,
        val_b: val_b_poly,
        val_c: val_c_poly,
        row_col: row_col_poly,
        evals_on_k: MatrixEvals { row, col, val_a, val_b, val_c, row_col },
    }
}

fn lookup(matrix: &[(usize, usize, Fr)], i: usize, j: usize) -> Fr {
    matrix
        .iter()
        .find(|&&(r, c, _)| r == i && c == j)
        .map(|&(_, _, v)| v)
        .unwrap_or_else(Fr::zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_poly::Polynomial;
    use ark_std::test_rng;

    /// Sum of `val_M(k)/((α − row(k))·(β − col(k)))` over `k ∈ K`, divided by
    /// `v_H(α)·v_H(β)`, should reproduce `M(α, β) = λ(α)ᵀ·M·λ(β)`.
    #[test]
    fn arithmetization_matches_bivariate_eval() {
        let rng = &mut test_rng();
        let n = 8;
        let dom_h = Radix2EvaluationDomain::<Fr>::new(n).unwrap();

        // Sparse-ish matrices.
        let a: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
        let b: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, (i + 1) % n, Fr::rand(rng))).collect();
        let c: Vec<(usize, usize, Fr)> = vec![
            (0, 0, Fr::rand(rng)),
            (3, 7, Fr::rand(rng)),
            (5, 2, Fr::rand(rng)),
        ];

        // |K| = next pow2 ≥ joint support.
        let joint_size = {
            let mut s = BTreeSet::new();
            for &(i, j, _) in &a {
                s.insert((i, j));
            }
            for &(i, j, _) in &b {
                s.insert((i, j));
            }
            for &(i, j, _) in &c {
                s.insert((i, j));
            }
            s.len()
        };
        let k_size = joint_size.max(2).next_power_of_two();
        let dom_k = Radix2EvaluationDomain::<Fr>::new(k_size).unwrap();

        let arith = arithmetize(&a, &b, &c, &dom_h, &dom_k);

        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);

        let v_h_a = dom_h.evaluate_vanishing_polynomial(alpha);
        let v_h_b = dom_h.evaluate_vanishing_polynomial(beta);
        let lambda_alpha = dom_h.evaluate_all_lagrange_coefficients(alpha);
        let lambda_beta = dom_h.evaluate_all_lagrange_coefficients(beta);

        for (matrix, val_evals) in [
            (&a, &arith.evals_on_k.val_a),
            (&b, &arith.evals_on_k.val_b),
            (&c, &arith.evals_on_k.val_c),
        ] {
            // Reference value of M(α, β).
            let reference: Fr = matrix
                .iter()
                .map(|&(i, j, m)| m * lambda_alpha[i] * lambda_beta[j])
                .sum();

            // Sum over K of val(k) / ((α − row(k))(β − col(k))).
            let mut acc = Fr::zero();
            for k in 0..dom_k.size() {
                let denom = (alpha - arith.evals_on_k.row[k])
                    * (beta - arith.evals_on_k.col[k]);
                if !denom.is_zero() {
                    acc += val_evals[k] * denom.inverse().unwrap();
                }
            }
            let recovered = v_h_a * v_h_b * acc;
            assert_eq!(recovered, reference, "arithmetization roundtrip failed");
        }
    }

    /// row_col(X) interpolates row(k)·col(k) on K — verify via evaluations.
    #[test]
    fn row_col_interpolates_product_on_k() {
        let rng = &mut test_rng();
        let n = 4;
        let dom_h = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let a: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
        let dom_k = Radix2EvaluationDomain::<Fr>::new(4).unwrap();

        let arith = arithmetize(&a, &[], &[], &dom_h, &dom_k);

        for k_idx in 0..dom_k.size() {
            let element = dom_k.element(k_idx);
            let row_at_k = arith.row.evaluate(&element);
            let col_at_k = arith.col.evaluate(&element);
            let row_col_at_k = arith.row_col.evaluate(&element);
            assert_eq!(row_at_k * col_at_k, row_col_at_k, "row_col mismatch at k={k_idx}");
        }
    }
}
