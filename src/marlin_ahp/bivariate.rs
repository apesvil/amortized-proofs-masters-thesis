// Vendored from `arkworks-rs/marlin` `src/ahp/mod.rs` (MIT/Apache-2.0).
// Trimmed and re-typed for our Fr; adapted to arkworks 0.4.
//
// The "unnormalized bivariate Lagrange polynomial" is
//   u_H(X, Y) = (v_H(X) − v_H(Y)) / (X − Y)
// where `v_H(X) = X^|H| − 1` is the vanishing polynomial of the subgroup `H`.
// For `Y ∈ H` we have `v_H(Y) = 0`, so `u_H(X, h_i) = v_H(X)/(X − h_i)` —
// this is the building block of the Lagrange basis on `H`.

use ark_bls12_381::Fr;
use ark_ff::{fields::batch_inversion, Field};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

pub trait UnnormalizedBivariateLagrangePoly {
    /// Evaluate `u_H(x, y)` at a single point.
    fn eval_unnormalized_bivariate_lagrange_poly(&self, x: Fr, y: Fr) -> Fr;

    /// Batch evaluate `u_H(x, h_i)` for every `h_i ∈ H`, given a fixed `x`.
    /// Caller is responsible for `x ∉ H` (otherwise a denominator is zero).
    fn batch_eval_unnormalized_bivariate_lagrange_poly_with_same_inputs(&self, x: Fr) -> Vec<Fr>;
}

impl UnnormalizedBivariateLagrangePoly for Radix2EvaluationDomain<Fr> {
    fn eval_unnormalized_bivariate_lagrange_poly(&self, x: Fr, y: Fr) -> Fr {
        if x != y {
            let v_h_x = self.evaluate_vanishing_polynomial(x);
            let v_h_y = self.evaluate_vanishing_polynomial(y);
            (v_h_x - v_h_y) / (x - y)
        } else {
            // The derivative `v_H'(x) = n · x^{n-1}` is the analytic continuation
            // at the diagonal `x = y`.
            self.size_as_field_element * x.pow([(self.size() - 1) as u64])
        }
    }

    fn batch_eval_unnormalized_bivariate_lagrange_poly_with_same_inputs(&self, x: Fr) -> Vec<Fr> {
        let v_h_x = self.evaluate_vanishing_polynomial(x);
        let elements: Vec<Fr> = self.elements().collect();
        let mut denoms: Vec<Fr> = elements.iter().map(|&h| x - h).collect();
        batch_inversion(&mut denoms);
        denoms.iter().map(|&d_inv| v_h_x * d_inv).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    /// `u_H(α, h_i)·h_i/n` should equal the standard Lagrange `L_i(α)`.
    #[test]
    fn batch_matches_lagrange() {
        let n = 8usize;
        let dom = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let rng = &mut test_rng();
        let alpha = Fr::rand(rng);

        let n_fr = Fr::from(n as u64);
        let elems: Vec<Fr> = dom.elements().collect();
        let us = dom.batch_eval_unnormalized_bivariate_lagrange_poly_with_same_inputs(alpha);
        let lagrange = dom.evaluate_all_lagrange_coefficients(alpha);

        for (i, &h_i) in elems.iter().enumerate() {
            assert_eq!(us[i] * h_i / n_fr, lagrange[i], "L_i(α) mismatch at i={i}");
        }
    }
}
