use ark_bls12_381::Fr;
use ark_ff::{fields::batch_inversion, Field, One, Zero};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial,
    Radix2EvaluationDomain,
};

use crate::marlin_ahp::arithmetize::{arithmetize, MatrixArithmetization};
use crate::pc::{self, Comm, Opening, Poly, Srs};
use crate::reductions::poly_util::x_shift_dense;
use crate::relations::p::PStatement;
use crate::transcript::Blake3Transcript;

/// Marlin's inner sumcheck ("lincheck"), reused as the proof of our R_P
/// statement `(α, β, y, η)` where `y = P_A(α,β) + η·P_B(α,β) + η²·P_C(α,β)`.
///
/// Protocol (after the matrices have been indexed once):
/// 1. Prover and verifier reconstruct (α, β, y, η) from the PStatement.
/// 2. Prover builds `f(X)` over `K` such that `f(k) = a(k)/b(k)`, where
///    `a(X) = v_H(α)·v_H(β) · (val_A + η·val_B + η²·val_C)(X)` and
///    `b(X) = (α − row(X))·(β − col(X))`. Decomposes `f(X) = y/|K| + X·g_2(X)`.
/// 3. Prover commits `g_2, h_2` where `h_2 = (a − b̂·f)/v_K` and `b̂` is the
///    degree-`|K|−1` "denom proxy" `αβ − α·col − β·row + row_col`, plus the
///    shifted `g'_2(X) = X^{D−(|K|−1)}·g_2(X)`. The shift binds
///    `deg g_2 ≤ |K| − 2`, which univariate-sumcheck soundness requires:
///    without it, `f̃ = y'/|K| + X·g_2` can absorb a multiple of `v_K` and
///    pass for an arbitrary claimed sum `y'`
///    (see `tests::high_degree_g2_forgery_rejected`).
/// 4. Verifier squeezes `γ` and asks for openings of
///    `g_2, h_2, row, col, val_A, val_B, val_C, row_col` at `γ`.
/// 5. Verifier checks `a(γ) − b̂(γ)·(γ·g_2(γ) + y/|K|) = h_2(γ)·v_K(γ)`.
/// 6. A single η'-batched KZG opening discharges all 9 openings — the 8
///    evaluations plus the shifted-`g_2` slot, whose expected value the
///    verifier derives as `γ^{D−(|K|−1)}·ev_g2`.
pub struct InnerLincheck;

/// One-time per (matrix triple, n): arithmetize and commit to the six index
/// polynomials. Reused across many R_P proofs.
pub struct InnerLincheckIndex {
    pub arith: MatrixArithmetization,
    pub domain_h: Radix2EvaluationDomain<Fr>,
    pub domain_k: Radix2EvaluationDomain<Fr>,
    pub c_row: Comm,
    pub c_col: Comm,
    pub c_val_a: Comm,
    pub c_val_b: Comm,
    pub c_val_c: Comm,
    pub c_row_col: Comm,
}

#[derive(Clone)]
pub struct InnerLincheckProof {
    pub c_g2: Comm,
    pub c_h2: Comm,
    /// Commitment to `X^{D−(|K|−1)}·g_2(X)` — the degree-binding shift.
    pub c_g2_shift: Comm,
    pub ev_g2: Fr,
    pub ev_h2: Fr,
    pub ev_row: Fr,
    pub ev_col: Fr,
    pub ev_val_a: Fr,
    pub ev_val_b: Fr,
    pub ev_val_c: Fr,
    pub ev_row_col: Fr,
    pub batched_opening: Opening,
}

impl InnerLincheck {
    /// Indexer phase: one-time per `(A, B, C, n)` setup. The SRS must support
    /// degrees up to `|K| − 1` (we share one SRS across the whole project,
    /// sized for the largest user — see `docs/decisions/shared_srs.md`).
    pub fn index(
        srs: &Srs,
        matrix_a: &[(usize, usize, Fr)],
        matrix_b: &[(usize, usize, Fr)],
        matrix_c: &[(usize, usize, Fr)],
        n: usize,
    ) -> InnerLincheckIndex {
        let domain_h = Radix2EvaluationDomain::<Fr>::new(n).expect("n must be a power of two");
        let joint = joint_support_size(matrix_a, matrix_b, matrix_c);
        let k_size = joint.max(2).next_power_of_two();
        let domain_k = Radix2EvaluationDomain::<Fr>::new(k_size).expect("|K| power of two");

        let arith = arithmetize(matrix_a, matrix_b, matrix_c, &domain_h, &domain_k);

        let c_row = pc::commit(srs, &Poly::Dense(arith.row.clone()));
        let c_col = pc::commit(srs, &Poly::Dense(arith.col.clone()));
        let c_val_a = pc::commit(srs, &Poly::Dense(arith.val_a.clone()));
        let c_val_b = pc::commit(srs, &Poly::Dense(arith.val_b.clone()));
        let c_val_c = pc::commit(srs, &Poly::Dense(arith.val_c.clone()));
        let c_row_col = pc::commit(srs, &Poly::Dense(arith.row_col.clone()));

        InnerLincheckIndex {
            arith,
            domain_h,
            domain_k,
            c_row,
            c_col,
            c_val_a,
            c_val_b,
            c_val_c,
            c_row_col,
        }
    }

    pub fn prove(
        srs: &Srs,
        index: &InnerLincheckIndex,
        stmt: &PStatement,
        transcript: &mut Blake3Transcript,
    ) -> InnerLincheckProof {
        let domain_h = &index.domain_h;
        let domain_k = &index.domain_k;
        let evals = &index.arith.evals_on_k;
        let k_size = domain_k.size();
        let k_size_fr = Fr::from(k_size as u64);

        absorb_stmt(transcript, stmt);
        absorb_index_meta(transcript, index);

        // Cached scalars.
        let v_h_alpha = domain_h.evaluate_vanishing_polynomial(stmt.alpha);
        let v_h_beta = domain_h.evaluate_vanishing_polynomial(stmt.beta);
        let v_h_prod = v_h_alpha * v_h_beta;
        let eta_b = stmt.eta;
        let eta_c = stmt.eta * stmt.eta;

        // a(k) and b(k) on K.
        let mut a_evals = vec![Fr::zero(); k_size];
        let mut b_evals = vec![Fr::zero(); k_size];
        for k in 0..k_size {
            a_evals[k] = v_h_prod
                * (evals.val_a[k] + eta_b * evals.val_b[k] + eta_c * evals.val_c[k]);
            b_evals[k] = (stmt.alpha - evals.row[k]) * (stmt.beta - evals.col[k]);
        }

        // f(k) = a(k) / b(k) via batch inversion.
        let mut b_inv = b_evals.clone();
        batch_inversion(&mut b_inv);
        let f_evals: Vec<Fr> = a_evals.iter().zip(&b_inv).map(|(a, bi)| *a * *bi).collect();

        debug_assert_eq!(
            f_evals.iter().copied().sum::<Fr>(),
            stmt.y,
            "Σ f(k) must equal stmt.y"
        );

        // f(X) coefficients via IFFT.
        let f_coeffs = domain_k.ifft(&f_evals);
        let k_inv = k_size_fr.inverse().expect("|K| non-zero");
        let y_over_k = stmt.y * k_inv;
        debug_assert_eq!(f_coeffs[0], y_over_k);

        // g_2(X) := (f(X) − y/|K|) / X, i.e. drop the constant term.
        let g2_coeffs: Vec<Fr> = f_coeffs[1..].to_vec();
        let g2_poly = DensePolynomial::from_coefficients_vec(g2_coeffs);
        let f_poly = DensePolynomial::from_coefficients_vec(f_coeffs);

        // a(X) and b̂(X) as polynomials.
        let a_poly = DensePolynomial::from_coefficients_vec(domain_k.ifft(&a_evals));
        let b_hat_poly = build_b_hat(
            stmt.alpha,
            stmt.beta,
            &index.arith.col,
            &index.arith.row,
            &index.arith.row_col,
        );

        // h_2(X) = (a − b̂·f) / v_K.
        let bf_poly = &b_hat_poly * &f_poly;
        let numerator = &a_poly - &bf_poly;
        let (h2_poly, remainder) = numerator.divide_by_vanishing_poly(*domain_k);
        debug_assert!(remainder.is_zero(), "a − b̂·f must vanish on K");

        // Degree binding for g_2: sumcheck soundness needs deg g_2 ≤ |K| − 2.
        // Committing X^{D−(|K|−1)}·g_2 is only possible under the size-D SRS
        // when that bound holds; the batch below ties the two commitments
        // together at γ (slot value γ^{D−(|K|−1)}·ev_g2).
        let big_d = srs.powers_g1.len();
        let g2_shift_poly = x_shift_dense(&g2_poly, big_d - (k_size - 1));
        let c_g2_shift = pc::commit(srs, &Poly::Dense(g2_shift_poly.clone()));

        let c_g2 = pc::commit(srs, &Poly::Dense(g2_poly.clone()));
        let c_h2 = pc::commit(srs, &Poly::Dense(h2_poly.clone()));

        transcript.absorb(b"il::c_g2", &c_g2);
        transcript.absorb(b"il::c_h2", &c_h2);
        transcript.absorb(b"il::c_g2_shift", &c_g2_shift);

        let gamma: Fr = transcript.squeeze_field(b"il::gamma");

        let ev_g2 = g2_poly.evaluate(&gamma);
        let ev_h2 = h2_poly.evaluate(&gamma);
        let ev_row = index.arith.row.evaluate(&gamma);
        let ev_col = index.arith.col.evaluate(&gamma);
        let ev_val_a = index.arith.val_a.evaluate(&gamma);
        let ev_val_b = index.arith.val_b.evaluate(&gamma);
        let ev_val_c = index.arith.val_c.evaluate(&gamma);
        let ev_row_col = index.arith.row_col.evaluate(&gamma);

        transcript.absorb(b"il::ev_g2", &ev_g2);
        transcript.absorb(b"il::ev_h2", &ev_h2);
        transcript.absorb(b"il::ev_row", &ev_row);
        transcript.absorb(b"il::ev_col", &ev_col);
        transcript.absorb(b"il::ev_val_a", &ev_val_a);
        transcript.absorb(b"il::ev_val_b", &ev_val_b);
        transcript.absorb(b"il::ev_val_c", &ev_val_c);
        transcript.absorb(b"il::ev_row_col", &ev_row_col);

        let eta_prime: Fr = transcript.squeeze_field(b"il::eta_prime");

        // η'-batched polynomial in the same order as commits/evals below.
        // The 9th slot is the shifted g_2, whose expected evaluation the
        // verifier derives as γ^{D−(|K|−1)}·ev_g2.
        let polys = [
            &g2_poly,
            &h2_poly,
            &index.arith.row,
            &index.arith.col,
            &index.arith.val_a,
            &index.arith.val_b,
            &index.arith.val_c,
            &index.arith.row_col,
            &g2_shift_poly,
        ];
        let mut p_batch = DensePolynomial::<Fr>::zero();
        let mut pow = Fr::one();
        for p in polys {
            p_batch = &p_batch + &scale_dense(p, pow);
            pow *= eta_prime;
        }

        let (batched_opening, _opened) = pc::prove(srs, &Poly::Dense(p_batch), gamma);

        InnerLincheckProof {
            c_g2,
            c_h2,
            c_g2_shift,
            ev_g2,
            ev_h2,
            ev_row,
            ev_col,
            ev_val_a,
            ev_val_b,
            ev_val_c,
            ev_row_col,
            batched_opening,
        }
    }

    pub fn verify(
        srs: &Srs,
        index: &InnerLincheckIndex,
        stmt: &PStatement,
        proof: &InnerLincheckProof,
        transcript: &mut Blake3Transcript,
    ) -> bool {
        let domain_h = &index.domain_h;
        let domain_k = &index.domain_k;
        let k_size_fr = Fr::from(domain_k.size() as u64);

        absorb_stmt(transcript, stmt);
        absorb_index_meta(transcript, index);

        transcript.absorb(b"il::c_g2", &proof.c_g2);
        transcript.absorb(b"il::c_h2", &proof.c_h2);
        transcript.absorb(b"il::c_g2_shift", &proof.c_g2_shift);

        let gamma: Fr = transcript.squeeze_field(b"il::gamma");

        transcript.absorb(b"il::ev_g2", &proof.ev_g2);
        transcript.absorb(b"il::ev_h2", &proof.ev_h2);
        transcript.absorb(b"il::ev_row", &proof.ev_row);
        transcript.absorb(b"il::ev_col", &proof.ev_col);
        transcript.absorb(b"il::ev_val_a", &proof.ev_val_a);
        transcript.absorb(b"il::ev_val_b", &proof.ev_val_b);
        transcript.absorb(b"il::ev_val_c", &proof.ev_val_c);
        transcript.absorb(b"il::ev_row_col", &proof.ev_row_col);

        let eta_prime: Fr = transcript.squeeze_field(b"il::eta_prime");

        // Polynomial identity check at γ.
        let v_h_alpha = domain_h.evaluate_vanishing_polynomial(stmt.alpha);
        let v_h_beta = domain_h.evaluate_vanishing_polynomial(stmt.beta);
        let v_h_prod = v_h_alpha * v_h_beta;
        let eta_b = stmt.eta;
        let eta_c = stmt.eta * stmt.eta;
        let a_gamma = v_h_prod
            * (proof.ev_val_a + eta_b * proof.ev_val_b + eta_c * proof.ev_val_c);
        let b_gamma = stmt.alpha * stmt.beta
            - stmt.alpha * proof.ev_col
            - stmt.beta * proof.ev_row
            + proof.ev_row_col;
        let v_k_gamma = domain_k.evaluate_vanishing_polynomial(gamma);
        let k_inv = match k_size_fr.inverse() {
            Some(inv) => inv,
            None => return false,
        };
        let y_over_k = stmt.y * k_inv;
        let lhs = a_gamma - b_gamma * (gamma * proof.ev_g2 + y_over_k);
        let rhs = proof.ev_h2 * v_k_gamma;
        if lhs != rhs {
            return false;
        }

        // η'-batched KZG opening verification. The 9th slot ties the shifted
        // g_2 commitment to the unshifted one: its expected value is
        // γ^{D−(|K|−1)}·ev_g2, which (with both commitments fixed before γ)
        // enforces `g'_2 ≡ X^{D−(|K|−1)}·g_2` by Schwartz–Zippel and hence
        // deg g_2 ≤ |K| − 2 via SRS committability.
        let big_d = srs.powers_g1.len();
        let k_size = domain_k.size();
        let s_g2 = gamma.pow([(big_d - (k_size - 1)) as u64]);
        let commits = [
            proof.c_g2,
            proof.c_h2,
            index.c_row,
            index.c_col,
            index.c_val_a,
            index.c_val_b,
            index.c_val_c,
            index.c_row_col,
            proof.c_g2_shift,
        ];
        let evals = [
            proof.ev_g2,
            proof.ev_h2,
            proof.ev_row,
            proof.ev_col,
            proof.ev_val_a,
            proof.ev_val_b,
            proof.ev_val_c,
            proof.ev_row_col,
            s_g2 * proof.ev_g2,
        ];
        let mut c_batch = Comm::zero();
        let mut v_batch = Fr::zero();
        let mut pow = Fr::one();
        for (c, v) in commits.iter().zip(&evals) {
            c_batch += *c * pow;
            v_batch += *v * pow;
            pow *= eta_prime;
        }
        pc::verify(srs, &c_batch, gamma, v_batch, &proof.batched_opening)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn absorb_stmt(t: &mut Blake3Transcript, s: &PStatement) {
    t.absorb(b"il::alpha", &s.alpha);
    t.absorb(b"il::beta", &s.beta);
    t.absorb(b"il::y", &s.y);
    t.absorb(b"il::eta", &s.eta);
}

fn absorb_index_meta(t: &mut Blake3Transcript, idx: &InnerLincheckIndex) {
    t.absorb(b"il::c_row", &idx.c_row);
    t.absorb(b"il::c_col", &idx.c_col);
    t.absorb(b"il::c_val_a", &idx.c_val_a);
    t.absorb(b"il::c_val_b", &idx.c_val_b);
    t.absorb(b"il::c_val_c", &idx.c_val_c);
    t.absorb(b"il::c_row_col", &idx.c_row_col);
    t.absorb_usize(b"il::n", idx.domain_h.size());
    t.absorb_usize(b"il::k", idx.domain_k.size());
}

fn joint_support_size(
    a: &[(usize, usize, Fr)],
    b: &[(usize, usize, Fr)],
    c: &[(usize, usize, Fr)],
) -> usize {
    use std::collections::BTreeSet;
    let mut s: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(i, j, _) in a {
        s.insert((i, j));
    }
    for &(i, j, _) in b {
        s.insert((i, j));
    }
    for &(i, j, _) in c {
        s.insert((i, j));
    }
    s.len()
}

fn build_b_hat(
    alpha: Fr,
    beta: Fr,
    col: &DensePolynomial<Fr>,
    row: &DensePolynomial<Fr>,
    row_col: &DensePolynomial<Fr>,
) -> DensePolynomial<Fr> {
    // αβ − α·col(X) − β·row(X) + row_col(X)
    let mut acc = DensePolynomial::from_coefficients_vec(vec![alpha * beta]);
    acc = &acc + &scale_dense(col, -alpha);
    acc = &acc + &scale_dense(row, -beta);
    acc = &acc + row_col;
    acc
}

fn scale_dense(p: &DensePolynomial<Fr>, c: Fr) -> DensePolynomial<Fr> {
    DensePolynomial::from_coefficients_vec(p.coeffs.iter().map(|&x| x * c).collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    /// Three small matrices + a random `(α, β, η)`; honest `y` computed from
    /// the matrices; roundtrip must verify.
    fn synthetic_instance(
        n: usize,
    ) -> (
        Srs,
        Vec<(usize, usize, Fr)>,
        Vec<(usize, usize, Fr)>,
        Vec<(usize, usize, Fr)>,
        InnerLincheckIndex,
        PStatement,
    ) {
        let rng = &mut test_rng();
        let a: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
        let b: Vec<(usize, usize, Fr)> = (0..n)
            .map(|i| (i, (i + 1) % n, Fr::rand(rng)))
            .collect();
        let c: Vec<(usize, usize, Fr)> = vec![
            (0, 0, Fr::rand(rng)),
            (n / 2, n - 1, Fr::rand(rng)),
        ];

        // SRS must cover deg ≤ |K| − 1. |K| ≤ 4n here, so 4n is safe.
        let srs = pc::setup(4 * n, rng);
        let index = InnerLincheck::index(&srs, &a, &b, &c, n);

        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let eta = Fr::rand(rng);

        let dom = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let lambda_alpha = dom.evaluate_all_lagrange_coefficients(alpha);
        let lambda_beta = dom.evaluate_all_lagrange_coefficients(beta);
        let p_a: Fr = a
            .iter()
            .map(|&(i, j, m)| m * lambda_alpha[i] * lambda_beta[j])
            .sum();
        let p_b: Fr = b
            .iter()
            .map(|&(i, j, m)| m * lambda_alpha[i] * lambda_beta[j])
            .sum();
        let p_c: Fr = c
            .iter()
            .map(|&(i, j, m)| m * lambda_alpha[i] * lambda_beta[j])
            .sum();
        let y = p_a + eta * p_b + eta * eta * p_c;

        let stmt = PStatement { alpha, beta, y, eta };
        (srs, a, b, c, index, stmt)
    }

    #[test]
    fn roundtrip_n4() {
        let (srs, _, _, _, index, stmt) = synthetic_instance(4);
        let mut t_p = Blake3Transcript::new(b"il::test");
        let proof = InnerLincheck::prove(&srs, &index, &stmt, &mut t_p);
        let mut t_v = Blake3Transcript::new(b"il::test");
        assert!(InnerLincheck::verify(&srs, &index, &stmt, &proof, &mut t_v));
    }

    #[test]
    fn roundtrip_n8() {
        let (srs, _, _, _, index, stmt) = synthetic_instance(8);
        let mut t_p = Blake3Transcript::new(b"il::test");
        let proof = InnerLincheck::prove(&srs, &index, &stmt, &mut t_p);
        let mut t_v = Blake3Transcript::new(b"il::test");
        assert!(InnerLincheck::verify(&srs, &index, &stmt, &proof, &mut t_v));
    }

    #[test]
    fn tampered_y_rejected() {
        let (srs, _, _, _, index, mut stmt) = synthetic_instance(4);
        let mut t_p = Blake3Transcript::new(b"il::test");
        let proof = InnerLincheck::prove(&srs, &index, &stmt, &mut t_p);
        stmt.y += Fr::from(1u64);
        let mut t_v = Blake3Transcript::new(b"il::test");
        assert!(!InnerLincheck::verify(&srs, &index, &stmt, &proof, &mut t_v));
    }

    #[test]
    fn tampered_alpha_rejected() {
        let (srs, _, _, _, index, mut stmt) = synthetic_instance(4);
        let mut t_p = Blake3Transcript::new(b"il::test");
        let proof = InnerLincheck::prove(&srs, &index, &stmt, &mut t_p);
        stmt.alpha += Fr::from(1u64);
        let mut t_v = Blake3Transcript::new(b"il::test");
        assert!(!InnerLincheck::verify(&srs, &index, &stmt, &proof, &mut t_v));
    }

    #[test]
    fn tampered_proof_evaluation_rejected() {
        let (srs, _, _, _, index, stmt) = synthetic_instance(4);
        let mut t_p = Blake3Transcript::new(b"il::test");
        let mut proof = InnerLincheck::prove(&srs, &index, &stmt, &mut t_p);
        proof.ev_val_a += Fr::from(1u64);
        let mut t_v = Blake3Transcript::new(b"il::test");
        assert!(!InnerLincheck::verify(&srs, &index, &stmt, &proof, &mut t_v));
    }

    /// Feeding `rok_p`'s emitted PStatement directly into the lincheck must
    /// verify — closes the boundary loop end-to-end.
    #[test]
    fn feeding_rok_p_output_works() {
        use crate::reductions::rok_p::RokP;
        use crate::relations::abc::{leaf_instance, AbcParams};

        let rng = &mut test_rng();
        let n = 4;
        let a: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        let b: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, (i + 1) % n, Fr::from(1u64))).collect();
        let c: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
        let srs = pc::setup(4 * n, rng);
        let abc = AbcParams {
            srs: srs.clone(),
            matrix_a: a.clone(),
            matrix_b: b.clone(),
            matrix_c: c.clone(),
            n,
        };

        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (stmt, wit) = leaf_instance(&abc, alpha, beta);

        let mut t_p = Blake3Transcript::new(b"rokp+il");
        let (p_stmt, _pco, _pco_w, _rok_proof) =
            RokP::reduce(&abc, &stmt, &wit, &mut t_p);

        // Now discharge p_stmt via the lincheck.
        let index = InnerLincheck::index(&srs, &a, &b, &c, n);
        let mut t_pl = Blake3Transcript::new(b"il::test");
        let proof = InnerLincheck::prove(&srs, &index, &p_stmt, &mut t_pl);
        let mut t_vl = Blake3Transcript::new(b"il::test");
        assert!(InnerLincheck::verify(&srs, &index, &p_stmt, &proof, &mut t_vl));
    }

    /// Replacing the shifted-g_2 commitment with junk must be rejected —
    /// pins that the 9th batch slot is load-bearing.
    #[test]
    fn tampered_c_g2_shift_rejected() {
        let (srs, _, _, _, index, stmt) = synthetic_instance(4);
        let mut t_p = Blake3Transcript::new(b"il::test");
        let mut proof = InnerLincheck::prove(&srs, &index, &stmt, &mut t_p);

        let junk = DensePolynomial::from_coefficients_vec(vec![Fr::from(42u64)]);
        proof.c_g2_shift = pc::commit(&srs, &Poly::Dense(junk));

        let mut t_v = Blake3Transcript::new(b"il::test");
        assert!(!InnerLincheck::verify(&srs, &index, &stmt, &proof, &mut t_v));
    }

    /// The g_2 degree-bound regression. Univariate-sumcheck soundness needs
    /// `deg g_2 ≤ |K| − 2`; without enforcement, a prover can pass for an
    /// arbitrary claimed sum `y' = y + Δ` by setting
    /// ```text
    ///   g_2' = g_2 + c·X^{|K|−1},   h_2' = h_2 − c·b̂,   c = (y − y')/|K|
    /// ```
    /// because then `f̃ = y'/|K| + X·g_2' = f + c·v_K`, which agrees with
    /// `f` on K, and `a − b̂·f̃ = (h_2 − c·b̂)·v_K` exactly. This test builds
    /// that forged prover run end-to-end, asserts in-test that the bare
    /// polynomial-identity check accepts it (i.e. the attack is real), and
    /// then asserts the fixed verifier rejects it: the forger cannot commit
    /// `X^{D−(|K|−1)}·g_2'` (degree D exceeds the SRS), so its shifted slot
    /// cannot open to `γ^{D−(|K|−1)}·ev_g2'`.
    #[test]
    fn high_degree_g2_forgery_rejected() {
        let (srs, _, _, _, index, honest_stmt) = synthetic_instance(8);
        let domain_h = &index.domain_h;
        let domain_k = &index.domain_k;
        let k_size = domain_k.size();
        let k_fr = Fr::from(k_size as u64);
        let big_d = srs.powers_g1.len();

        // False claim: y' = y + 1.
        let mut stmt = honest_stmt.clone();
        stmt.y += Fr::one();

        // Rebuild the honest f (whose sum over K is the TRUE y).
        let v_h_alpha = domain_h.evaluate_vanishing_polynomial(stmt.alpha);
        let v_h_beta = domain_h.evaluate_vanishing_polynomial(stmt.beta);
        let v_h_prod = v_h_alpha * v_h_beta;
        let eta_b = stmt.eta;
        let eta_c = stmt.eta * stmt.eta;
        let evals_k = &index.arith.evals_on_k;
        let mut a_evals = vec![Fr::zero(); k_size];
        let mut b_evals = vec![Fr::zero(); k_size];
        for k in 0..k_size {
            a_evals[k] = v_h_prod
                * (evals_k.val_a[k] + eta_b * evals_k.val_b[k] + eta_c * evals_k.val_c[k]);
            b_evals[k] = (stmt.alpha - evals_k.row[k]) * (stmt.beta - evals_k.col[k]);
        }
        let mut b_inv = b_evals.clone();
        batch_inversion(&mut b_inv);
        let f_evals: Vec<Fr> =
            a_evals.iter().zip(&b_inv).map(|(a, bi)| *a * *bi).collect();
        let f_coeffs = domain_k.ifft(&f_evals);
        assert_eq!(f_coeffs[0] * k_fr, honest_stmt.y, "sanity: Σ f = true y");

        // Forge: g_2' = g_2 + c·X^{|K|−1}, h_2' = h_2 − c·b̂, c = (y − y')/|K|.
        let c_top = (honest_stmt.y - stmt.y) * k_fr.inverse().unwrap();
        let g2_true = DensePolynomial::from_coefficients_vec(f_coeffs[1..].to_vec());
        let mut g2_forged_coeffs = g2_true.coeffs.clone();
        g2_forged_coeffs.resize(k_size, Fr::zero());
        g2_forged_coeffs[k_size - 1] += c_top;
        let g2_forged = DensePolynomial::from_coefficients_vec(g2_forged_coeffs);

        let a_poly = DensePolynomial::from_coefficients_vec(domain_k.ifft(&a_evals));
        let b_hat = build_b_hat(
            stmt.alpha,
            stmt.beta,
            &index.arith.col,
            &index.arith.row,
            &index.arith.row_col,
        );
        let f_poly = DensePolynomial::from_coefficients_vec(f_coeffs);
        let bf = &b_hat * &f_poly;
        let numerator = &a_poly - &bf;
        let (h2_true, rem) = numerator.divide_by_vanishing_poly(*domain_k);
        assert!(rem.is_zero());
        let h2_forged = &h2_true + &scale_dense(&b_hat, -c_top);

        // The forger CANNOT commit X^{D−(|K|−1)}·g_2' — that has degree D,
        // beyond the SRS. Best effort: ship the shift of the truncated g_2
        // (= g_2_true) and hope the slot isn't checked.
        let shift = big_d - (k_size - 1);
        let g2_shift_junk = x_shift_dense(&g2_true, shift);

        let c_g2 = pc::commit(&srs, &Poly::Dense(g2_forged.clone()));
        let c_h2 = pc::commit(&srs, &Poly::Dense(h2_forged.clone()));
        let c_g2_shift = pc::commit(&srs, &Poly::Dense(g2_shift_junk.clone()));

        // Forged prover transcript, mirroring `prove` exactly.
        let mut t = Blake3Transcript::new(b"il::forgery");
        absorb_stmt(&mut t, &stmt);
        absorb_index_meta(&mut t, &index);
        t.absorb(b"il::c_g2", &c_g2);
        t.absorb(b"il::c_h2", &c_h2);
        t.absorb(b"il::c_g2_shift", &c_g2_shift);
        let gamma: Fr = t.squeeze_field(b"il::gamma");

        let ev_g2 = g2_forged.evaluate(&gamma);
        let ev_h2 = h2_forged.evaluate(&gamma);
        let ev_row = index.arith.row.evaluate(&gamma);
        let ev_col = index.arith.col.evaluate(&gamma);
        let ev_val_a = index.arith.val_a.evaluate(&gamma);
        let ev_val_b = index.arith.val_b.evaluate(&gamma);
        let ev_val_c = index.arith.val_c.evaluate(&gamma);
        let ev_row_col = index.arith.row_col.evaluate(&gamma);

        // In-test demonstration that the attack is real: the bare polynomial
        // identity accepts the forged (y', g_2', h_2').
        let a_gamma = v_h_prod * (ev_val_a + eta_b * ev_val_b + eta_c * ev_val_c);
        let b_gamma = stmt.alpha * stmt.beta - stmt.alpha * ev_col - stmt.beta * ev_row
            + ev_row_col;
        let v_k_gamma = domain_k.evaluate_vanishing_polynomial(gamma);
        let y_over_k = stmt.y * k_fr.inverse().unwrap();
        assert_eq!(
            a_gamma - b_gamma * (gamma * ev_g2 + y_over_k),
            ev_h2 * v_k_gamma,
            "forged identity must hold — attack construction broken otherwise"
        );

        t.absorb(b"il::ev_g2", &ev_g2);
        t.absorb(b"il::ev_h2", &ev_h2);
        t.absorb(b"il::ev_row", &ev_row);
        t.absorb(b"il::ev_col", &ev_col);
        t.absorb(b"il::ev_val_a", &ev_val_a);
        t.absorb(b"il::ev_val_b", &ev_val_b);
        t.absorb(b"il::ev_val_c", &ev_val_c);
        t.absorb(b"il::ev_row_col", &ev_row_col);
        let eta_prime: Fr = t.squeeze_field(b"il::eta_prime");

        // Honest batched opening of the (forged) committed polynomials.
        let polys = [
            &g2_forged,
            &h2_forged,
            &index.arith.row,
            &index.arith.col,
            &index.arith.val_a,
            &index.arith.val_b,
            &index.arith.val_c,
            &index.arith.row_col,
            &g2_shift_junk,
        ];
        let mut p_batch = DensePolynomial::<Fr>::zero();
        let mut pow = Fr::one();
        for p in polys {
            p_batch = &p_batch + &scale_dense(p, pow);
            pow *= eta_prime;
        }
        let (batched_opening, _) = pc::prove(&srs, &Poly::Dense(p_batch), gamma);

        let forged = InnerLincheckProof {
            c_g2,
            c_h2,
            c_g2_shift,
            ev_g2,
            ev_h2,
            ev_row,
            ev_col,
            ev_val_a,
            ev_val_b,
            ev_val_c,
            ev_row_col,
            batched_opening,
        };

        // The shifted slot's expected value γ^{D−(|K|−1)}·ev_g2' disagrees
        // with what the junk shift commitment opens to, so the batch fails.
        let mut t_v = Blake3Transcript::new(b"il::forgery");
        assert!(!InnerLincheck::verify(&srs, &index, &stmt, &forged, &mut t_v));
    }
}
