use ark_bls12_381::Fr;
use ark_ff::{Field, One, Zero};
use ark_poly::{
    univariate::{DensePolynomial, SparsePolynomial},
    DenseUVPolynomial, EvaluationDomain, Polynomial, Radix2EvaluationDomain,
};

use crate::pc::{self, Comm, Opening, Poly};
use crate::reductions::poly_util::{x_shift, x_shift_dense};
use crate::relations::abc::{AbcParams, AbcStatement, AbcWitness};
use crate::relations::p::PStatement;
use crate::relations::pco::{PcoStatement, PcoWitness};
use crate::transcript::Blake3Transcript;

/// `Π_P : R_{A,B,C} → R_P × R_PCO` (page 27 of the updated paper,
/// generalized to three matrices).
///
/// Introduces `w_X = X·v` for `X ∈ {A, B, C}` as auxiliary witnesses and
/// proves all six halves (each `w_X = X·v` and each `y_X = uᵀ·w_X`) via a
/// single η-weighted univariate sumcheck over `H`. The combined sumcheck
/// value is `y_A + η·y_B + η²·y_C`; the verifier's reconstruction of
/// `G(β)` only depends on the linear combination `μ_A + η·μ_B + η²·μ_C`,
/// so the prover ships a single `y` field instead of three separate `μ`s.
///
/// Output:
/// - `PStatement(α, β, y, η)` where `y = μ_A + η·μ_B + η²·μ_C` and
///   `η` is this reduction's first-round Fiat-Shamir challenge,
/// - `PcoStatement` carrying the batched KZG opening obligation at `β`
///   over all 18 polynomials (9 unshifted + 9 shifted).
///
/// **Polynomial-identity assumption.** The verifier reads `u(β)` and
/// `v(β)` from the rational pairs as `N(β)/(n·T(β))`. This is only equal
/// to the interpolant evaluation if `N(X) = n·T(X)·u(X)` holds as a
/// *polynomial* identity (not just at `h_i ∈ H`). R_{A,B,C}'s own
/// `is_satisfied` is per-vertex; the polynomial identity is established
/// at leaves by `leaf_correctness_pcc` and preserved by Π_{ABC}'s product
/// structure (`N_u^* = N_u^A·T_u^B + γ·N_u^B·T_u^A`, `T_u^* = T_u^A·T_u^B`
/// keeps the identity exact). All those upstream R_PCC promises are
/// discharged by the downstream R_PCO chain, so this reduction is sound
/// only as part of the full pipeline — do not call it standalone on a raw
/// R_{A,B,C} instance.
pub struct RokP;

/// Prover's wire message.
#[derive(Clone)]
pub struct RokPProof {
    // Round 1.
    pub c_w_a: Comm,
    pub c_w_b: Comm,
    pub c_w_c: Comm,
    pub c_w_a_shift: Comm,
    pub c_w_b_shift: Comm,
    pub c_w_c_shift: Comm,
    pub c_n_u_shift: Comm,
    pub c_t_u_shift: Comm,
    pub c_n_v_shift: Comm,
    pub c_t_v_shift: Comm,
    // Round 2.
    pub c_q0: Comm,
    pub c_q1: Comm,
    pub c_q0_shift: Comm,
    pub c_q1_shift: Comm,
    // Evaluations at β (unshifted only; shifted derived as `β^shift · ev`).
    pub ev_n_u: Fr,
    pub ev_t_u: Fr,
    pub ev_n_v: Fr,
    pub ev_t_v: Fr,
    pub ev_w_a: Fr,
    pub ev_w_b: Fr,
    pub ev_w_c: Fr,
    pub ev_q0: Fr,
    pub ev_q1: Fr,
    /// Combined claim `μ_A + η·μ_B + η²·μ_C`; becomes the R_P `y`.
    pub y: Fr,
    /// η'-batched KZG opening at β.
    pub batched_opening: Opening,
}

impl RokP {
    /// Prover side. Returns `(R_P, R_PCO, R_PCO witness, proof)`.
    pub fn reduce(
        params: &AbcParams,
        stmt: &AbcStatement,
        wit: &AbcWitness,
        transcript: &mut Blake3Transcript,
    ) -> (PStatement, PcoStatement, PcoWitness, RokPProof) {
        let srs = &params.srs;
        let n = params.n;
        let big_d = srs.powers_g1.len();
        let dom = Radix2EvaluationDomain::<Fr>::new(n)
            .expect("n must be a power of two");
        let n_fr = Fr::from(n as u64);

        absorb_abc_statement(transcript, stmt);

        // --- Compute w_X = X·v vectors, then IFFT interpolants (DENSE — see comment below).
        // NOTE on the dense/sparse split (the v0.5 speedup): polynomials that come
        // out of IFFT are generically dense in coefficients, and their pairwise
        // products (u·w, Λ·w, M·v below) cost O(n²) under SparsePolynomial::mul
        // but only O(n log n) under DensePolynomial's FFT-based mul (arkworks
        // auto-switches around deg 128). The witness's N/T polynomials are
        // genuinely sparse (3 monomials at a leaf), so we leave those as
        // SparsePolynomial and convert only at the batched-poly accumulator.
        let w_a_vec = matvec(&params.matrix_a, &wit.v, n);
        let w_b_vec = matvec(&params.matrix_b, &wit.v, n);
        let w_c_vec = matvec(&params.matrix_c, &wit.v, n);
        let w_a_poly = ifft_dense(&dom, &w_a_vec);
        let w_b_poly = ifft_dense(&dom, &w_b_vec);
        let w_c_poly = ifft_dense(&dom, &w_c_vec);
        let u_poly = ifft_dense(&dom, &wit.u);
        let v_poly = ifft_dense(&dom, &wit.v);

        // Strict degree bounds.
        let strict_d_n_u = stmt.d_n_u + 1;
        let strict_d_t_u = stmt.d_t_u + 1;
        let strict_d_n_v = stmt.d_n_v + 1;
        let strict_d_t_v = stmt.d_t_v + 1;
        let strict_d_w = n;
        let strict_d_q = n - 1;

        // --- Round 1 shifts and commitments.
        // N/T shifts stay sparse (witness polynomials are sparse).
        let n_u_shift = x_shift(&wit.n_u, big_d - strict_d_n_u);
        let t_u_shift = x_shift(&wit.t_u, big_d - strict_d_t_u);
        let n_v_shift = x_shift(&wit.n_v, big_d - strict_d_n_v);
        let t_v_shift = x_shift(&wit.t_v, big_d - strict_d_t_v);
        // w shifts are dense (IFFT result is dense).
        let w_a_shift = x_shift_dense(&w_a_poly, big_d - strict_d_w);
        let w_b_shift = x_shift_dense(&w_b_poly, big_d - strict_d_w);
        let w_c_shift = x_shift_dense(&w_c_poly, big_d - strict_d_w);

        let c_w_a = pc::commit(srs, &Poly::Dense(w_a_poly.clone()));
        let c_w_b = pc::commit(srs, &Poly::Dense(w_b_poly.clone()));
        let c_w_c = pc::commit(srs, &Poly::Dense(w_c_poly.clone()));
        let c_w_a_shift = pc::commit(srs, &Poly::Dense(w_a_shift.clone()));
        let c_w_b_shift = pc::commit(srs, &Poly::Dense(w_b_shift.clone()));
        let c_w_c_shift = pc::commit(srs, &Poly::Dense(w_c_shift.clone()));
        let c_n_u_shift = pc::commit(srs, &Poly::Sparse(n_u_shift.clone()));
        let c_t_u_shift = pc::commit(srs, &Poly::Sparse(t_u_shift.clone()));
        let c_n_v_shift = pc::commit(srs, &Poly::Sparse(n_v_shift.clone()));
        let c_t_v_shift = pc::commit(srs, &Poly::Sparse(t_v_shift.clone()));

        transcript.absorb(b"rok_p::c_w_a", &c_w_a);
        transcript.absorb(b"rok_p::c_w_b", &c_w_b);
        transcript.absorb(b"rok_p::c_w_c", &c_w_c);
        transcript.absorb(b"rok_p::c_w_a_shift", &c_w_a_shift);
        transcript.absorb(b"rok_p::c_w_b_shift", &c_w_b_shift);
        transcript.absorb(b"rok_p::c_w_c_shift", &c_w_c_shift);
        transcript.absorb(b"rok_p::c_n_u_shift", &c_n_u_shift);
        transcript.absorb(b"rok_p::c_t_u_shift", &c_t_u_shift);
        transcript.absorb(b"rok_p::c_n_v_shift", &c_n_v_shift);
        transcript.absorb(b"rok_p::c_t_v_shift", &c_t_v_shift);

        let alpha: Fr = transcript.squeeze_field(b"rok_p::alpha");
        let eta: Fr = transcript.squeeze_field(b"rok_p::eta");

        // --- Build G(X) and decompose for the sumcheck.
        let lambda_alpha_vec = dom.evaluate_all_lagrange_coefficients(alpha);
        let lambda_alpha_poly = ifft_dense(&dom, &lambda_alpha_vec);

        // Per-matrix column-evaluated polynomials M_X(α, ·).
        let m_a_alpha_poly = ifft_dense(
            &dom,
            &m_evaluated_at_alpha(&params.matrix_a, &lambda_alpha_vec, n),
        );
        let m_b_alpha_poly = ifft_dense(
            &dom,
            &m_evaluated_at_alpha(&params.matrix_b, &lambda_alpha_vec, n),
        );
        let m_c_alpha_poly = ifft_dense(
            &dom,
            &m_evaluated_at_alpha(&params.matrix_c, &lambda_alpha_vec, n),
        );

        // g_X(X)  = u(X)·w_X(X)                       — sum = y_X
        // g'_X(X) = Λ(α,X)·w_X(X) − M_X(α,X)·v(X)     — sum = 0
        // Dense × Dense uses arkworks' FFT-based mul above degree ~128.
        let g_a = &u_poly * &w_a_poly;
        let g_b = &u_poly * &w_b_poly;
        let g_c = &u_poly * &w_c_poly;
        let lam_a = &lambda_alpha_poly * &w_a_poly;
        let lam_b = &lambda_alpha_poly * &w_b_poly;
        let lam_c = &lambda_alpha_poly * &w_c_poly;
        let mv_a = &m_a_alpha_poly * &v_poly;
        let mv_b = &m_b_alpha_poly * &v_poly;
        let mv_c = &m_c_alpha_poly * &v_poly;
        let g_p_a = &lam_a - &mv_a;
        let g_p_b = &lam_b - &mv_b;
        let g_p_c = &lam_c - &mv_c;

        // G(X) = g_A + η·g_B + η²·g_C + η³·g'_A + η⁴·g'_B + η⁵·g'_C
        let eta2 = eta * eta;
        let eta3 = eta2 * eta;
        let eta4 = eta3 * eta;
        let eta5 = eta4 * eta;
        let big_g = {
            let mut acc = g_a;
            acc = &acc + &scale_dense(&g_b, eta);
            acc = &acc + &scale_dense(&g_c, eta2);
            acc = &acc + &scale_dense(&g_p_a, eta3);
            acc = &acc + &scale_dense(&g_p_b, eta4);
            acc = &acc + &scale_dense(&g_p_c, eta5);
            acc
        };

        // Combined sumcheck value y_comb_sum = y_A + η·y_B + η²·y_C.
        let y_comb_sum = stmt.y_a + eta * stmt.y_b + eta2 * stmt.y_c;
        let (q0_poly, q1_poly) = sumcheck_decompose(&big_g, y_comb_sum / n_fr, n);

        let q0_shift = x_shift_dense(&q0_poly, big_d - strict_d_q);
        let q1_shift = x_shift_dense(&q1_poly, big_d - strict_d_q);

        let c_q0 = pc::commit(srs, &Poly::Dense(q0_poly.clone()));
        let c_q1 = pc::commit(srs, &Poly::Dense(q1_poly.clone()));
        let c_q0_shift = pc::commit(srs, &Poly::Dense(q0_shift.clone()));
        let c_q1_shift = pc::commit(srs, &Poly::Dense(q1_shift.clone()));

        transcript.absorb(b"rok_p::c_q0", &c_q0);
        transcript.absorb(b"rok_p::c_q1", &c_q1);
        transcript.absorb(b"rok_p::c_q0_shift", &c_q0_shift);
        transcript.absorb(b"rok_p::c_q1_shift", &c_q1_shift);

        let beta: Fr = transcript.squeeze_field(b"rok_p::beta");

        // --- Evaluations at β (the round-3 prover message).
        let ev_n_u = wit.n_u.evaluate(&beta);
        let ev_t_u = wit.t_u.evaluate(&beta);
        let ev_n_v = wit.n_v.evaluate(&beta);
        let ev_t_v = wit.t_v.evaluate(&beta);
        let ev_w_a = w_a_poly.evaluate(&beta);
        let ev_w_b = w_b_poly.evaluate(&beta);
        let ev_w_c = w_c_poly.evaluate(&beta);
        let ev_q0 = q0_poly.evaluate(&beta);
        let ev_q1 = q1_poly.evaluate(&beta);

        // μ_X = M_X(α, β) = λ(α)ᵀ·X·λ(β).
        let lambda_beta_vec = dom.evaluate_all_lagrange_coefficients(beta);
        let mu_a = mu_at(&params.matrix_a, &lambda_alpha_vec, &lambda_beta_vec);
        let mu_b = mu_at(&params.matrix_b, &lambda_alpha_vec, &lambda_beta_vec);
        let mu_c = mu_at(&params.matrix_c, &lambda_alpha_vec, &lambda_beta_vec);
        // Combined claim shipped to R_P (and used by the verifier in G(β)).
        let y_combined = mu_a + eta * mu_b + eta2 * mu_c;

        // Bind every claim the η'-batch combines BEFORE deriving η'.
        absorb_beta_round_claims(
            transcript,
            ev_n_u, ev_t_u, ev_n_v, ev_t_v,
            ev_w_a, ev_w_b, ev_w_c, ev_q0, ev_q1,
            y_combined,
        );
        let eta_p: Fr = transcript.squeeze_field(b"rok_p::eta_p");

        // --- Batched polynomial, commitment, and value.
        // Mixed sparse (N/T) + dense (w, q) sources — accumulate into a single
        // dense `Vec<Fr>` of length `big_d`, then wrap once at the end.
        let commits = [
            stmt.c_n_u, stmt.c_t_u, stmt.c_n_v, stmt.c_t_v,
            c_w_a, c_w_b, c_w_c,
            c_q0, c_q1,
            c_n_u_shift, c_t_u_shift, c_n_v_shift, c_t_v_shift,
            c_w_a_shift, c_w_b_shift, c_w_c_shift,
            c_q0_shift, c_q1_shift,
        ];
        let evals = batched_evals(
            beta, ev_n_u, ev_t_u, ev_n_v, ev_t_v,
            ev_w_a, ev_w_b, ev_w_c, ev_q0, ev_q1,
            big_d, strict_d_n_u, strict_d_t_u, strict_d_n_v, strict_d_t_v,
            strict_d_w, strict_d_q,
        );

        // Pass 1: scalar accumulators (cheap — just 18 commit/eval multiplies).
        let mut c_batch = Comm::zero();
        let mut v_batch = Fr::zero();
        let mut pow = Fr::one();
        for (c, v) in commits.iter().zip(&evals) {
            c_batch += *c * pow;
            v_batch += *v * pow;
            pow *= eta_p;
        }

        // Pass 2: polynomial coefficient accumulation into a dense buffer.
        let mut p_batch_coeffs: Vec<Fr> = vec![Fr::zero(); big_d];
        let mut pow = Fr::one();
        // Order MUST match `commits`/`evals` above.
        add_sparse_into(&mut p_batch_coeffs, &wit.n_u, pow);    pow *= eta_p;
        add_sparse_into(&mut p_batch_coeffs, &wit.t_u, pow);    pow *= eta_p;
        add_sparse_into(&mut p_batch_coeffs, &wit.n_v, pow);    pow *= eta_p;
        add_sparse_into(&mut p_batch_coeffs, &wit.t_v, pow);    pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &w_a_poly, pow);    pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &w_b_poly, pow);    pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &w_c_poly, pow);    pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &q0_poly, pow);     pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &q1_poly, pow);     pow *= eta_p;
        add_sparse_into(&mut p_batch_coeffs, &n_u_shift, pow);  pow *= eta_p;
        add_sparse_into(&mut p_batch_coeffs, &t_u_shift, pow);  pow *= eta_p;
        add_sparse_into(&mut p_batch_coeffs, &n_v_shift, pow);  pow *= eta_p;
        add_sparse_into(&mut p_batch_coeffs, &t_v_shift, pow);  pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &w_a_shift, pow);   pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &w_b_shift, pow);   pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &w_c_shift, pow);   pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &q0_shift, pow);    pow *= eta_p;
        add_dense_into(&mut p_batch_coeffs, &q1_shift, pow);

        let p_batch_dense = DensePolynomial::from_coefficients_vec(p_batch_coeffs);
        let (batched_opening, opened_value) =
            pc::prove(srs, &Poly::Dense(p_batch_dense.clone()), beta);
        debug_assert_eq!(opened_value, v_batch, "batched opening must match v_batch");

        // PcoWitness stores SparsePolynomial; convert once at the boundary.
        let p_batch_sparse = dense_to_sparse(&p_batch_dense);

        let p_stmt = PStatement { alpha, beta, y: y_combined, eta };
        let pco_stmt = PcoStatement { commitment: c_batch, point: beta, value: v_batch };
        let pco_wit = PcoWitness { polynomial: p_batch_sparse };
        let proof = RokPProof {
            c_w_a, c_w_b, c_w_c,
            c_w_a_shift, c_w_b_shift, c_w_c_shift,
            c_n_u_shift, c_t_u_shift, c_n_v_shift, c_t_v_shift,
            c_q0, c_q1, c_q0_shift, c_q1_shift,
            ev_n_u, ev_t_u, ev_n_v, ev_t_v,
            ev_w_a, ev_w_b, ev_w_c,
            ev_q0, ev_q1,
            y: y_combined,
            batched_opening,
        };
        (p_stmt, pco_stmt, pco_wit, proof)
    }

    /// Verifier side. Returns `None` if any check fails.
    pub fn verify(
        params: &AbcParams,
        stmt: &AbcStatement,
        proof: &RokPProof,
        transcript: &mut Blake3Transcript,
    ) -> Option<(PStatement, PcoStatement)> {
        let srs = &params.srs;
        let n = params.n;
        let big_d = srs.powers_g1.len();
        let n_fr = Fr::from(n as u64);

        absorb_abc_statement(transcript, stmt);

        transcript.absorb(b"rok_p::c_w_a", &proof.c_w_a);
        transcript.absorb(b"rok_p::c_w_b", &proof.c_w_b);
        transcript.absorb(b"rok_p::c_w_c", &proof.c_w_c);
        transcript.absorb(b"rok_p::c_w_a_shift", &proof.c_w_a_shift);
        transcript.absorb(b"rok_p::c_w_b_shift", &proof.c_w_b_shift);
        transcript.absorb(b"rok_p::c_w_c_shift", &proof.c_w_c_shift);
        transcript.absorb(b"rok_p::c_n_u_shift", &proof.c_n_u_shift);
        transcript.absorb(b"rok_p::c_t_u_shift", &proof.c_t_u_shift);
        transcript.absorb(b"rok_p::c_n_v_shift", &proof.c_n_v_shift);
        transcript.absorb(b"rok_p::c_t_v_shift", &proof.c_t_v_shift);

        let alpha: Fr = transcript.squeeze_field(b"rok_p::alpha");
        let eta: Fr = transcript.squeeze_field(b"rok_p::eta");

        transcript.absorb(b"rok_p::c_q0", &proof.c_q0);
        transcript.absorb(b"rok_p::c_q1", &proof.c_q1);
        transcript.absorb(b"rok_p::c_q0_shift", &proof.c_q0_shift);
        transcript.absorb(b"rok_p::c_q1_shift", &proof.c_q1_shift);

        let beta: Fr = transcript.squeeze_field(b"rok_p::beta");

        // Mirror the prover: bind the claimed evaluations and y before η'.
        absorb_beta_round_claims(
            transcript,
            proof.ev_n_u, proof.ev_t_u, proof.ev_n_v, proof.ev_t_v,
            proof.ev_w_a, proof.ev_w_b, proof.ev_w_c, proof.ev_q0, proof.ev_q1,
            proof.y,
        );
        let eta_p: Fr = transcript.squeeze_field(b"rok_p::eta_p");

        // Edge cases.
        let t_u_n = n_fr * proof.ev_t_u;
        let t_v_n = n_fr * proof.ev_t_v;
        let t_u_n_inv = t_u_n.inverse()?;
        let t_v_n_inv = t_v_n.inverse()?;
        let beta_minus_alpha_inv = (beta - alpha).inverse()?;
        let n_inv = n_fr.inverse()?;

        let u_at_beta = proof.ev_n_u * t_u_n_inv;
        let v_at_beta = proof.ev_n_v * t_v_n_inv;

        let beta_n = pow_usize(beta, n);
        let alpha_n = pow_usize(alpha, n);
        let lambda_alpha_beta =
            ((beta_n - Fr::one()) * alpha - (alpha_n - Fr::one()) * beta)
                * (n_inv * beta_minus_alpha_inv);

        // G(β) = [u(β) + η³·Λ(α,β)]·ev_W − v(β)·η³·proof.y
        // where ev_W = ev_w_a + η·ev_w_b + η²·ev_w_c.
        let eta2 = eta * eta;
        let eta3 = eta2 * eta;
        let ev_w = proof.ev_w_a + eta * proof.ev_w_b + eta2 * proof.ev_w_c;
        let g_at_beta =
            (u_at_beta + eta3 * lambda_alpha_beta) * ev_w
                - v_at_beta * eta3 * proof.y;

        // Sumcheck identity at β: G(β) ?= (y_A + η·y_B + η²·y_C)/n + β·q0(β) + (β^n − 1)·q1(β).
        let y_combined_sum = stmt.y_a + eta * stmt.y_b + eta2 * stmt.y_c;
        let rhs = y_combined_sum * n_inv
            + beta * proof.ev_q0
            + (beta_n - Fr::one()) * proof.ev_q1;
        if g_at_beta != rhs {
            return None;
        }

        // Reconstruct batched commitment and value.
        let strict_d_n_u = stmt.d_n_u + 1;
        let strict_d_t_u = stmt.d_t_u + 1;
        let strict_d_n_v = stmt.d_n_v + 1;
        let strict_d_t_v = stmt.d_t_v + 1;
        let strict_d_w = n;
        let strict_d_q = n - 1;

        let evals = batched_evals(
            beta, proof.ev_n_u, proof.ev_t_u, proof.ev_n_v, proof.ev_t_v,
            proof.ev_w_a, proof.ev_w_b, proof.ev_w_c, proof.ev_q0, proof.ev_q1,
            big_d, strict_d_n_u, strict_d_t_u, strict_d_n_v, strict_d_t_v,
            strict_d_w, strict_d_q,
        );
        let commits = [
            stmt.c_n_u, stmt.c_t_u, stmt.c_n_v, stmt.c_t_v,
            proof.c_w_a, proof.c_w_b, proof.c_w_c,
            proof.c_q0, proof.c_q1,
            proof.c_n_u_shift, proof.c_t_u_shift, proof.c_n_v_shift, proof.c_t_v_shift,
            proof.c_w_a_shift, proof.c_w_b_shift, proof.c_w_c_shift,
            proof.c_q0_shift, proof.c_q1_shift,
        ];

        let mut c_batch = Comm::zero();
        let mut v_batch = Fr::zero();
        let mut pow = Fr::one();
        for (c, v) in commits.iter().zip(&evals) {
            c_batch += *c * pow;
            v_batch += *v * pow;
            pow *= eta_p;
        }

        if !pc::verify(srs, &c_batch, beta, v_batch, &proof.batched_opening) {
            return None;
        }

        let p_stmt = PStatement { alpha, beta, y: proof.y, eta };
        let pco_stmt = PcoStatement { commitment: c_batch, point: beta, value: v_batch };
        Some((p_stmt, pco_stmt))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn matvec(matrix: &[(usize, usize, Fr)], v: &[Fr], n: usize) -> Vec<Fr> {
    let mut w = vec![Fr::zero(); n];
    for &(i, j, m_ij) in matrix {
        w[i] += m_ij * v[j];
    }
    w
}

/// Column-evaluated matrix polynomial: `m_α[j] = Σ_i M_{ij} · L_i(α)`.
fn m_evaluated_at_alpha(
    matrix: &[(usize, usize, Fr)],
    lambda_alpha: &[Fr],
    n: usize,
) -> Vec<Fr> {
    let mut out = vec![Fr::zero(); n];
    for &(i, j, m_ij) in matrix {
        out[j] += m_ij * lambda_alpha[i];
    }
    out
}

/// `μ = λ(α)ᵀ·M·λ(β) = Σ_{(i,j,m_ij) ∈ M} m_ij · L_i(α) · L_j(β)`.
fn mu_at(
    matrix: &[(usize, usize, Fr)],
    lambda_alpha: &[Fr],
    lambda_beta: &[Fr],
) -> Fr {
    matrix
        .iter()
        .map(|&(i, j, m_ij)| m_ij * lambda_alpha[i] * lambda_beta[j])
        .sum()
}

fn ifft_dense(dom: &Radix2EvaluationDomain<Fr>, evals: &[Fr]) -> DensePolynomial<Fr> {
    DensePolynomial::from_coefficients_vec(dom.ifft(evals))
}

fn scale_dense(p: &DensePolynomial<Fr>, c: Fr) -> DensePolynomial<Fr> {
    DensePolynomial::from_coefficients_vec(p.coeffs.iter().map(|&x| x * c).collect())
}

fn dense_to_sparse(p: &DensePolynomial<Fr>) -> SparsePolynomial<Fr> {
    SparsePolynomial::from_coefficients_vec(
        p.coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, &c)| (i, c))
            .collect(),
    )
}

/// Decompose `G(X) = y_combined/n + X·q_0(X) + (X^n − 1)·q_1(X)`,
/// returning `(q_0, q_1)`. Both have degree ≤ n − 2.
fn sumcheck_decompose(
    big_g: &DensePolynomial<Fr>,
    y_over_n: Fr,
    n: usize,
) -> (DensePolynomial<Fr>, DensePolynomial<Fr>) {
    let mut c = big_g.coeffs.clone();
    if c.len() < 2 * n - 1 {
        c.resize(2 * n - 1, Fr::zero());
    }

    let mut r = vec![Fr::zero(); n];
    for k in 0..(n - 1) {
        r[k] = c[k] + c[k + n];
    }
    r[n - 1] = c[n - 1];
    debug_assert_eq!(r[0], y_over_n, "sumcheck identity must hold");

    let q1: Vec<Fr> = (0..(n - 1)).map(|j| c[j + n]).collect();
    let q0: Vec<Fr> = (0..(n - 1)).map(|j| r[j + 1]).collect();

    (
        DensePolynomial::from_coefficients_vec(q0),
        DensePolynomial::from_coefficients_vec(q1),
    )
}

fn pow_usize(x: Fr, k: usize) -> Fr {
    x.pow([k as u64])
}

/// Accumulate `scale · p(X)` into the coefficient buffer `acc`.
fn add_sparse_into(acc: &mut [Fr], p: &SparsePolynomial<Fr>, scale: Fr) {
    for &(i, c) in p.iter() {
        acc[i] += scale * c;
    }
}

fn add_dense_into(acc: &mut [Fr], p: &DensePolynomial<Fr>, scale: Fr) {
    for (i, &c) in p.coeffs.iter().enumerate() {
        acc[i] += scale * c;
    }
}

#[allow(clippy::too_many_arguments)]
fn batched_evals(
    beta: Fr,
    ev_n_u: Fr, ev_t_u: Fr, ev_n_v: Fr, ev_t_v: Fr,
    ev_w_a: Fr, ev_w_b: Fr, ev_w_c: Fr,
    ev_q0: Fr, ev_q1: Fr,
    big_d: usize,
    strict_d_n_u: usize, strict_d_t_u: usize,
    strict_d_n_v: usize, strict_d_t_v: usize,
    strict_d_w: usize, strict_d_q: usize,
) -> [Fr; 18] {
    let s_n_u = pow_usize(beta, big_d - strict_d_n_u);
    let s_t_u = pow_usize(beta, big_d - strict_d_t_u);
    let s_n_v = pow_usize(beta, big_d - strict_d_n_v);
    let s_t_v = pow_usize(beta, big_d - strict_d_t_v);
    let s_w = pow_usize(beta, big_d - strict_d_w);
    let s_q = pow_usize(beta, big_d - strict_d_q);
    [
        // unshifted (k = 0..8)
        ev_n_u, ev_t_u, ev_n_v, ev_t_v,
        ev_w_a, ev_w_b, ev_w_c,
        ev_q0, ev_q1,
        // shifted (k = 9..17)
        s_n_u * ev_n_u, s_t_u * ev_t_u, s_n_v * ev_n_v, s_t_v * ev_t_v,
        s_w * ev_w_a, s_w * ev_w_b, s_w * ev_w_c,
        s_q * ev_q0, s_q * ev_q1,
    ]
}

/// Absorb the β-round claims (nine evaluations plus the combined μ claim `y`)
/// before squeezing the batching challenge η'. η' must be derived after every
/// value it linearly combines is fixed: if the prover knows η' first, it can
/// move claims within the kernel of the η'-weighting — voiding per-slot
/// binding and, with it, the X-shift degree enforcement (see
/// `tests::kernel_forgery_on_y_rejected`).
#[allow(clippy::too_many_arguments)]
fn absorb_beta_round_claims(
    t: &mut Blake3Transcript,
    ev_n_u: Fr, ev_t_u: Fr, ev_n_v: Fr, ev_t_v: Fr,
    ev_w_a: Fr, ev_w_b: Fr, ev_w_c: Fr,
    ev_q0: Fr, ev_q1: Fr,
    y: Fr,
) {
    t.absorb(b"rok_p::ev_n_u", &ev_n_u);
    t.absorb(b"rok_p::ev_t_u", &ev_t_u);
    t.absorb(b"rok_p::ev_n_v", &ev_n_v);
    t.absorb(b"rok_p::ev_t_v", &ev_t_v);
    t.absorb(b"rok_p::ev_w_a", &ev_w_a);
    t.absorb(b"rok_p::ev_w_b", &ev_w_b);
    t.absorb(b"rok_p::ev_w_c", &ev_w_c);
    t.absorb(b"rok_p::ev_q0", &ev_q0);
    t.absorb(b"rok_p::ev_q1", &ev_q1);
    t.absorb(b"rok_p::y", &y);
}

fn absorb_abc_statement(t: &mut Blake3Transcript, s: &AbcStatement) {
    t.absorb(b"rok_p::abc_c_n_u", &s.c_n_u);
    t.absorb(b"rok_p::abc_c_t_u", &s.c_t_u);
    t.absorb(b"rok_p::abc_c_n_v", &s.c_n_v);
    t.absorb(b"rok_p::abc_c_t_v", &s.c_t_v);
    t.absorb_usize(b"rok_p::abc_d_n_u", s.d_n_u);
    t.absorb_usize(b"rok_p::abc_d_t_u", s.d_t_u);
    t.absorb_usize(b"rok_p::abc_d_n_v", s.d_n_v);
    t.absorb_usize(b"rok_p::abc_d_t_v", s.d_t_v);
    t.absorb(b"rok_p::abc_y_a", &s.y_a);
    t.absorb(b"rok_p::abc_y_b", &s.y_b);
    t.absorb(b"rok_p::abc_y_c", &s.y_c);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Relation;
    use crate::relations::abc::leaf_instance;
    use crate::relations::pco::{PcoParams, PcoRelation};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    fn identity_leaf(n: usize) -> (AbcParams, AbcStatement, AbcWitness) {
        let rng = &mut test_rng();
        let a: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        let b: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, (i + 1) % n, Fr::from(1u64))).collect();
        let c: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
        let srs = pc::setup(2 * n + 3, rng);
        let params = AbcParams { srs, matrix_a: a, matrix_b: b, matrix_c: c, n };
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (s, w) = leaf_instance(&params, alpha, beta);
        (params, s, w)
    }

    #[test]
    fn roundtrip_n4() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (p_p, pco_p, pco_wit, proof) =
            RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        let (p_v, pco_v) =
            RokP::verify(&params, &stmt, &proof, &mut t_v).expect("must verify");

        assert_eq!(p_p, p_v);
        assert_eq!(pco_p, pco_v);

        let pco_params = PcoParams { srs: params.srs.clone() };
        assert!(PcoRelation::is_satisfied(&pco_params, &pco_p, &pco_wit));
    }

    #[test]
    fn roundtrip_n8() {
        let (params, stmt, wit) = identity_leaf(8);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (p_p, pco_p, pco_wit, proof) =
            RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        let (p_v, pco_v) =
            RokP::verify(&params, &stmt, &proof, &mut t_v).expect("must verify");

        assert_eq!(p_p, p_v);
        assert_eq!(pco_p, pco_v);

        let pco_params = PcoParams { srs: params.srs.clone() };
        assert!(PcoRelation::is_satisfied(&pco_params, &pco_p, &pco_wit));
    }

    /// The emitted `PStatement.y` matches the explicit combined matrix
    /// evaluation `P_A(α,β) + η·P_B(α,β) + η²·P_C(α,β)` computed from
    /// the matrices and the squeezed `(α, β, η)`.
    #[test]
    fn emitted_p_y_matches_combined_matrix_eval() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (p_stmt, _, _, _) = RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let dom = Radix2EvaluationDomain::<Fr>::new(params.n).unwrap();
        let lambda_alpha = dom.evaluate_all_lagrange_coefficients(p_stmt.alpha);
        let lambda_beta = dom.evaluate_all_lagrange_coefficients(p_stmt.beta);
        let p_a = mu_at(&params.matrix_a, &lambda_alpha, &lambda_beta);
        let p_b = mu_at(&params.matrix_b, &lambda_alpha, &lambda_beta);
        let p_c = mu_at(&params.matrix_c, &lambda_alpha, &lambda_beta);
        let expected = p_a + p_stmt.eta * p_b + p_stmt.eta * p_stmt.eta * p_c;
        assert_eq!(p_stmt.y, expected);
    }

    #[test]
    fn tampered_y_a_in_stmt_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (_, _, _, proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let mut bad_stmt = stmt.clone();
        bad_stmt.y_a += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        assert!(RokP::verify(&params, &bad_stmt, &proof, &mut t_v).is_none());
    }

    #[test]
    fn tampered_y_b_in_stmt_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (_, _, _, proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let mut bad_stmt = stmt.clone();
        bad_stmt.y_b += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        assert!(RokP::verify(&params, &bad_stmt, &proof, &mut t_v).is_none());
    }

    #[test]
    fn tampered_proof_y_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (_, _, _, mut proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);
        proof.y += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        assert!(RokP::verify(&params, &stmt, &proof, &mut t_v).is_none());
    }

    #[test]
    fn tampered_ev_w_a_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (_, _, _, mut proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);
        proof.ev_w_a += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        assert!(RokP::verify(&params, &stmt, &proof, &mut t_v).is_none());
    }

    #[test]
    fn transcript_mismatch_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::label_a");
        let (_, _, _, proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let mut t_v = Blake3Transcript::new(b"rok_p::label_b");
        assert!(RokP::verify(&params, &stmt, &proof, &mut t_v).is_none());
    }

    /// Uncoordinated tampering of the (ev_q0, ev_q1) pair must be rejected.
    #[test]
    fn tampered_ev_pair_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (_, _, _, mut proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);
        proof.ev_q0 += Fr::from(3u64);
        proof.ev_q1 -= Fr::from(5u64);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        assert!(RokP::verify(&params, &stmt, &proof, &mut t_v).is_none());
    }

    /// The η'-ordering regression. Before the fix, η' was squeezed *before*
    /// the claimed evaluations were absorbed, so a prover knowing η' could
    /// claim a false `y` (the R_P output!) and shift `(ev_q0, ev_q1)` inside
    /// the kernel of the η'-weighting: the sumcheck identity re-balances,
    /// while the batched commitment and value stay bit-identical — so the
    /// honest opening still verifies.
    ///
    /// Part (a) proves the forgery is real: with the pre-fix η' (replayed
    /// from the commitment-only transcript) the forged claims satisfy both
    /// the sumcheck identity and the batched value, i.e. the pre-fix
    /// verifier accepts them with the honest opening. Part (b) asserts the
    /// fixed verifier rejects the same forgery, because η' now depends on
    /// the claims themselves.
    #[test]
    fn kernel_forgery_on_y_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let n = params.n;
        let n_fr = Fr::from(n as u64);
        let big_d = params.srs.powers_g1.len();

        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (_, _, _, proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);

        // --- Attacker replay: challenges up to β depend only on statement
        // and commitments; the PRE-FIX η' followed β with no ev absorbs.
        let mut t = Blake3Transcript::new(b"rok_p::test");
        absorb_abc_statement(&mut t, &stmt);
        t.absorb(b"rok_p::c_w_a", &proof.c_w_a);
        t.absorb(b"rok_p::c_w_b", &proof.c_w_b);
        t.absorb(b"rok_p::c_w_c", &proof.c_w_c);
        t.absorb(b"rok_p::c_w_a_shift", &proof.c_w_a_shift);
        t.absorb(b"rok_p::c_w_b_shift", &proof.c_w_b_shift);
        t.absorb(b"rok_p::c_w_c_shift", &proof.c_w_c_shift);
        t.absorb(b"rok_p::c_n_u_shift", &proof.c_n_u_shift);
        t.absorb(b"rok_p::c_t_u_shift", &proof.c_t_u_shift);
        t.absorb(b"rok_p::c_n_v_shift", &proof.c_n_v_shift);
        t.absorb(b"rok_p::c_t_v_shift", &proof.c_t_v_shift);
        let _alpha: Fr = t.squeeze_field(b"rok_p::alpha");
        let eta: Fr = t.squeeze_field(b"rok_p::eta");
        t.absorb(b"rok_p::c_q0", &proof.c_q0);
        t.absorb(b"rok_p::c_q1", &proof.c_q1);
        t.absorb(b"rok_p::c_q0_shift", &proof.c_q0_shift);
        t.absorb(b"rok_p::c_q1_shift", &proof.c_q1_shift);
        let beta: Fr = t.squeeze_field(b"rok_p::beta");
        let eta_p_prefix: Fr = t.squeeze_field(b"rok_p::eta_p");

        // --- Solve the 2×2 system for Δy = 1:
        //   sumcheck row: β·δ0 + (β^n − 1)·δ1 = −v(β)·η³
        //   batch row   : (η'^7 + s_q·η'^16)·δ0 + (η'^8 + s_q·η'^17)·δ1 = 0
        let eta2 = eta * eta;
        let eta3 = eta2 * eta;
        let v_at_beta = proof.ev_n_v * (n_fr * proof.ev_t_v).inverse().unwrap();
        let beta_n = pow_usize(beta, n);
        let s_q = pow_usize(beta, big_d - (n - 1)); // strict_d_q = n − 1
        let a_coef = eta_p_prefix.pow([7u64]) + s_q * eta_p_prefix.pow([16u64]);
        let b_coef = eta_p_prefix.pow([8u64]) + s_q * eta_p_prefix.pow([17u64]);
        let ratio = b_coef * a_coef.inverse().unwrap();
        let denom = (beta_n - Fr::one()) - beta * ratio;
        let delta1 = -v_at_beta * eta3 * denom.inverse().unwrap();
        let delta0 = -ratio * delta1;

        let mut forged = proof.clone();
        forged.y += Fr::one();
        forged.ev_q0 += delta0;
        forged.ev_q1 += delta1;

        // --- (a) The pre-fix verifier accepts this forgery.
        // (a1) The forged claims satisfy the sumcheck identity at β.
        {
            let u_at_beta = forged.ev_n_u * (n_fr * forged.ev_t_u).inverse().unwrap();
            let alpha_n = pow_usize(_alpha, n);
            let n_inv = n_fr.inverse().unwrap();
            let lambda_alpha_beta = ((beta_n - Fr::one()) * _alpha
                - (alpha_n - Fr::one()) * beta)
                * (n_inv * (beta - _alpha).inverse().unwrap());
            let ev_w = forged.ev_w_a + eta * forged.ev_w_b + eta2 * forged.ev_w_c;
            let g_at_beta = (u_at_beta + eta3 * lambda_alpha_beta) * ev_w
                - v_at_beta * eta3 * forged.y;
            let y_sum = stmt.y_a + eta * stmt.y_b + eta2 * stmt.y_c;
            let rhs = y_sum * n_inv
                + beta * forged.ev_q0
                + (beta_n - Fr::one()) * forged.ev_q1;
            assert_eq!(g_at_beta, rhs, "forged sumcheck identity must hold");
        }
        // (a2) Under the pre-fix η', the batched value is unchanged (the
        // perturbation lies in the kernel), and the commitments are untouched
        // — so the honest batched opening verifies the forged claims too.
        {
            let strict = (stmt.d_n_u + 1, stmt.d_t_u + 1, stmt.d_n_v + 1, stmt.d_t_v + 1);
            let evals_honest = batched_evals(
                beta, proof.ev_n_u, proof.ev_t_u, proof.ev_n_v, proof.ev_t_v,
                proof.ev_w_a, proof.ev_w_b, proof.ev_w_c, proof.ev_q0, proof.ev_q1,
                big_d, strict.0, strict.1, strict.2, strict.3, n, n - 1,
            );
            let evals_forged = batched_evals(
                beta, forged.ev_n_u, forged.ev_t_u, forged.ev_n_v, forged.ev_t_v,
                forged.ev_w_a, forged.ev_w_b, forged.ev_w_c, forged.ev_q0, forged.ev_q1,
                big_d, strict.0, strict.1, strict.2, strict.3, n, n - 1,
            );
            let mut v_honest = Fr::zero();
            let mut v_forged = Fr::zero();
            let mut pow = Fr::one();
            for (h, f) in evals_honest.iter().zip(&evals_forged) {
                v_honest += *h * pow;
                v_forged += *f * pow;
                pow *= eta_p_prefix;
            }
            assert_eq!(
                v_honest, v_forged,
                "forged claims must batch to the same value under the pre-fix η'"
            );
        }

        // --- (b) The fixed verifier rejects it.
        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        assert!(RokP::verify(&params, &stmt, &forged, &mut t_v).is_none());
    }
}
