use ark_bls12_381::Fr;
use ark_ff::{Field, One, Zero};
use ark_poly::{
    univariate::SparsePolynomial, EvaluationDomain, Polynomial, Radix2EvaluationDomain,
};

use crate::pc::{self, Comm, Opening, Poly};
use crate::reductions::poly_util::x_shift;
use crate::relations::m::{MParams, MStatement, MWitness};
use crate::relations::p::PStatement;
use crate::relations::pco::{PcoStatement, PcoWitness};
use crate::transcript::Blake3Transcript;

/// `Π_P : R_M → R_P × R_PCO` (page 27 of the updated paper).
///
/// Introduces `w = M·v` as an auxiliary witness and proves both halves
/// (`w = M·v` and `y = uᵀ·w`) via a single batched univariate sumcheck
/// over `H`. The output is one `R_P` instance `(α, β, μ = M(α, β))`
/// — discharged by direct evaluation of the matrix-encoded polynomial —
/// and one `R_PCO` instance carrying the batched KZG opening obligation
/// at `β` for all 14 polynomials touched by the protocol.
///
/// **Polynomial-identity assumption.** The reduction reads `u(β)` and
/// `v(β)` from the rational pairs as `N(β)/(n·T(β))`. This is only
/// equal to the interpolant evaluation if `N(X) = n·T(X)·u(X)` holds
/// as a *polynomial* identity (not just at `h_i ∈ H`). R_M's own
/// `is_satisfied` is per-vertex; the polynomial identity is established
/// at leaves by `leaf_correctness_pcc` (the Q_u, Q_v constraints there
/// imply it) and preserved by Π_M's product structure
/// (`N_u^* = N_u^A·T_u^B + γ·N_u^B·T_u^A`, `T_u^* = T_u^A·T_u^B` keeps
/// the identity exact). All those upstream R_PCC promises are
/// discharged by the downstream R_PCO chain, so this reduction is sound
/// only as part of the full pipeline — do not call it standalone on a
/// raw R_M instance.
pub struct RokP;

/// Prover's wire message.
#[derive(Clone)]
pub struct RokPProof {
    // Round 1.
    pub c_w: Comm,
    pub c_w_shift: Comm,
    pub c_n_u_shift: Comm,
    pub c_t_u_shift: Comm,
    pub c_n_v_shift: Comm,
    pub c_t_v_shift: Comm,
    // Round 2.
    pub c_q0: Comm,
    pub c_q1: Comm,
    pub c_q0_shift: Comm,
    pub c_q1_shift: Comm,
    // Evaluations at β (unshifted only; shifted entries derived as `β^shift · ev`).
    pub ev_n_u: Fr,
    pub ev_t_u: Fr,
    pub ev_n_v: Fr,
    pub ev_t_v: Fr,
    pub ev_w: Fr,
    pub ev_q0: Fr,
    pub ev_q1: Fr,
    // Prover's claim for `M(α, β)`; becomes the R_P `y`.
    pub mu: Fr,
    // η'-batched KZG opening at β.
    pub batched_opening: Opening,
}

impl RokP {
    /// Prover side. Returns `(R_P, R_PCO, R_PCO witness, proof)`.
    pub fn reduce(
        params: &MParams,
        stmt: &MStatement,
        wit: &MWitness,
        transcript: &mut Blake3Transcript,
    ) -> (PStatement, PcoStatement, PcoWitness, RokPProof) {
        let srs = &params.srs;
        let n = params.n;
        let big_d = srs.powers_g1.len();
        let dom = Radix2EvaluationDomain::<Fr>::new(n)
            .expect("n must be a power of two");
        let n_fr = Fr::from(n as u64);

        absorb_m_statement(transcript, stmt);

        // --- Compute w = M·v as a vector, then its IFFT interpolant.
        let w_vec = matvec(&params.matrix, &wit.v, n);
        let w_poly = ifft_sparse(&dom, &w_vec);
        let u_poly = ifft_sparse(&dom, &wit.u);
        let v_poly = ifft_sparse(&dom, &wit.v);

        // Strict degree bounds (M's `d_*` are inclusive).
        let strict_d_n_u = stmt.d_n_u + 1;
        let strict_d_t_u = stmt.d_t_u + 1;
        let strict_d_n_v = stmt.d_n_v + 1;
        let strict_d_t_v = stmt.d_t_v + 1;
        // u, v, w live on H of size n, so deg < n, strict bound n.
        let strict_d_w = n;
        // q_0, q_1 both have strict bound n-1 (the tightened bound).
        let strict_d_q = n - 1;

        // --- Round 1 commitments and shifts.
        let n_u_shift = x_shift(&wit.n_u, big_d - strict_d_n_u);
        let t_u_shift = x_shift(&wit.t_u, big_d - strict_d_t_u);
        let n_v_shift = x_shift(&wit.n_v, big_d - strict_d_n_v);
        let t_v_shift = x_shift(&wit.t_v, big_d - strict_d_t_v);
        let w_shift = x_shift(&w_poly, big_d - strict_d_w);

        let c_w = pc::commit(srs, &Poly::Sparse(w_poly.clone()));
        let c_w_shift = pc::commit(srs, &Poly::Sparse(w_shift.clone()));
        let c_n_u_shift = pc::commit(srs, &Poly::Sparse(n_u_shift.clone()));
        let c_t_u_shift = pc::commit(srs, &Poly::Sparse(t_u_shift.clone()));
        let c_n_v_shift = pc::commit(srs, &Poly::Sparse(n_v_shift.clone()));
        let c_t_v_shift = pc::commit(srs, &Poly::Sparse(t_v_shift.clone()));

        transcript.absorb(b"rok_p::c_w", &c_w);
        transcript.absorb(b"rok_p::c_w_shift", &c_w_shift);
        transcript.absorb(b"rok_p::c_n_u_shift", &c_n_u_shift);
        transcript.absorb(b"rok_p::c_t_u_shift", &c_t_u_shift);
        transcript.absorb(b"rok_p::c_n_v_shift", &c_n_v_shift);
        transcript.absorb(b"rok_p::c_t_v_shift", &c_t_v_shift);

        let alpha: Fr = transcript.squeeze_field(b"rok_p::alpha");
        let eta: Fr = transcript.squeeze_field(b"rok_p::eta");

        // --- Build g''(X) and decompose it for the sumcheck.
        let lambda_alpha_vec = dom.evaluate_all_lagrange_coefficients(alpha);
        let lambda_alpha_poly = ifft_sparse(&dom, &lambda_alpha_vec);
        let m_alpha_vec = m_evaluated_at_alpha(&params.matrix, &lambda_alpha_vec, n);
        let m_alpha_poly = ifft_sparse(&dom, &m_alpha_vec);

        let uw = u_poly.mul(&w_poly);
        let lambda_w = lambda_alpha_poly.mul(&w_poly);
        let m_alpha_v = m_alpha_poly.mul(&v_poly);
        // g(X) = Λ(α,X)·w(X) − M(α,X)·v(X)
        let g_poly = &lambda_w + &(&m_alpha_v * -Fr::one());
        // g''(X) = u·w + η·g
        let g_pp = &uw + &(&g_poly * eta);

        let (q0_poly, q1_poly) = sumcheck_decompose(&g_pp, stmt.value / n_fr, n);

        let q0_shift = x_shift(&q0_poly, big_d - strict_d_q);
        let q1_shift = x_shift(&q1_poly, big_d - strict_d_q);

        let c_q0 = pc::commit(srs, &Poly::Sparse(q0_poly.clone()));
        let c_q1 = pc::commit(srs, &Poly::Sparse(q1_poly.clone()));
        let c_q0_shift = pc::commit(srs, &Poly::Sparse(q0_shift.clone()));
        let c_q1_shift = pc::commit(srs, &Poly::Sparse(q1_shift.clone()));

        transcript.absorb(b"rok_p::c_q0", &c_q0);
        transcript.absorb(b"rok_p::c_q1", &c_q1);
        transcript.absorb(b"rok_p::c_q0_shift", &c_q0_shift);
        transcript.absorb(b"rok_p::c_q1_shift", &c_q1_shift);

        let beta: Fr = transcript.squeeze_field(b"rok_p::beta");
        let eta_p: Fr = transcript.squeeze_field(b"rok_p::eta_p");

        // --- Evaluations at β.
        let ev_n_u = wit.n_u.evaluate(&beta);
        let ev_t_u = wit.t_u.evaluate(&beta);
        let ev_n_v = wit.n_v.evaluate(&beta);
        let ev_t_v = wit.t_v.evaluate(&beta);
        let ev_w = w_poly.evaluate(&beta);
        let ev_q0 = q0_poly.evaluate(&beta);
        let ev_q1 = q1_poly.evaluate(&beta);

        // μ = M(α, β) = λ(α)ᵀ·M·λ(β).
        let lambda_beta_vec = dom.evaluate_all_lagrange_coefficients(beta);
        let mu: Fr = params
            .matrix
            .iter()
            .map(|&(i, j, m_ij)| m_ij * lambda_alpha_vec[i] * lambda_beta_vec[j])
            .sum();

        // --- Batched polynomial and the corresponding commitment / value.
        let polys = [
            &wit.n_u, &wit.t_u, &wit.n_v, &wit.t_v,
            &w_poly, &q0_poly, &q1_poly,
            &n_u_shift, &t_u_shift, &n_v_shift, &t_v_shift,
            &w_shift, &q0_shift, &q1_shift,
        ];
        let commits = [
            stmt.c_n_u, stmt.c_t_u, stmt.c_n_v, stmt.c_t_v,
            c_w, c_q0, c_q1,
            c_n_u_shift, c_t_u_shift, c_n_v_shift, c_t_v_shift,
            c_w_shift, c_q0_shift, c_q1_shift,
        ];
        let evals = batched_evals(
            beta, ev_n_u, ev_t_u, ev_n_v, ev_t_v, ev_w, ev_q0, ev_q1,
            big_d, strict_d_n_u, strict_d_t_u, strict_d_n_v, strict_d_t_v,
            strict_d_w, strict_d_q,
        );

        let mut p_batch = SparsePolynomial::<Fr>::zero();
        let mut c_batch = Comm::zero();
        let mut v_batch = Fr::zero();
        let mut pow = Fr::one();
        for ((poly, c), v) in polys.iter().zip(&commits).zip(&evals) {
            p_batch = &p_batch + &(*poly * pow);
            c_batch += *c * pow;
            v_batch += *v * pow;
            pow *= eta_p;
        }

        let (batched_opening, opened_value) =
            pc::prove(srs, &Poly::Sparse(p_batch.clone()), beta);
        debug_assert_eq!(opened_value, v_batch, "batched opening must match v_batch");

        let p_stmt = PStatement { alpha, beta, y: mu };
        let pco_stmt = PcoStatement { commitment: c_batch, point: beta, value: v_batch };
        let pco_wit = PcoWitness { polynomial: p_batch };
        let proof = RokPProof {
            c_w, c_w_shift, c_n_u_shift, c_t_u_shift, c_n_v_shift, c_t_v_shift,
            c_q0, c_q1, c_q0_shift, c_q1_shift,
            ev_n_u, ev_t_u, ev_n_v, ev_t_v, ev_w, ev_q0, ev_q1,
            mu,
            batched_opening,
        };
        (p_stmt, pco_stmt, pco_wit, proof)
    }

    /// Verifier side. Returns `None` if any check fails.
    pub fn verify(
        params: &MParams,
        stmt: &MStatement,
        proof: &RokPProof,
        transcript: &mut Blake3Transcript,
    ) -> Option<(PStatement, PcoStatement)> {
        let srs = &params.srs;
        let n = params.n;
        let big_d = srs.powers_g1.len();
        let n_fr = Fr::from(n as u64);

        absorb_m_statement(transcript, stmt);

        transcript.absorb(b"rok_p::c_w", &proof.c_w);
        transcript.absorb(b"rok_p::c_w_shift", &proof.c_w_shift);
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
        let eta_p: Fr = transcript.squeeze_field(b"rok_p::eta_p");

        // Edge cases: T_u(β), T_v(β), and (β − α) must all be invertible.
        let t_u_n = n_fr * proof.ev_t_u;
        let t_v_n = n_fr * proof.ev_t_v;
        let t_u_n_inv = t_u_n.inverse()?;
        let t_v_n_inv = t_v_n.inverse()?;
        let beta_minus_alpha_inv = (beta - alpha).inverse()?;

        let u_at_beta = proof.ev_n_u * t_u_n_inv;
        let v_at_beta = proof.ev_n_v * t_v_n_inv;

        // Λ(α, β) = ((β^n − 1)·α − (α^n − 1)·β) / (n·(β − α)).
        let beta_n = pow_usize(beta, n);
        let alpha_n = pow_usize(alpha, n);
        let lambda_alpha_beta =
            ((beta_n - Fr::one()) * alpha - (alpha_n - Fr::one()) * beta)
                * (n_fr.inverse()? * beta_minus_alpha_inv);

        // g''(β) using μ in place of M(α, β).
        let g_prime = u_at_beta * proof.ev_w;
        let g_at_beta = lambda_alpha_beta * proof.ev_w - proof.mu * v_at_beta;
        let g_pp_at_beta = g_prime + eta * g_at_beta;

        // Sumcheck identity at β: g''(β) ?= y/n + β·q_0(β) + (β^n − 1)·q_1(β).
        let rhs = stmt.value * n_fr.inverse()?
            + beta * proof.ev_q0
            + (beta_n - Fr::one()) * proof.ev_q1;
        if g_pp_at_beta != rhs {
            return None;
        }

        // Reconstruct the batched commitment and value.
        let strict_d_n_u = stmt.d_n_u + 1;
        let strict_d_t_u = stmt.d_t_u + 1;
        let strict_d_n_v = stmt.d_n_v + 1;
        let strict_d_t_v = stmt.d_t_v + 1;
        let strict_d_w = n;
        let strict_d_q = n - 1;

        let evals = batched_evals(
            beta, proof.ev_n_u, proof.ev_t_u, proof.ev_n_v, proof.ev_t_v,
            proof.ev_w, proof.ev_q0, proof.ev_q1,
            big_d, strict_d_n_u, strict_d_t_u, strict_d_n_v, strict_d_t_v,
            strict_d_w, strict_d_q,
        );
        let commits = [
            stmt.c_n_u, stmt.c_t_u, stmt.c_n_v, stmt.c_t_v,
            proof.c_w, proof.c_q0, proof.c_q1,
            proof.c_n_u_shift, proof.c_t_u_shift, proof.c_n_v_shift, proof.c_t_v_shift,
            proof.c_w_shift, proof.c_q0_shift, proof.c_q1_shift,
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

        let p_stmt = PStatement { alpha, beta, y: proof.mu };
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

fn ifft_sparse(dom: &Radix2EvaluationDomain<Fr>, evals: &[Fr]) -> SparsePolynomial<Fr> {
    let coeffs = dom.ifft(evals);
    vec_to_sparse(&coeffs)
}

fn vec_to_sparse(v: &[Fr]) -> SparsePolynomial<Fr> {
    SparsePolynomial::from_coefficients_vec(
        v.iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, &c)| (i, c))
            .collect(),
    )
}

fn sparse_to_dense_vec(p: &SparsePolynomial<Fr>, len: usize) -> Vec<Fr> {
    let mut v = vec![Fr::zero(); len];
    for &(i, c) in p.iter() {
        if i < len {
            v[i] = c;
        }
    }
    v
}

/// Decompose `g''(X) = y/n + X·q_0(X) + (X^n − 1)·q_1(X)`,
/// returning `(q_0, q_1)`. Both have degree ≤ n − 2.
fn sumcheck_decompose(
    g_pp: &SparsePolynomial<Fr>,
    y_over_n: Fr,
    n: usize,
) -> (SparsePolynomial<Fr>, SparsePolynomial<Fr>) {
    // Materialize g'' as a dense coefficient vector of length 2n−1.
    let c = sparse_to_dense_vec(g_pp, 2 * n - 1);

    // r[k] = c_k + c_{k+n} for k = 0..n-2, r[n-1] = c_{n-1}.
    let mut r = vec![Fr::zero(); n];
    for k in 0..(n - 1) {
        r[k] = c[k] + c[k + n];
    }
    r[n - 1] = c[n - 1];
    debug_assert_eq!(r[0], y_over_n, "sumcheck identity must hold");

    // q_1[j] = c_{j+n} for j = 0..n−2.
    let q1: Vec<Fr> = (0..(n - 1)).map(|j| c[j + n]).collect();
    // q_0[j] = r[j+1] for j = 0..n−2.
    let q0: Vec<Fr> = (0..(n - 1)).map(|j| r[j + 1]).collect();

    (vec_to_sparse(&q0), vec_to_sparse(&q1))
}

fn pow_usize(x: Fr, k: usize) -> Fr {
    x.pow([k as u64])
}

#[allow(clippy::too_many_arguments)]
fn batched_evals(
    beta: Fr,
    ev_n_u: Fr, ev_t_u: Fr, ev_n_v: Fr, ev_t_v: Fr,
    ev_w: Fr, ev_q0: Fr, ev_q1: Fr,
    big_d: usize,
    strict_d_n_u: usize, strict_d_t_u: usize,
    strict_d_n_v: usize, strict_d_t_v: usize,
    strict_d_w: usize, strict_d_q: usize,
) -> [Fr; 14] {
    let s_n_u = pow_usize(beta, big_d - strict_d_n_u);
    let s_t_u = pow_usize(beta, big_d - strict_d_t_u);
    let s_n_v = pow_usize(beta, big_d - strict_d_n_v);
    let s_t_v = pow_usize(beta, big_d - strict_d_t_v);
    let s_w = pow_usize(beta, big_d - strict_d_w);
    let s_q = pow_usize(beta, big_d - strict_d_q);
    [
        // unshifted (k = 0..6)
        ev_n_u, ev_t_u, ev_n_v, ev_t_v, ev_w, ev_q0, ev_q1,
        // shifted (k = 7..13)
        s_n_u * ev_n_u, s_t_u * ev_t_u, s_n_v * ev_n_v, s_t_v * ev_t_v,
        s_w * ev_w, s_q * ev_q0, s_q * ev_q1,
    ]
}

fn absorb_m_statement(t: &mut Blake3Transcript, s: &MStatement) {
    t.absorb(b"rok_p::m_c_n_u", &s.c_n_u);
    t.absorb(b"rok_p::m_c_t_u", &s.c_t_u);
    t.absorb(b"rok_p::m_c_n_v", &s.c_n_v);
    t.absorb(b"rok_p::m_c_t_v", &s.c_t_v);
    t.absorb_usize(b"rok_p::m_d_n_u", s.d_n_u);
    t.absorb_usize(b"rok_p::m_d_t_u", s.d_t_u);
    t.absorb_usize(b"rok_p::m_d_n_v", s.d_n_v);
    t.absorb_usize(b"rok_p::m_d_t_v", s.d_t_v);
    t.absorb(b"rok_p::m_y", &s.value);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Relation;
    use crate::relations::m::leaf_instance;
    use crate::relations::pco::{PcoParams, PcoRelation};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    /// Random K×K identity-like leaf instance, with `srs` of capacity
    /// large enough to accommodate `D ≥ 2n` for all degree binding.
    fn identity_leaf(n: usize) -> (MParams, MStatement, MWitness) {
        let rng = &mut test_rng();
        let matrix: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        // SRS supports degrees up to (big_d - 1) = (2n + 4) - 1, i.e. D = 2n+4.
        // Plenty of headroom for the shifts D - 1 < D.
        let srs = pc::setup(2 * n + 3, rng);
        let params = MParams { srs, matrix, n };
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (stmt, wit) = leaf_instance(&params, alpha, beta);
        (params, stmt, wit)
    }

    /// Random sparse-but-non-identity matrix.
    fn random_matrix_leaf(n: usize) -> (MParams, MStatement, MWitness) {
        let rng = &mut test_rng();
        // Sparse: each row has two non-zero entries.
        let mut matrix: Vec<(usize, usize, Fr)> = Vec::new();
        for i in 0..n {
            matrix.push((i, i, Fr::rand(rng)));
            matrix.push((i, (i + 1) % n, Fr::rand(rng)));
        }
        let srs = pc::setup(2 * n + 3, rng);
        let params = MParams { srs, matrix, n };
        let alpha = Fr::rand(rng);
        let beta = Fr::rand(rng);
        let (stmt, wit) = leaf_instance(&params, alpha, beta);
        (params, stmt, wit)
    }

    #[test]
    fn roundtrip_identity_n4() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (p_p, pco_p, pco_wit, proof) =
            RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        let (p_v, pco_v) =
            RokP::verify(&params, &stmt, &proof, &mut t_v).expect("must verify");

        assert_eq!(p_p, p_v);
        assert_eq!(pco_p, pco_v);

        // The emitted PCO instance, taken with its witness, satisfies R_PCO.
        let pco_params = PcoParams { srs: params.srs.clone() };
        assert!(PcoRelation::is_satisfied(&pco_params, &pco_p, &pco_wit));
    }

    #[test]
    fn roundtrip_identity_n8() {
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

    #[test]
    fn roundtrip_random_matrix() {
        let (params, stmt, wit) = random_matrix_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (p_p, _, pco_wit, proof) =
            RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        let (p_v, pco_v) =
            RokP::verify(&params, &stmt, &proof, &mut t_v).expect("must verify");

        assert_eq!(p_p, p_v);
        let pco_params = PcoParams { srs: params.srs.clone() };
        assert!(PcoRelation::is_satisfied(&pco_params, &pco_v, &pco_wit));
    }

    /// Sanity: the emitted `(α, β, μ)` satisfies the R_P relation
    /// (i.e. `μ = λ(α)ᵀ·M·λ(β)`), checked manually since `p.rs` has
    /// no `is_satisfied`.
    #[test]
    fn emitted_p_statement_is_correct() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (p_stmt, _, _, _) = RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let dom = Radix2EvaluationDomain::<Fr>::new(params.n).unwrap();
        let lambda_alpha = dom.evaluate_all_lagrange_coefficients(p_stmt.alpha);
        let lambda_beta = dom.evaluate_all_lagrange_coefficients(p_stmt.beta);
        let expected_mu: Fr = params
            .matrix
            .iter()
            .map(|&(i, j, m_ij)| m_ij * lambda_alpha[i] * lambda_beta[j])
            .sum();
        assert_eq!(p_stmt.y, expected_mu);
    }

    /// Tampering `m_stmt.value` between prove and verify must reject
    /// (sumcheck identity no longer holds).
    #[test]
    fn tampered_value_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (_, _, _, proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let mut bad_stmt = stmt.clone();
        bad_stmt.value += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        assert!(RokP::verify(&params, &bad_stmt, &proof, &mut t_v).is_none());
    }

    /// Tampering `proof.mu` must reject.
    #[test]
    fn tampered_mu_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (_, _, _, mut proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);
        proof.mu += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        assert!(RokP::verify(&params, &stmt, &proof, &mut t_v).is_none());
    }

    /// Tampering any unshifted evaluation in the proof must reject
    /// (batched opening fails).
    #[test]
    fn tampered_evaluation_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::test");
        let (_, _, _, mut proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);
        proof.ev_w += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"rok_p::test");
        assert!(RokP::verify(&params, &stmt, &proof, &mut t_v).is_none());
    }

    /// Different transcript label between prover and verifier must reject.
    #[test]
    fn transcript_mismatch_rejected() {
        let (params, stmt, wit) = identity_leaf(4);
        let mut t_p = Blake3Transcript::new(b"rok_p::label_a");
        let (_, _, _, proof) = RokP::reduce(&params, &stmt, &wit, &mut t_p);

        let mut t_v = Blake3Transcript::new(b"rok_p::label_b");
        assert!(RokP::verify(&params, &stmt, &proof, &mut t_v).is_none());
    }
}
