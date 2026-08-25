//! Marlin's **outer** sumcheck — rounds 1 and 2, the witness-dependent half of
//! the prover. Ported from `arkworks-rs/marlin` `src/ahp/prover.rs`
//! (`prover_first_round` / `prover_second_round`, MIT/Apache-2.0), re-typed for
//! our `Fr` and rebuilt on our KZG + Blake3 transcript instead of
//! ark-poly-commit's `LabeledPolynomial` machinery.
//!
//! ```text
//!   round 1 (witness):  ŵ = (ẑ − x̂)/v_X ,  ẑ_A , ẑ_B  [, mask]
//!   round 2 (α, η):     t(X) ,  q_1 = [mask +] r_α·Σ_M η_M ẑ_M − t·ẑ
//!                       q_1 = h_1·v_H + X·g_1        (the sumcheck claims Σ = 0)
//! ```
//!
//! The output is the claim `t(β)`, which Marlin's round 3 discharges.
//!
//! **What this does and does not prove.** Round 1-2 verification treats `t(β)`
//! as a *claim*: nothing here ties the committed `t` to the actual matrices.
//! That is precisely round 3's job, and it is why this claim is the seam where
//! delegation becomes possible. Same structure as upstream.
//!
//! # The normalization seam
//!
//! `t(β)` is **not** this repo's `R_P` value at the same `(α, β, η)`:
//!
//! ```text
//!   t(β) = Σ_M η_M Σ_{i,j} M[i,j] · u_H(α,h_i) · L_j(β)     (Marlin)
//!   y    = Σ_M η_M Σ_{i,j} M[i,j] · L_i(α)     · L_j(β)     (R_P, see arithmetize.rs)
//! ```
//!
//! with `u_H(α,h_i) = |H|·L_i(α)/h_i` — a per-`i` factor, so no rescaling turns
//! one into the other. Marlin weights by the *unnormalized* `u_H` precisely so
//! its verifier can evaluate `r_α(β) = u_H(α,β)` in `O(log|H|)`; weighting by
//! `L_i(α)` instead would make `r_α` an interpolant with no closed form and
//! cost the verifier `O(|H|)`. The sumcheck identity forces the weight inside
//! `t` to match the `r_α` inside `q_1`, so the choice is not free. This repo
//! normalizes both sides because `P_M(α,β) = Σ M[i,j]λ_i(α)λ_j(β)` is the
//! bivariate evaluation the delegation construction is built on.
//!
//! Consequence: these rounds and `InnerLincheck` do not compose into one
//! end-to-end verifiable proof — each half is correct under its own
//! convention. For **cost** measurement, which is what these rounds exist for,
//! the difference is nil: identical `|K|`, identical degrees, identical
//! operations, different field values. Use [`repo_p_statement`] to obtain the
//! R_P claim at the same `(α, β, η)` for feeding round 3.
//!
//! **Departures from upstream, all deliberate:**
//! * Matrices index `z` directly over `H` rather than interleaving public
//!   inputs at multiples of `|H|/|X|`, so `calculate_t`'s index remap collapses
//!   to `index = j`. Same simplification `arithmetize.rs` already documents.
//! * One public input (`z[0] = 1`), so `|X| = 1` and `v_X(X) = X − 1`.
//! * Rounds 1-2 carry their own batched opening at `β` rather than joining
//!   round 3's. Upstream issues a single `PC::open_combinations` across every
//!   round; we issue two. That bills the outer rounds one extra degree-`D` MSM,
//!   which **inflates** the witness-dependent share — the conservative
//!   direction for any claim about delegation being worthwhile.
//! * Zero-knowledge is a flag. With `zk = false` there is no mask polynomial
//!   and no `v_H` blinding, which matches `inner_lincheck`'s non-hiding KZG;
//!   with `zk = true` the witness-dependent share grows. Reporting both brackets
//!   the answer rather than picking one.

use ark_bls12_381::Fr;
use ark_ff::{One, UniformRand, Zero};
use ark_poly::{
    univariate::{DenseOrSparsePolynomial, DensePolynomial},
    DenseUVPolynomial, EvaluationDomain, Polynomial, Radix2EvaluationDomain,
};
use ark_std::rand::RngCore;

use crate::marlin_ahp::bivariate::UnnormalizedBivariateLagrangePoly;
use crate::marlin_ahp::r1cs::R1csInstance;
use crate::pc::{self, Comm, Opening, Poly, Srs};
use crate::reductions::poly_util::x_shift_dense;
use crate::relations::p::PStatement;
use crate::transcript::Blake3Transcript;

pub struct OuterSumcheck;

/// What the outer sumcheck leaves for round 3: `t(β)` in Marlin's
/// normalization, plus the challenges that produced it.
///
/// Deliberately *not* a [`PStatement`] — see the module docs on the
/// normalization seam. Feeding `t_beta` to `InnerLincheck` would discharge a
/// different quantity than the one it was built to prove.
#[derive(Clone, Copy, Debug)]
pub struct OuterClaim {
    pub alpha: Fr,
    pub beta: Fr,
    pub eta: Fr,
    pub t_beta: Fr,
}

/// The R_P claim at the same `(α, β, η)`, in *this repo's* convention:
/// `y = Σ_M η_M Σ_{i,j} M[i,j]·λ_i(α)·λ_j(β)`.
///
/// `O(|H| + nnz)`, so keep it out of any timed region — it is a harness
/// convenience for chaining round 3 onto the outer rounds, not part of either
/// prover.
pub fn repo_p_statement(
    inst: &R1csInstance,
    domain_h: &Radix2EvaluationDomain<Fr>,
    claim: &OuterClaim,
) -> PStatement {
    let l_alpha = domain_h.evaluate_all_lagrange_coefficients(claim.alpha);
    let l_beta = domain_h.evaluate_all_lagrange_coefficients(claim.beta);
    let eval = |m: &[(usize, usize, Fr)]| -> Fr {
        m.iter().map(|&(i, j, v)| v * l_alpha[i] * l_beta[j]).sum()
    };
    let y = eval(&inst.matrix_a)
        + claim.eta * eval(&inst.matrix_b)
        + claim.eta * claim.eta * eval(&inst.matrix_c);
    PStatement { alpha: claim.alpha, beta: claim.beta, y, eta: claim.eta }
}

/// Round-1 output: the witness-dependent oracles, retained for round 2.
pub struct OuterFirstRound {
    pub w: DensePolynomial<Fr>,
    /// `ẑ = ŵ·v_X + x̂`, rebuilt *after* blinding so the identity holds exactly.
    pub z_hat: DensePolynomial<Fr>,
    pub z_a: DensePolynomial<Fr>,
    pub z_b: DensePolynomial<Fr>,
    pub mask: Option<DensePolynomial<Fr>>,
    pub c_w: Comm,
    pub c_z_a: Comm,
    pub c_z_b: Comm,
    pub c_mask: Option<Comm>,
}

/// A complete outer-sumcheck proof: eight commitment/evaluation slots and one
/// batched KZG opening at `β`.
pub struct OuterProof {
    pub c_w: Comm,
    pub c_z_a: Comm,
    pub c_z_b: Comm,
    pub c_mask: Option<Comm>,
    pub c_t: Comm,
    pub c_g1: Comm,
    pub c_h1: Comm,
    pub c_g1_shift: Comm,
    pub ev_w: Fr,
    pub ev_z_a: Fr,
    pub ev_z_b: Fr,
    pub ev_mask: Option<Fr>,
    pub ev_t: Fr,
    pub ev_g1: Fr,
    pub ev_h1: Fr,
    pub batched_opening: Opening,
}

impl OuterSumcheck {
    /// Round 1. Interpolates the witness-dependent oracles and commits them.
    pub fn first_round(
        srs: &Srs,
        inst: &R1csInstance,
        domain_h: &Radix2EvaluationDomain<Fr>,
        zk: bool,
        rng: &mut impl RngCore,
        transcript: &mut Blake3Transcript,
    ) -> OuterFirstRound {
        let n = domain_h.size();
        assert_eq!(n, inst.n, "domain_h must match the instance");

        // The matrix-vector products are prover work — they exist only to feed
        // the ẑ_A, ẑ_B oracles — so they are computed here, inside the round
        // and inside its timing, rather than cached on the instance. The
        // witness `z` itself is an input and is not charged to the prover.
        let z_a_evals = inst.matvec(&inst.matrix_a);
        let z_b_evals = inst.matvec(&inst.matrix_b);

        let z_hat_raw = DensePolynomial::from_coefficients_vec(domain_h.ifft(&inst.z));
        let mut z_a = DensePolynomial::from_coefficients_vec(domain_h.ifft(&z_a_evals));
        let mut z_b = DensePolynomial::from_coefficients_vec(domain_h.ifft(&z_b_evals));

        // ŵ = (ẑ − x̂) / v_X with x̂ = 1 and v_X = X − 1. The division is exact
        // because ẑ(1) = z[0] = 1.
        let x_hat = DensePolynomial::from_coefficients_vec(vec![Fr::one()]);
        let v_x = DensePolynomial::from_coefficients_vec(vec![-Fr::one(), Fr::one()]);
        let numerator = &z_hat_raw - &x_hat;
        let (mut w, rem) = DenseOrSparsePolynomial::from(&numerator)
            .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&v_x))
            .expect("v_X is non-zero");
        debug_assert!(rem.is_zero(), "ẑ(1) must equal the public input");

        if zk {
            // Blinding by a multiple of v_H leaves every value on H unchanged,
            // so the R1CS relation — and hence the sumcheck identity — survives.
            let v_h = vanishing_poly(n);
            w = &w + &scale(&v_h, Fr::rand(rng));
            z_a = &z_a + &scale(&v_h, Fr::rand(rng));
            z_b = &z_b + &scale(&v_h, Fr::rand(rng));
        }

        // Rebuild ẑ from the (possibly blinded) ŵ so ẑ = ŵ·v_X + x̂ holds exactly.
        let z_hat = &(&w * &v_x) + &x_hat;
        debug_assert!(
            domain_h
                .elements()
                .zip(&inst.z)
                .all(|(h, &zi)| z_hat.evaluate(&h) == zi),
            "blinded ẑ must still interpolate z on H",
        );

        let mask = if zk { Some(zero_sum_mask(n, rng)) } else { None };

        let c_w = pc::commit(srs, &Poly::Dense(w.clone()));
        let c_z_a = pc::commit(srs, &Poly::Dense(z_a.clone()));
        let c_z_b = pc::commit(srs, &Poly::Dense(z_b.clone()));
        let c_mask = mask.as_ref().map(|m| pc::commit(srs, &Poly::Dense(m.clone())));

        transcript.absorb(b"os::c_w", &c_w);
        transcript.absorb(b"os::c_z_a", &c_z_a);
        transcript.absorb(b"os::c_z_b", &c_z_b);
        transcript.absorb_usize(b"os::zk", usize::from(zk));
        if let Some(c) = &c_mask {
            transcript.absorb(b"os::c_mask", c);
        }

        OuterFirstRound { w, z_hat, z_a, z_b, mask, c_w, c_z_a, c_z_b, c_mask }
    }

    /// Round 2. Consumes `α, η`, builds `t`, `g_1`, `h_1`, and opens everything
    /// at `β`. Returns the proof and the claim `t(β)` that Marlin's round 3
    /// discharges (see the module docs on the normalization seam).
    pub fn second_round(
        srs: &Srs,
        inst: &R1csInstance,
        domain_h: &Radix2EvaluationDomain<Fr>,
        first: OuterFirstRound,
        transcript: &mut Blake3Transcript,
    ) -> (OuterProof, OuterClaim) {
        let n = domain_h.size();
        let big_d = srs.powers_g1.len();

        let alpha = squeeze_outside_h(transcript, b"os::alpha", domain_h);
        let eta: Fr = transcript.squeeze_field(b"os::eta");

        let r_alpha_evals =
            domain_h.batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(alpha);
        let r_alpha_poly =
            DensePolynomial::from_coefficients_vec(domain_h.ifft(&r_alpha_evals));

        let t_poly = calculate_t(inst, domain_h, eta, &r_alpha_evals);

        // Σ_M η_M·ẑ_M with η_A = 1, η_B = η, η_C = η² — the single-η convention
        // `PStatement` uses, so `y = t(β)` lines up with `InnerLincheck`.
        let eta_sq = eta * eta;
        let z_ab = &first.z_a * &first.z_b;
        let summed_z_m = &(&first.z_a + &scale(&first.z_b, eta)) + &scale(&z_ab, eta_sq);

        let mut q_1 = &(&r_alpha_poly * &summed_z_m) - &(&t_poly * &first.z_hat);
        if let Some(mask) = &first.mask {
            q_1 = &q_1 + mask;
        }

        let (h_1, x_g_1) = q_1.divide_by_vanishing_poly(*domain_h);
        debug_assert!(
            x_g_1.coeffs.first().copied().unwrap_or_else(Fr::zero).is_zero(),
            "outer sumcheck claims Σ_H q_1 = 0, so X·g_1 has no constant term",
        );
        let g_1 = DensePolynomial::from_coefficients_vec(
            x_g_1.coeffs.get(1..).unwrap_or(&[]).to_vec(),
        );
        debug_assert!(
            g_1.is_zero() || g_1.degree() <= n - 2,
            "sumcheck soundness needs deg g_1 ≤ |H| − 2",
        );

        // Degree binding for g_1, exactly as `inner_lincheck` does for g_2:
        // committing X^{D−(|H|−1)}·g_1 is only possible under the size-D SRS
        // when the bound holds, and the batch below ties the two together at β.
        let g1_shift = x_shift_dense(&g_1, big_d - (n - 1));

        let c_t = pc::commit(srs, &Poly::Dense(t_poly.clone()));
        let c_g1 = pc::commit(srs, &Poly::Dense(g_1.clone()));
        let c_h1 = pc::commit(srs, &Poly::Dense(h_1.clone()));
        let c_g1_shift = pc::commit(srs, &Poly::Dense(g1_shift.clone()));

        transcript.absorb(b"os::c_t", &c_t);
        transcript.absorb(b"os::c_g1", &c_g1);
        transcript.absorb(b"os::c_h1", &c_h1);
        transcript.absorb(b"os::c_g1_shift", &c_g1_shift);

        let beta = squeeze_outside_h(transcript, b"os::beta", domain_h);

        let ev_w = first.w.evaluate(&beta);
        let ev_z_a = first.z_a.evaluate(&beta);
        let ev_z_b = first.z_b.evaluate(&beta);
        let ev_mask = first.mask.as_ref().map(|m| m.evaluate(&beta));
        let ev_t = t_poly.evaluate(&beta);
        let ev_g1 = g_1.evaluate(&beta);
        let ev_h1 = h_1.evaluate(&beta);

        absorb_evals(transcript, ev_w, ev_z_a, ev_z_b, ev_mask, ev_t, ev_g1, ev_h1);
        let eta_prime: Fr = transcript.squeeze_field(b"os::eta_prime");

        // η'-batched opening, in the same slot order the verifier rebuilds.
        let mut polys: Vec<&DensePolynomial<Fr>> =
            vec![&first.w, &first.z_a, &first.z_b];
        if let Some(m) = &first.mask {
            polys.push(m);
        }
        polys.extend([&t_poly, &g_1, &h_1, &g1_shift]);

        let mut p_batch = DensePolynomial::<Fr>::zero();
        let mut pow = Fr::one();
        for p in polys {
            p_batch = &p_batch + &scale(p, pow);
            pow *= eta_prime;
        }
        let (batched_opening, _) = pc::prove(srs, &Poly::Dense(p_batch), beta);

        let proof = OuterProof {
            c_w: first.c_w,
            c_z_a: first.c_z_a,
            c_z_b: first.c_z_b,
            c_mask: first.c_mask,
            c_t,
            c_g1,
            c_h1,
            c_g1_shift,
            ev_w,
            ev_z_a,
            ev_z_b,
            ev_mask,
            ev_t,
            ev_g1,
            ev_h1,
            batched_opening,
        };
        let claim = OuterClaim { alpha, beta, eta, t_beta: ev_t };
        (proof, claim)
    }

    /// Both rounds, for callers that do not need to time them separately.
    pub fn prove(
        srs: &Srs,
        inst: &R1csInstance,
        domain_h: &Radix2EvaluationDomain<Fr>,
        zk: bool,
        rng: &mut impl RngCore,
        transcript: &mut Blake3Transcript,
    ) -> (OuterProof, OuterClaim) {
        let first = Self::first_round(srs, inst, domain_h, zk, rng, transcript);
        Self::second_round(srs, inst, domain_h, first, transcript)
    }

    /// Verifier. Re-derives `α, η, β` from the transcript, checks the sumcheck
    /// identity at `β`, and checks the batched opening. Returns the claim it is
    /// left holding — `t(β)` is asserted here, not proven; round 3 proves it
    /// against the matrices.
    pub fn verify(
        srs: &Srs,
        domain_h: &Radix2EvaluationDomain<Fr>,
        proof: &OuterProof,
        transcript: &mut Blake3Transcript,
    ) -> Option<OuterClaim> {
        let n = domain_h.size();
        let big_d = srs.powers_g1.len();
        let zk = proof.c_mask.is_some();

        transcript.absorb(b"os::c_w", &proof.c_w);
        transcript.absorb(b"os::c_z_a", &proof.c_z_a);
        transcript.absorb(b"os::c_z_b", &proof.c_z_b);
        transcript.absorb_usize(b"os::zk", usize::from(zk));
        if let Some(c) = &proof.c_mask {
            transcript.absorb(b"os::c_mask", c);
        }

        let alpha = squeeze_outside_h(transcript, b"os::alpha", domain_h);
        let eta: Fr = transcript.squeeze_field(b"os::eta");

        transcript.absorb(b"os::c_t", &proof.c_t);
        transcript.absorb(b"os::c_g1", &proof.c_g1);
        transcript.absorb(b"os::c_h1", &proof.c_h1);
        transcript.absorb(b"os::c_g1_shift", &proof.c_g1_shift);

        let beta = squeeze_outside_h(transcript, b"os::beta", domain_h);

        // Mask presence must match what the evaluations claim.
        if proof.ev_mask.is_some() != zk {
            return None;
        }

        absorb_evals(
            transcript, proof.ev_w, proof.ev_z_a, proof.ev_z_b, proof.ev_mask,
            proof.ev_t, proof.ev_g1, proof.ev_h1,
        );
        let eta_prime: Fr = transcript.squeeze_field(b"os::eta_prime");

        // Sumcheck identity at β.
        let r_alpha_beta =
            domain_h.eval_unnormalized_bivariate_lagrange_poly(alpha, beta);
        let z_beta = proof.ev_w * (beta - Fr::one()) + Fr::one();
        let eta_sq = eta * eta;
        let summed = proof.ev_z_a
            + eta * proof.ev_z_b
            + eta_sq * proof.ev_z_a * proof.ev_z_b;
        let mut lhs = r_alpha_beta * summed - proof.ev_t * z_beta;
        if let Some(m) = proof.ev_mask {
            lhs += m;
        }
        let v_h_beta = domain_h.evaluate_vanishing_polynomial(beta);
        let rhs = proof.ev_h1 * v_h_beta + beta * proof.ev_g1;
        if lhs != rhs {
            return None;
        }

        // Batched opening. The shifted slot's expected value is derived, not
        // sent: β^{D−(|H|−1)}·ev_g1.
        let shift_scalar = pow_usize(beta, big_d - (n - 1));
        let mut commits: Vec<Comm> = vec![proof.c_w, proof.c_z_a, proof.c_z_b];
        let mut evals: Vec<Fr> = vec![proof.ev_w, proof.ev_z_a, proof.ev_z_b];
        if let (Some(c), Some(v)) = (proof.c_mask, proof.ev_mask) {
            commits.push(c);
            evals.push(v);
        }
        commits.extend([proof.c_t, proof.c_g1, proof.c_h1, proof.c_g1_shift]);
        evals.extend([
            proof.ev_t,
            proof.ev_g1,
            proof.ev_h1,
            shift_scalar * proof.ev_g1,
        ]);

        let mut c_batch = Comm::zero();
        let mut v_batch = Fr::zero();
        let mut pow = Fr::one();
        for (c, v) in commits.iter().zip(&evals) {
            c_batch += *c * pow;
            v_batch += *v * pow;
            pow *= eta_prime;
        }
        if !pc::verify(srs, &c_batch, beta, v_batch, &proof.batched_opening) {
            return None;
        }

        Some(OuterClaim { alpha, beta, eta, t_beta: proof.ev_t })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `t(h_j) = Σ_M η_M · Σ_i M[i,j] · r_α(h_i)`, then interpolated over `H`.
///
/// Upstream reindexes the column through the public-input interleaving; our
/// matrices address `z` directly over `H`, so the column index is used as-is.
///
/// Public so `examples/bench_marlin_split.rs` can time it on its own: `t`
/// depends only on `α` and the matrices, never on the witness, so it is
/// witness-*independent* work sitting inside a witness-dependent round.
pub fn calculate_t(
    inst: &R1csInstance,
    domain_h: &Radix2EvaluationDomain<Fr>,
    eta: Fr,
    r_alpha_evals: &[Fr],
) -> DensePolynomial<Fr> {
    let n = domain_h.size();
    let mut t_evals = vec![Fr::zero(); n];
    let etas = [Fr::one(), eta, eta * eta];
    let matrices = [&inst.matrix_a, &inst.matrix_b, &inst.matrix_c];
    for (m, eta_m) in matrices.into_iter().zip(etas) {
        for &(i, j, v) in m {
            t_evals[j] += eta_m * v * r_alpha_evals[i];
        }
    }
    DensePolynomial::from_coefficients_vec(domain_h.ifft(&t_evals))
}

/// A random polynomial of degree `3|H| − 1` whose values sum to zero over `H`.
///
/// `Σ_{h ∈ H} p(h) = |H| · Σ_j coeff[j·|H|]`, since `Σ_h h^m` is `|H|` when
/// `|H| | m` and zero otherwise. So zeroing that subsum is a single adjustment
/// to the constant term.
fn zero_sum_mask(n: usize, rng: &mut impl RngCore) -> DensePolynomial<Fr> {
    let mut coeffs: Vec<Fr> = (0..3 * n).map(|_| Fr::rand(rng)).collect();
    let tail = coeffs[n] + coeffs[2 * n];
    coeffs[0] = -tail;
    let poly = DensePolynomial::from_coefficients_vec(coeffs);
    debug_assert!({
        let dom = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        dom.elements().map(|h| poly.evaluate(&h)).sum::<Fr>().is_zero()
    });
    poly
}

fn vanishing_poly(n: usize) -> DensePolynomial<Fr> {
    let mut coeffs = vec![Fr::zero(); n + 1];
    coeffs[0] = -Fr::one();
    coeffs[n] = Fr::one();
    DensePolynomial::from_coefficients_vec(coeffs)
}

fn scale(p: &DensePolynomial<Fr>, c: Fr) -> DensePolynomial<Fr> {
    DensePolynomial::from_coefficients_vec(p.coeffs.iter().map(|&x| x * c).collect())
}

fn pow_usize(x: Fr, k: usize) -> Fr {
    use ark_ff::Field;
    x.pow([k as u64])
}

/// Squeeze a challenge that is not in `H`. Marlin needs `α, β ∉ H` so that
/// `u_H(α, ·)` and the division by `(α − β)` are well defined. A collision has
/// probability `|H|/|F|`; the retry keeps the transcript deterministic.
fn squeeze_outside_h(
    transcript: &mut Blake3Transcript,
    label: &'static [u8],
    domain_h: &Radix2EvaluationDomain<Fr>,
) -> Fr {
    let mut candidate: Fr = transcript.squeeze_field(label);
    let mut tries = 0usize;
    while domain_h.evaluate_vanishing_polynomial(candidate).is_zero() {
        transcript.absorb_usize(b"os::retry", tries);
        candidate = transcript.squeeze_field(label);
        tries += 1;
    }
    candidate
}

#[allow(clippy::too_many_arguments)]
fn absorb_evals(
    t: &mut Blake3Transcript,
    ev_w: Fr,
    ev_z_a: Fr,
    ev_z_b: Fr,
    ev_mask: Option<Fr>,
    ev_t: Fr,
    ev_g1: Fr,
    ev_h1: Fr,
) {
    t.absorb(b"os::ev_w", &ev_w);
    t.absorb(b"os::ev_z_a", &ev_z_a);
    t.absorb(b"os::ev_z_b", &ev_z_b);
    if let Some(m) = ev_mask {
        t.absorb(b"os::ev_mask", &m);
    }
    t.absorb(b"os::ev_t", &ev_t);
    t.absorb(b"os::ev_g1", &ev_g1);
    t.absorb(b"os::ev_h1", &ev_h1);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marlin_ahp::r1cs::satisfiable_instance;
    use ark_std::test_rng;

    /// SRS large enough for the ZK mask (degree `3n − 1`) and the g_1 shift.
    fn setup(n: usize) -> (Srs, Radix2EvaluationDomain<Fr>) {
        let srs = pc::setup(4 * n - 1, &mut test_rng());
        let domain_h = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        (srs, domain_h)
    }

    fn run(n: usize, zk: bool) -> bool {
        let (srs, domain_h) = setup(n);
        let rng = &mut test_rng();
        let inst = satisfiable_instance(n, 2, rng);

        let mut t_p = Blake3Transcript::new(b"os::test");
        let (proof, claim) =
            OuterSumcheck::prove(&srs, &inst, &domain_h, zk, rng, &mut t_p);

        let mut t_v = Blake3Transcript::new(b"os::test");
        match OuterSumcheck::verify(&srs, &domain_h, &proof, &mut t_v) {
            Some(seen) => {
                assert_eq!(seen.alpha, claim.alpha);
                assert_eq!(seen.beta, claim.beta);
                assert_eq!(seen.t_beta, claim.t_beta);
                assert_eq!(seen.eta, claim.eta);
                true
            }
            None => false,
        }
    }

    #[test]
    fn roundtrip_no_zk() {
        for n in [4usize, 8, 16] {
            assert!(run(n, false), "n = {n}");
        }
    }

    #[test]
    fn roundtrip_zk() {
        for n in [4usize, 8, 16] {
            assert!(run(n, true), "n = {n}");
        }
    }

    /// `t(β)` must equal Marlin's closed form
    /// `Σ_M η_M Σ_{i,j} M[i,j]·u_H(α,h_i)·L_j(β)` — the row weight is the
    /// *unnormalized* `u_H`, which is what keeps the outer verifier succinct.
    /// If this drifts, `q_1`'s `r_α` and `t`'s weights have gone out of step
    /// and the sumcheck identity is accidental.
    #[test]
    fn t_at_beta_matches_marlin_closed_form() {
        let n = 16usize;
        let (srs, domain_h) = setup(n);
        let rng = &mut test_rng();
        let inst = satisfiable_instance(n, 2, rng);

        let mut t_p = Blake3Transcript::new(b"os::test");
        let (_proof, claim) =
            OuterSumcheck::prove(&srs, &inst, &domain_h, false, rng, &mut t_p);

        let u_alpha = domain_h
            .batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(claim.alpha);
        let l_beta = domain_h.evaluate_all_lagrange_coefficients(claim.beta);
        let eval = |m: &[(usize, usize, Fr)]| -> Fr {
            m.iter().map(|&(i, j, v)| v * u_alpha[i] * l_beta[j]).sum()
        };
        let expected = eval(&inst.matrix_a)
            + claim.eta * eval(&inst.matrix_b)
            + claim.eta * claim.eta * eval(&inst.matrix_c);
        assert_eq!(claim.t_beta, expected);
    }

    /// The normalization seam, pinned: Marlin's `t(β)` and this repo's R_P `y`
    /// are different functionals at the same `(α, β, η)`, and `u_H(α,h_i) =
    /// |H|·L_i(α)/h_i` is why. Documented as a test so that anyone who later
    /// "fixes" one to match the other has to confront the reason.
    #[test]
    fn t_beta_and_repo_r_p_differ_by_row_normalization() {
        let n = 16usize;
        let (srs, domain_h) = setup(n);
        let rng = &mut test_rng();
        let inst = satisfiable_instance(n, 2, rng);

        let mut t_p = Blake3Transcript::new(b"os::test");
        let (_proof, claim) =
            OuterSumcheck::prove(&srs, &inst, &domain_h, false, rng, &mut t_p);

        let p = repo_p_statement(&inst, &domain_h, &claim);
        assert_eq!(p.alpha, claim.alpha);
        assert_eq!(p.beta, claim.beta);
        assert_eq!(p.eta, claim.eta);
        assert_ne!(p.y, claim.t_beta, "the two conventions must not coincide");

        // The per-i factor: u_H(α,h_i) = |H|·L_i(α)/h_i.
        let u_alpha = domain_h
            .batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(claim.alpha);
        let l_alpha = domain_h.evaluate_all_lagrange_coefficients(claim.alpha);
        let n_fr = Fr::from(n as u64);
        for (i, h) in domain_h.elements().enumerate() {
            assert_eq!(u_alpha[i], n_fr * l_alpha[i] / h);
        }
    }

    /// `repo_p_statement` must agree with how every other benchmark in the
    /// crate builds an R_P claim, or round-3 timings would not be comparable.
    #[test]
    fn repo_p_statement_matches_bivariate_evaluation() {
        let n = 16usize;
        let (srs, domain_h) = setup(n);
        let rng = &mut test_rng();
        let inst = satisfiable_instance(n, 2, rng);

        let mut t_p = Blake3Transcript::new(b"os::test");
        let (_proof, claim) =
            OuterSumcheck::prove(&srs, &inst, &domain_h, false, rng, &mut t_p);
        let p = repo_p_statement(&inst, &domain_h, &claim);

        let l_alpha = domain_h.evaluate_all_lagrange_coefficients(p.alpha);
        let l_beta = domain_h.evaluate_all_lagrange_coefficients(p.beta);
        let eval = |m: &[(usize, usize, Fr)]| -> Fr {
            m.iter().map(|&(i, j, v)| v * l_alpha[i] * l_beta[j]).sum()
        };
        let expected = eval(&inst.matrix_a)
            + p.eta * eval(&inst.matrix_b)
            + p.eta * p.eta * eval(&inst.matrix_c);
        assert_eq!(p.y, expected);
    }

    #[test]
    fn tampered_t_rejected() {
        let n = 8usize;
        let (srs, domain_h) = setup(n);
        let rng = &mut test_rng();
        let inst = satisfiable_instance(n, 2, rng);

        let mut t_p = Blake3Transcript::new(b"os::test");
        let (mut proof, _) =
            OuterSumcheck::prove(&srs, &inst, &domain_h, false, rng, &mut t_p);
        proof.ev_t += Fr::one();

        let mut t_v = Blake3Transcript::new(b"os::test");
        assert!(OuterSumcheck::verify(&srs, &domain_h, &proof, &mut t_v).is_none());
    }

    #[test]
    fn tampered_h1_rejected() {
        let n = 8usize;
        let (srs, domain_h) = setup(n);
        let rng = &mut test_rng();
        let inst = satisfiable_instance(n, 2, rng);

        let mut t_p = Blake3Transcript::new(b"os::test");
        let (mut proof, _) =
            OuterSumcheck::prove(&srs, &inst, &domain_h, false, rng, &mut t_p);
        proof.ev_h1 += Fr::one();

        let mut t_v = Blake3Transcript::new(b"os::test");
        assert!(OuterSumcheck::verify(&srs, &domain_h, &proof, &mut t_v).is_none());
    }

    /// An unsatisfying assignment must fail: the sumcheck no longer sums to
    /// zero over H, so `X·g_1` picks up a constant term and the identity at β
    /// breaks.
    #[test]
    fn unsatisfying_witness_rejected() {
        let n = 8usize;
        let (srs, domain_h) = setup(n);
        let rng = &mut test_rng();
        let mut inst = satisfiable_instance(n, 2, rng);
        // Break the relation without touching the z used to build z_a / z_b.
        inst.matrix_c[0].2 += Fr::one();
        assert!(!inst.is_satisfied());

        let mut t_p = Blake3Transcript::new(b"os::test");
        let (proof, _) =
            OuterSumcheck::prove(&srs, &inst, &domain_h, false, rng, &mut t_p);

        let mut t_v = Blake3Transcript::new(b"os::test");
        assert!(OuterSumcheck::verify(&srs, &domain_h, &proof, &mut t_v).is_none());
    }
}
