//! Benchmark: amortization vs. independent inner-linchecks.
//!
//! Per the design discussion in `docs/decisions/shared_srs.md` and the
//! conversation in 2026-06: the "K outer sumchecks" cost cancels between
//! the two sides, so the differential cost is
//!
//!   Side 1 (no amortization): K · T_lincheck(n)
//!   Side 2 (with amortization): T_amortize(n, K) + 1 · T_lincheck(n)
//!
//! Crossover K* where the two are equal answers the thesis question
//! "is amortization worth it?" given a fixed Marlin discharge.

use ark_bls12_381::Fr;
use ark_ff::UniformRand;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::test_rng;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion,
};

use amortized_proofs_masters_thesis::marlin_ahp::inner_lincheck::{
    InnerLincheck, InnerLincheckIndex,
};
use amortized_proofs_masters_thesis::pc;
use amortized_proofs_masters_thesis::reductions::abc_fold::AbcFold;
use amortized_proofs_masters_thesis::reductions::discharge::prove_discharge;
use amortized_proofs_masters_thesis::reductions::rok_p::RokP;
use amortized_proofs_masters_thesis::relations::abc::{
    leaf_instance, AbcParams, AbcStatement, AbcWitness,
};
use amortized_proofs_masters_thesis::relations::p::PStatement;
use amortized_proofs_masters_thesis::relations::pcc::{PccParams, PccStatement, PccWitness};
use amortized_proofs_masters_thesis::transcript::Blake3Transcript;

/// Subgroup size — shared by all benchmarks here. Big enough that timings
/// are stable, small enough that benches finish in reasonable wall-clock.
const N: usize = 8;

/// `K` values to sweep. AbcFold requires powers of two ≥ 2.
const K_VALUES: &[usize] = &[1, 2, 4, 8, 16];
const K_VALUES_AMORTIZED: &[usize] = &[2, 4, 8, 16];

struct Setup {
    srs: pc::Srs,
    abc_params: AbcParams,
    index: InnerLincheckIndex,
    n: usize,
}

fn build_setup(n: usize) -> Setup {
    let rng = &mut test_rng();
    let a: Vec<(usize, usize, Fr)> =
        (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
    let b: Vec<(usize, usize, Fr)> =
        (0..n).map(|i| (i, (i + 1) % n, Fr::rand(rng))).collect();
    let c: Vec<(usize, usize, Fr)> =
        (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
    // SRS sized to safely cover both consumers (amortization ≈ 2n, lincheck ≈ |K|).
    let srs = pc::setup(4 * n, rng);
    let abc_params = AbcParams {
        srs: srs.clone(),
        matrix_a: a.clone(),
        matrix_b: b.clone(),
        matrix_c: c.clone(),
        n,
    };
    let index = InnerLincheck::index(&srs, &a, &b, &c, n);
    Setup { srs, abc_params, index, n }
}

/// `K` synthetic R_P-shaped claims `(α_k, β_k, y_k, η_k)`, with `y_k`
/// honestly computed from the matrices. Used by Side 1.
fn synthetic_claims(setup: &Setup, k: usize) -> Vec<PStatement> {
    let rng = &mut test_rng();
    let dom = Radix2EvaluationDomain::<Fr>::new(setup.n).unwrap();
    (0..k)
        .map(|_| {
            let alpha = Fr::rand(rng);
            let beta = Fr::rand(rng);
            let eta = Fr::rand(rng);
            let lambda_alpha = dom.evaluate_all_lagrange_coefficients(alpha);
            let lambda_beta = dom.evaluate_all_lagrange_coefficients(beta);
            let eval = |matrix: &[(usize, usize, Fr)]| -> Fr {
                matrix
                    .iter()
                    .map(|&(i, j, m)| m * lambda_alpha[i] * lambda_beta[j])
                    .sum()
            };
            let p_a = eval(&setup.abc_params.matrix_a);
            let p_b = eval(&setup.abc_params.matrix_b);
            let p_c = eval(&setup.abc_params.matrix_c);
            let y = p_a + eta * p_b + eta * eta * p_c;
            PStatement { alpha, beta, y, eta }
        })
        .collect()
}

/// `K` random leaves for the amortization side.
fn random_leaves(setup: &Setup, k: usize) -> Vec<(AbcStatement, AbcWitness)> {
    let rng = &mut test_rng();
    (0..k)
        .map(|_| leaf_instance(&setup.abc_params, Fr::rand(rng), Fr::rand(rng)))
        .collect()
}

/// Side 1: `K` independent linchecks on K synthetic R_P claims.
fn bench_side_1(c: &mut Criterion) {
    let setup = build_setup(N);
    let mut group = c.benchmark_group("side_1_K_linchecks");
    for &k in K_VALUES {
        let claims = synthetic_claims(&setup, k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| {
                for claim in &claims {
                    let mut t = Blake3Transcript::new(b"bench::side1");
                    let proof =
                        InnerLincheck::prove(&setup.srs, &setup.index, claim, &mut t);
                    black_box(proof);
                }
            });
        });
    }
    group.finish();
}

/// Side 2 (full sound pipeline): AbcFold on K leaves → discharge the K·κ fold
/// promises via FsMtPcc + PcoFold + one KZG opening → RokP + 1 inner-lincheck
/// on the root R_P.
fn bench_side_2(c: &mut Criterion) {
    let setup = build_setup(N);
    let pcc_params = PccParams { srs: setup.srs.clone() };
    let mut group = c.benchmark_group("side_2_amortize_plus_lincheck");
    for &k in K_VALUES_AMORTIZED {
        let leaves = random_leaves(&setup, k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter_batched(
                || leaves.clone(),
                |leaves| {
                    // 1. Fold: K leaves → root + K·κ R_PCC promises.
                    let mut t_fold = Blake3Transcript::new(b"bench::side2::fold");
                    let (root, _paths, bundles) =
                        AbcFold::prove(&setup.abc_params, leaves, &mut t_fold);

                    // 2. Discharge promises with a single materialized root
                    //    witness + one KZG opening.
                    let pcc_stmts: Vec<Vec<PccStatement>> = bundles
                        .iter()
                        .map(|bd| bd.iter().map(|(s, _)| s.clone()).collect())
                        .collect();
                    let pcc_wits: Vec<Vec<PccWitness>> = bundles
                        .into_iter()
                        .map(|bd| bd.into_iter().map(|(_, w)| w).collect())
                        .collect();
                    let mut t_fsmt = Blake3Transcript::new(b"bench::side2::fsmt");
                    let mut t_pcofold = Blake3Transcript::new(b"bench::side2::pcofold");
                    let discharge = prove_discharge(
                        &pcc_params,
                        &pcc_stmts,
                        &pcc_wits,
                        &mut t_fsmt,
                        &mut t_pcofold,
                    );

                    // 3. Root discharge: RokP + 1 lincheck.
                    let mut t_rok = Blake3Transcript::new(b"bench::side2::rok");
                    let (p_stmt, _, _, _) =
                        RokP::reduce(&setup.abc_params, &root.0, &root.1, &mut t_rok);
                    let mut t_lin = Blake3Transcript::new(b"bench::side2::lin");
                    let proof = InnerLincheck::prove(
                        &setup.srs,
                        &setup.index,
                        &p_stmt,
                        &mut t_lin,
                    );
                    let _ = black_box((proof, discharge.opening));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Isolated single-lincheck baseline (constant in K).
fn bench_single_lincheck(c: &mut Criterion) {
    let setup = build_setup(N);
    let claims = synthetic_claims(&setup, 1);
    let claim = &claims[0];
    c.bench_function("single_lincheck", |b| {
        b.iter(|| {
            let mut t = Blake3Transcript::new(b"bench::single");
            let proof = InnerLincheck::prove(&setup.srs, &setup.index, claim, &mut t);
            black_box(proof);
        });
    });
}

/// Isolated amortization-only cost (AbcFold + RokP, no lincheck).
fn bench_amortize_only(c: &mut Criterion) {
    let setup = build_setup(N);
    let mut group = c.benchmark_group("amortize_only");
    for &k in K_VALUES_AMORTIZED {
        let leaves = random_leaves(&setup, k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter_batched(
                || leaves.clone(),
                |leaves| {
                    let mut t_fold = Blake3Transcript::new(b"bench::amortize::fold");
                    let (root, _, _) =
                        AbcFold::prove(&setup.abc_params, leaves, &mut t_fold);
                    let mut t_rok = Blake3Transcript::new(b"bench::amortize::rok");
                    let (p_stmt, _, _, _) =
                        RokP::reduce(&setup.abc_params, &root.0, &root.1, &mut t_rok);
                    black_box(p_stmt);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_single_lincheck,
    bench_side_1,
    bench_amortize_only,
    bench_side_2,
);
criterion_main!(benches);
