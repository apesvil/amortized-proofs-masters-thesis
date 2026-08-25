//! Splits a full Marlin proving time into its witness-dependent and
//! witness-independent halves, over both problem size `n` and matrix density.
//!
//! ```text
//!   round 1  Az, Bz, ŵ, ẑ_A, ẑ_B [, mask]   witness-dependent
//!   round 2  t, g_1, h_1                     witness-dependent (except t)
//!   round 3  g_2, h_2                        witness-INdependent: the inner
//!                                            sumcheck, and the only part that
//!                                            can be delegated, since it is a
//!                                            claim about the public matrices
//! ```
//!
//! Two components are reported separately because they scale with the number of
//! non-zeros rather than with `n`, which is what the density sweep is about:
//!
//! * `matvec_ms` — the `Az`, `Bz` products. Included in `round1_ms`; broken out
//!   for visibility. These are prover work (the `ẑ_A`, `ẑ_B` oracles need
//!   them); producing the witness `z` is *not*, and stays untimed.
//! * `round2_t_ms` — `t(X)`, which depends only on `α` and the matrices, never
//!   on the witness. So the honest witness-independent share is not simply
//!   "round 3".
//!
//! Both are measured by re-running the identical computation rather than by
//! instrumenting the library, and are calibrated: work too short to time once
//! is repeated until the sample is long enough to be stable.
//!
//! # Density and `|K|`
//!
//! `--densities d` puts `d·n` non-zeros across `(A, B, C)` jointly. The indexer
//! domain is `|K| = next_pow2(d·n)`, so `d = 3` and `d = 4` share `|K| = 4n`,
//! as do `d = 5` and `d = 6` at `8n`. Round 3 tracks `|K|` and is therefore a
//! step function of density, while `matvec_ms` and `round2_t_ms` track the true
//! `d·n`. Both `density` and `k_size` are emitted so the two can be told apart.
//!
//! # SRS
//!
//! `D` is **fixed across the whole run** at `max(4n, next_pow2(max density)·n)`
//! — large enough for the ZK mask (degree `3n − 1`) and for the largest `|K|`
//! in the sweep. Letting `D` follow each density would confound density with
//! SRS size, since each round carries one degree-`D` shifted commitment and one
//! degree-`D` batched opening. That symmetry is also why the choice barely
//! moves the split: measured at `D = 2n` versus `D = 4n`, the
//! witness-independent share shifts by about one point.
//!
//! Because `D` is fixed per run, rows from runs with different `--densities`
//! (hence different `D`) are not directly comparable; keep a sweep in one file.
//!
//! `--srs-mult 2 --zk off` reproduces the `D = 2n` the other benchmarks use, so
//! `round3_ms` can be checked against `p_local_lincheck_ms` in
//! `results_verifier.csv`.
//!
//! Usage:
//!   cargo run --release --example bench_marlin_split -- \
//!       --ns 1024,4096,16384,65536,262144 --densities 2 --reps 3 \
//!       --zk both --out results_marlin_split.csv
//!
//!   cargo run --release --example bench_marlin_split -- \
//!       --ns 4096,16384,65536 --densities 2,3,4,5,6,8 --reps 3 \
//!       --zk both --out results_marlin_density.csv

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use ark_bls12_381::Fr;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::test_rng;

use amortized_proofs_masters_thesis::marlin_ahp::bivariate::UnnormalizedBivariateLagrangePoly;
use amortized_proofs_masters_thesis::marlin_ahp::inner_lincheck::{
    InnerLincheck, InnerLincheckIndex,
};
use amortized_proofs_masters_thesis::marlin_ahp::outer_sumcheck::{
    calculate_t, repo_p_statement, OuterSumcheck,
};
use amortized_proofs_masters_thesis::marlin_ahp::r1cs::{
    satisfiable_instance, R1csInstance,
};
use amortized_proofs_masters_thesis::pc;
use amortized_proofs_masters_thesis::transcript::Blake3Transcript;

/// How long a calibrated component measurement should sample for.
const CALIBRATION_MS: u128 = 5;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    ns: Vec<usize>,
    densities: Vec<usize>,
    reps: usize,
    zk: Vec<bool>,
    srs_mult: Option<usize>,
    out: Option<String>,
}

fn parse_args() -> Args {
    let mut ns = vec![1024usize, 4096, 16384, 65536];
    let mut densities = vec![2usize];
    let mut reps = 3usize;
    let mut zk = vec![false, true];
    let mut srs_mult: Option<usize> = None;
    let mut out: Option<String> = None;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--ns" => {
                ns = it.next().expect("--ns needs a value")
                    .split(',').map(|s| s.parse().expect("bad n")).collect();
            }
            "--densities" => {
                densities = it.next().expect("--densities needs a value")
                    .split(',').map(|s| s.parse().expect("bad density")).collect();
            }
            "--reps" => {
                reps = it.next().expect("--reps needs a value")
                    .parse().expect("bad reps");
            }
            "--zk" => {
                zk = match it.next().expect("--zk needs a value").as_str() {
                    "both" => vec![false, true],
                    "on" | "true" => vec![true],
                    "off" | "false" => vec![false],
                    other => panic!("--zk expects both|on|off, got {other}"),
                };
            }
            "--srs-mult" => {
                srs_mult = Some(it.next().expect("--srs-mult needs a value")
                    .parse().expect("bad srs-mult"));
            }
            "--out" => out = Some(it.next().expect("--out needs a path")),
            "--help" | "-h" => {
                eprintln!(
                    "bench_marlin_split [--ns 1024,...] [--densities 2,3,...] \
                     [--reps 3] [--zk both|on|off] [--srs-mult N] [--out f.csv]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }
    assert!(!densities.is_empty(), "--densities must be non-empty");
    Args { ns, densities, reps, zk, srs_mult, out }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

/// Time `f`, repeating it until the sample is long enough to be stable.
///
/// The budget is a **floor on sampling time, not a deadline**: the loop tests
/// between iterations and never interrupts one, so work that already exceeds
/// the budget simply runs once and is reported as a single measurement. The
/// warmup matters at those sizes — without it the lone sample would be
/// cold-cache while the round it is a component of is warm.
fn calibrated_ms(mut f: impl FnMut()) -> f64 {
    f();
    let t0 = Instant::now();
    let mut iters = 0u32;
    loop {
        f();
        iters += 1;
        if t0.elapsed().as_millis() >= CALIBRATION_MS {
            break;
        }
    }
    ms_since(t0) / f64::from(iters)
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

struct Setup {
    index: InnerLincheckIndex,
    domain_h: Radix2EvaluationDomain<Fr>,
    inst: R1csInstance,
}

fn build_setup(srs: &pc::Srs, n: usize, density: usize) -> Setup {
    let rng = &mut test_rng();
    let inst = satisfiable_instance(n, density, rng);
    let index = InnerLincheck::index(
        srs, &inst.matrix_a, &inst.matrix_b, &inst.matrix_c, n,
    );
    let domain_h = Radix2EvaluationDomain::<Fr>::new(n).expect("n a power of two");
    Setup { index, domain_h, inst }
}

struct Row {
    round1: f64,
    matvec: f64,
    round2: f64,
    round2_t: f64,
    round3: f64,
}

fn measure(srs: &pc::Srs, setup: &Setup, zk: bool, reps: usize) -> Row {
    let rng = &mut test_rng();
    let Setup { index, domain_h, inst } = setup;

    // Sanity, untimed: the outer rounds must actually verify before we report
    // any timing taken from them.
    {
        let mut t_p = Blake3Transcript::new(b"split::check");
        let (proof, _) = OuterSumcheck::prove(srs, inst, domain_h, zk, rng, &mut t_p);
        let mut t_v = Blake3Transcript::new(b"split::check");
        assert!(
            OuterSumcheck::verify(srs, domain_h, &proof, &mut t_v).is_some(),
            "outer sumcheck must verify (zk = {zk})",
        );
    }

    let mut r1_v = Vec::with_capacity(reps);
    let mut r2_v = Vec::with_capacity(reps);
    let mut rt_v = Vec::with_capacity(reps);
    let mut r3_v = Vec::with_capacity(reps);
    let mut mv_v = Vec::with_capacity(reps);

    for rep in 0..=reps {
        let mut t_p = Blake3Transcript::new(b"split::bench");

        let t0 = Instant::now();
        let first =
            OuterSumcheck::first_round(srs, inst, domain_h, zk, rng, &mut t_p);
        let r1 = ms_since(t0);

        let t0 = Instant::now();
        let (outer_proof, claim) =
            OuterSumcheck::second_round(srs, inst, domain_h, first, &mut t_p);
        let r2 = ms_since(t0);

        // Components, re-run rather than instrumented so the library stays free
        // of timing hooks. Identical computations on the same inputs.
        let mv = calibrated_ms(|| {
            black_box(inst.matvec(&inst.matrix_a));
            black_box(inst.matvec(&inst.matrix_b));
        });
        let rt = calibrated_ms(|| {
            let r_alpha = domain_h
                .batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(
                    claim.alpha,
                );
            black_box(calculate_t(inst, domain_h, claim.eta, &r_alpha));
        });

        // Round 3 on the R_P claim at the same (α, β, η). Built outside the
        // timed region: `repo_p_statement` is harness glue, not prover work.
        // It re-derives `y` in this repo's normalization — see the seam note in
        // `outer_sumcheck`'s module docs.
        let p_stmt = repo_p_statement(inst, domain_h, &claim);

        let t0 = Instant::now();
        let mut t_lin = Blake3Transcript::new(b"split::lincheck");
        let lincheck_proof = InnerLincheck::prove(srs, index, &p_stmt, &mut t_lin);
        let r3 = ms_since(t0);

        let _ = black_box((
            outer_proof.batched_opening,
            lincheck_proof.batched_opening,
        ));

        if rep > 0 {
            r1_v.push(r1);
            r2_v.push(r2);
            rt_v.push(rt);
            r3_v.push(r3);
            mv_v.push(mv);
        }
    }

    Row {
        round1: median(r1_v),
        matvec: median(mv_v),
        round2: median(r2_v),
        round2_t: median(rt_v),
        round3: median(r3_v),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    // One SRS per n, shared by every density and ZK setting in the run, so that
    // D is never a confounder. |K| = next_pow2(d)·n for a power-of-two n, and
    // the ZK mask needs D ≥ 3n, hence the floor of 4.
    let max_density = *args.densities.iter().max().expect("non-empty");
    let srs_mult = args
        .srs_mult
        .unwrap_or_else(|| max_density.next_power_of_two().max(4));
    for &zk in &args.zk {
        assert!(
            !zk || srs_mult >= 4,
            "ZK needs D ≥ 3n for the mask polynomial; use --srs-mult 4 or more",
        );
    }
    eprintln!("SRS fixed at D = {srs_mult}n for the whole run");

    let mut rows: Vec<String> = Vec::new();
    rows.push(
        "n,density,k_size,zk,round1_ms,matvec_ms,round2_ms,round2_t_ms,\
         round3_ms,wd_ms,total_ms"
            .to_string(),
    );

    for &n in &args.ns {
        let srs = pc::setup(srs_mult * n - 1, &mut test_rng());
        for &density in &args.densities {
            let setup = build_setup(&srs, n, density);
            let k_size = setup.inst.k_size();
            for &zk in &args.zk {
                let r = measure(&srs, &setup, zk, args.reps);
                let wd = r.round1 + r.round2;
                let total = wd + r.round3;
                rows.push(format!(
                    "{n},{density},{k_size},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    if zk { 1 } else { 0 },
                    r.round1, r.matvec, r.round2, r.round2_t, r.round3, wd, total,
                ));
                eprintln!(
                    "n={n:>7} d={density} |K|={k_size:>8} zk={:<5} \
                     r1={:9.2} (mv={:7.2}) r2={:9.2} (t={:7.2}) r3={:9.2}  \
                     WD={:9.2} ({:.0}% of {:.2})",
                    zk, r.round1, r.matvec, r.round2, r.round2_t, r.round3,
                    wd, 100.0 * wd / total, total,
                );
            }
        }
    }

    let body = format!("{}\n", rows.join("\n"));
    match args.out {
        Some(path) => {
            File::create(&path)
                .expect("create output file")
                .write_all(body.as_bytes())
                .expect("write output");
            eprintln!("wrote {path}");
        }
        None => print!("{body}"),
    }
}
