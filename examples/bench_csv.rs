//! CSV-emitting benchmark for the amortization vs. independent-lincheck
//! comparison. Sweeps `(n, K)` and writes one row per `(n, k, side)` with
//! both prover and verifier timings.
//!
//! Usage:
//!   cargo run --release --example bench_csv -- \
//!       --ns 8,16,32 --ks 1,2,4,8,16,32 --reps 5 --out results.csv
//!
//! Default: --ns 8,16 --ks 1,2,4,8,16 --reps 5, output to stdout.
//!
//! Delegation framing: K local parties hand their computation to a Server.
//!
//! Side 1 — no amortization:
//!   * Prover (Server): K independent lincheck proofs (one per delegated job).
//!   * Verifier (per local party): 1 lincheck verification (their own).
//!
//! Side 2 — amortization (full sound pipeline, per paper §5 `lfsc`):
//!   * Prover (Server):
//!       1. AbcFold: K leaves → root R_{A,B,C} + K·κ R_PCC fold promises.
//!       2. Promise discharge: FsMt(Π_PC ∘ Π_DT) reduces every leaf's promise
//!          bundle to one R_PCO claim at a shared point x; PcoFold folds the K
//!          claims into one; a single KZG opening proves it.
//!       3. Root discharge: RokP (root → R_P, with a self-contained R_PCO
//!          opening in its proof) + 1 Marlin lincheck on the R_P claim.
//!   * Verifier (per local party): 1 fold path-verify (recovering this leaf's
//!     promises) + 1 FsMtPcc verify + 1 PcoFold path-verify + 1 final KZG
//!     opening check + 1 RokP verify + 1 lincheck verify.
//!
//! The K-1 verifiers from Side 1 that "disappear" in Side 2 are not modeled —
//! in the delegation setting they're absorbed into the Server's job. So the
//! Verifier column is per-verifier (one of the K local parties), independent
//! of K up to a `log K` term that comes from the path length.
//!
//! K=1 special case: no fold and hence no promises; run RokP directly on the
//! leaf (no path, no promise discharge).

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use ark_bls12_381::Fr;
use ark_ff::UniformRand;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::test_rng;

use amortized_proofs_masters_thesis::marlin_ahp::inner_lincheck::{
    InnerLincheck, InnerLincheckIndex,
};
use amortized_proofs_masters_thesis::pc;
use amortized_proofs_masters_thesis::reductions::abc_fold::AbcFold;
use amortized_proofs_masters_thesis::reductions::discharge::prove_discharge;
use amortized_proofs_masters_thesis::reductions::fsmt_pcc::FsMtPcc;
use amortized_proofs_masters_thesis::reductions::pco_fold::PcoFold;
use amortized_proofs_masters_thesis::reductions::rok_p::RokP;
use amortized_proofs_masters_thesis::relations::abc::{
    leaf_instance, AbcParams, AbcStatement, AbcWitness,
};
use amortized_proofs_masters_thesis::relations::p::PStatement;
use amortized_proofs_masters_thesis::relations::pcc::{PccParams, PccStatement, PccWitness};
use amortized_proofs_masters_thesis::transcript::Blake3Transcript;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    ns: Vec<usize>,
    ks: Vec<usize>,
    reps: usize,
    out: Option<String>,
}

fn parse_args() -> Args {
    let mut ns = vec![8usize, 16];
    let mut ks = vec![1usize, 2, 4, 8, 16];
    let mut reps = 5usize;
    let mut out: Option<String> = None;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--ns" => {
                ns = it.next().expect("--ns needs a value")
                    .split(',').map(|s| s.parse().expect("bad n")).collect();
            }
            "--ks" => {
                ks = it.next().expect("--ks needs a value")
                    .split(',').map(|s| s.parse().expect("bad k")).collect();
            }
            "--reps" => {
                reps = it.next().expect("--reps needs a value")
                    .parse().expect("bad reps");
            }
            "--out" => {
                out = Some(it.next().expect("--out needs a path"));
            }
            "--help" | "-h" => {
                eprintln!("bench_csv [--ns 8,16,32] [--ks 1,2,4,8] [--reps 5] [--out file.csv]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }
    Args { ns, ks, reps, out }
}

// ---------------------------------------------------------------------------
// Setup + workload helpers
// ---------------------------------------------------------------------------

struct Setup {
    srs: pc::Srs,
    abc_params: AbcParams,
    index: InnerLincheckIndex,
    n: usize,
}

/// Size the SRS for the **largest** `(n, K)` in this run. The binding
/// constraints are:
///   * AbcFold's root `N_u(X)` reaches degree `n + K − 1` after κ = log₂(K)
///     fold steps (each step multiplies one N by a T, growing degree by
///     deg(T) = 1, 2, 4, … = 2^level).
///   * Lincheck's arithmetization polynomials (`row, col, val_M, row_col`)
///     have degree `|K_matrix| − 1`, where `|K_matrix| =
///     next_pow2(joint_support(A, B, C))`.
///
/// `pc::setup(max_degree)` produces an SRS with strict bound `max_degree + 1`,
/// so we need `max_degree ≥ max(n + K − 1, |K_matrix| − 1)`.
fn build_setup(n: usize, max_k: usize) -> Setup {
    let rng = &mut test_rng();
    let a: Vec<(usize, usize, Fr)> =
        (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
    let b: Vec<(usize, usize, Fr)> =
        (0..n).map(|i| (i, (i + 1) % n, Fr::rand(rng))).collect();
    let c: Vec<(usize, usize, Fr)> =
        (0..n).map(|i| (i, i, Fr::rand(rng))).collect();

    let joint_supp = {
        use std::collections::BTreeSet;
        let mut s: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &(i, j, _) in &a { s.insert((i, j)); }
        for &(i, j, _) in &b { s.insert((i, j)); }
        for &(i, j, _) in &c { s.insert((i, j)); }
        s.len()
    };
    let k_matrix = joint_supp.max(2).next_power_of_two();
    let max_deg =
        (n + max_k.saturating_sub(1)).max(k_matrix.saturating_sub(1));
    let srs = pc::setup(max_deg, rng);

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
            let eval = |m: &[(usize, usize, Fr)]| -> Fr {
                m.iter()
                    .map(|&(i, j, v)| v * lambda_alpha[i] * lambda_beta[j])
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

fn random_leaves(setup: &Setup, k: usize) -> Vec<(AbcStatement, AbcWitness)> {
    let rng = &mut test_rng();
    (0..k)
        .map(|_| leaf_instance(&setup.abc_params, Fr::rand(rng), Fr::rand(rng)))
        .collect()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

// ---------------------------------------------------------------------------
// Sides
// ---------------------------------------------------------------------------

/// Returns (server_prove_ms, per_verifier_verify_ms) for Side 1.
/// Server does K lincheck proofs; each of the K local verifiers does 1 verify
/// — we measure one representative.
fn time_side_1(setup: &Setup, claims: &[PStatement]) -> (f64, f64) {
    let t0 = Instant::now();
    let mut proofs = Vec::with_capacity(claims.len());
    for c in claims {
        let mut t = Blake3Transcript::new(b"bench::s1");
        proofs.push(InnerLincheck::prove(&setup.srs, &setup.index, c, &mut t));
    }
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Per-verifier: pick the first verifier as representative.
    let t0 = Instant::now();
    let mut t = Blake3Transcript::new(b"bench::s1");
    assert!(InnerLincheck::verify(&setup.srs, &setup.index, &claims[0], &proofs[0], &mut t));
    let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;

    (prove_ms, verify_ms)
}

/// Returns (server_prove_ms, per_verifier_verify_ms) for Side 2.
///
/// Server (K > 1): AbcFold → discharge the K·κ fold promises via FsMtPcc +
/// PcoFold + one KZG opening → RokP + 1 lincheck on the root. Each of the K
/// local verifiers checks its own fold path, its promise discharge (FsMtPcc +
/// PcoFold + the shared opening), the RokP reduction, and the lincheck — we
/// measure one representative (leaf 0). At K = 1 there is no fold and no
/// promise to discharge, so only RokP + 1 lincheck run.
fn time_side_2(setup: &Setup, leaves: Vec<(AbcStatement, AbcWitness)>) -> (f64, f64) {
    let k = leaves.len();
    let leaf_stmts: Vec<AbcStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();
    let pcc_params = PccParams { srs: setup.srs.clone() };

    // -------------------- Prover (Server) --------------------
    let t0 = Instant::now();

    // 1. Fold K leaves into the root R_{A,B,C}, collecting the per-leaf R_PCC
    //    promises (κ per leaf).
    let (root_stmt, root_wit, paths, pcc_bundles): (
        AbcStatement,
        AbcWitness,
        Vec<_>,
        Vec<Vec<(PccStatement, PccWitness)>>,
    ) = if k == 1 {
        let (s, w) = leaves.into_iter().next().unwrap();
        (s, w, Vec::new(), Vec::new())
    } else {
        let mut t = Blake3Transcript::new(b"bench::s2::fold");
        let (root, paths, bundles) = AbcFold::prove(&setup.abc_params, leaves, &mut t);
        (root.0, root.1, paths, bundles)
    };

    // 2. Discharge the promises: FsMt(Π_PC ∘ Π_DT) → K R_PCO claims at a shared
    //    point x; PcoFold → 1 root R_PCO; one KZG opening proves it. A single
    //    root witness is assembled directly (see `reductions::discharge`), so
    //    the K per-leaf witnesses are never materialized. (No-op at K = 1,
    //    where there are no promises.)
    let discharge = if k == 1 {
        None
    } else {
        let pcc_stmts: Vec<Vec<PccStatement>> = pcc_bundles
            .iter()
            .map(|b| b.iter().map(|(s, _)| s.clone()).collect())
            .collect();
        let pcc_wits: Vec<Vec<PccWitness>> = pcc_bundles
            .into_iter()
            .map(|b| b.into_iter().map(|(_, w)| w).collect())
            .collect();

        let mut t_fsmt = Blake3Transcript::new(b"bench::s2::fsmt");
        let mut t_pcofold = Blake3Transcript::new(b"bench::s2::pcofold");
        Some(prove_discharge(
            &pcc_params,
            &pcc_stmts,
            &pcc_wits,
            &mut t_fsmt,
            &mut t_pcofold,
        ))
    };

    // 3. Root discharge: RokP (root → R_P, with the R_PCO opening self-contained
    //    in its proof) + one Marlin lincheck on the R_P claim.
    let mut t_rok = Blake3Transcript::new(b"bench::s2::rok");
    let (p_stmt, _pco_stmt, _pco_wit, rok_proof) =
        RokP::reduce(&setup.abc_params, &root_stmt, &root_wit, &mut t_rok);
    let mut t_lin = Blake3Transcript::new(b"bench::s2::lin");
    let lincheck_proof =
        InnerLincheck::prove(&setup.srs, &setup.index, &p_stmt, &mut t_lin);
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // -------------------- Verifier (one local party, leaf 0) --------------------
    let t0 = Instant::now();

    // 1. Fold path-verify → recover this leaf's κ promise statements.
    let promises_0 = if paths.is_empty() {
        Vec::new()
    } else {
        let mut t = Blake3Transcript::new(b"bench::s2::fold");
        AbcFold::verify(&setup.abc_params, 0, &leaf_stmts[0], &paths[0], &root_stmt, &mut t)
            .expect("AbcFold verify")
    };

    // 2. Discharge-side verify: FsMtPcc reconstructs this leaf's R_PCO claim at
    //    x, PcoFold checks it folds into the shared root, and the final KZG
    //    opening is checked once.
    if let Some(d) = &discharge {
        let mut t_fsmtv = Blake3Transcript::new(b"bench::s2::fsmt");
        let leaf_pco_0 =
            FsMtPcc::verify(&pcc_params, 0, &promises_0, &d.fsmt_proofs[0], &mut t_fsmtv)
                .expect("FsMtPcc verify");

        let mut t_pcofoldv = Blake3Transcript::new(b"bench::s2::pcofold");
        assert!(PcoFold::verify(0, &leaf_pco_0, &d.pco_paths[0], &d.root_stmt, &mut t_pcofoldv));

        assert!(pc::verify(
            &setup.srs,
            &d.root_stmt.commitment,
            d.root_stmt.point,
            d.root_stmt.value,
            &d.opening,
        ));
    }

    // 3. Root discharge: RokP verify + 1 lincheck verify.
    let mut t_rok_v = Blake3Transcript::new(b"bench::s2::rok");
    let _ = RokP::verify(&setup.abc_params, &root_stmt, &rok_proof, &mut t_rok_v)
        .expect("RokP verify");
    let mut t_lin_v = Blake3Transcript::new(b"bench::s2::lin");
    assert!(InnerLincheck::verify(
        &setup.srs,
        &setup.index,
        &p_stmt,
        &lincheck_proof,
        &mut t_lin_v,
    ));
    let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;

    (prove_ms, verify_ms)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();
    let max_k = *args.ks.iter().max().expect("--ks must be non-empty");

    let mut rows: Vec<String> = Vec::new();
    rows.push("n,k,side,prove_ms,verify_ms".to_string());

    for &n in &args.ns {
        let setup = build_setup(n, max_k);
        for &k in &args.ks {
            // --- Side 1: K linchecks
            let claims = synthetic_claims(&setup, k);
            let _ = time_side_1(&setup, &claims); // warmup
            let mut p_v: Vec<f64> = Vec::with_capacity(args.reps);
            let mut v_v: Vec<f64> = Vec::with_capacity(args.reps);
            for _ in 0..args.reps {
                let (p, v) = time_side_1(&setup, &claims);
                p_v.push(p);
                v_v.push(v);
            }
            rows.push(format!(
                "{n},{k},side1,{:.3},{:.3}",
                median(p_v),
                median(v_v)
            ));

            // --- Side 2: AbcFold + RokP + 1 lincheck
            let leaves = random_leaves(&setup, k);
            let _ = time_side_2(&setup, leaves.clone()); // warmup
            let mut p_v: Vec<f64> = Vec::with_capacity(args.reps);
            let mut v_v: Vec<f64> = Vec::with_capacity(args.reps);
            for _ in 0..args.reps {
                let (p, v) = time_side_2(&setup, leaves.clone());
                p_v.push(p);
                v_v.push(v);
            }
            rows.push(format!(
                "{n},{k},side2,{:.3},{:.3}",
                median(p_v),
                median(v_v)
            ));

            eprintln!("n={n:>4} k={k:>3} done");
        }
    }

    // Trailing newline so the file is a well-formed CSV and stays parseable
    // when concatenated with other runs (each row on its own physical line).
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
