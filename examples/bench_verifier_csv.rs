//! Verifier-side CSV benchmark for the two delegation figures.
//!
//! Unlike `bench_csv`, the Side-2 prover runs **once** per `(n, K)` and only the
//! verifier is re-timed, so a full sweep costs minutes rather than hours. The
//! local party's work is split into the stages the two figures need:
//!
//! ```text
//!   amortization      = statement preparation      (commit to N_u,T_u,N_v,T_v)
//!                     + fold path verify           (AbcFold::verify)
//!                     + promise discharge verify   (FsMtPcc + PcoFold + 1 KZG)
//!                     + RokP verify                (root R_{A,B,C} → R_P)
//!   folded-stmt check = 1 inner-lincheck verify on the root R_P claim
//! ```
//!
//! The split is drawn where the pipeline stops being SNARK-agnostic: everything
//! up to and including `RokP::verify` reads only the SRS and `n` (no matrices,
//! no `InnerLincheckIndex`), so it is independent of which SNARK discharges the
//! final R_P claim. Only the last stage is Marlin-specific.
//!
//! **Statement preparation is the client's share of `abc::leaf_instance`, not
//! all of it.** `leaf_instance` is an honest-instance generator: it returns the
//! statement *and* the witness. Its O(n) half — `u = λ(α)`, `v = λ(β)` and the
//! three bilinear forms `y_M = uᵀ·M·v` — is the witness plus the *answer being
//! delegated*, which the client neither knows nor computes; it picks `(α, β)`,
//! sends them, and gets the `y`s back with a proof. The client's own share is
//! the four sparse leaf polynomials and their commitments, which is O(1) group
//! operations. Charging it the full `leaf_instance` would mean charging it for
//! the very computation it is paying the Server to do.
//!
//! Figure 1 plots the amortization part alone. Figure 2 plots the local party's
//! total delegated cost against what it would pay to prove its own lincheck.
//!
//! Baselines for the no-delegation path are emitted too:
//!   * `p_local_lincheck_ms`   — `InnerLincheck::prove`: the plain Marlin
//!     lincheck prover. It is *not* charged a separate statement-preparation
//!     term: it builds `f` over `K` regardless and `y = Σ_κ f(κ)` falls out of
//!     that table (see the `debug_assert` in `inner_lincheck::prove`), so
//!     knowing `y` costs it `O(|K|)` additions it already pays for.
//!   * `p_local_rokp_ms`       — `RokP::reduce`, the prover step that turns its
//!     own R_{A,B,C} leaf into the R_P claim the lincheck then proves.
//!   * `ref_direct_eval_ms`    — reference only, no proof involved: evaluating
//!     `P_A(α,β), P_B(α,β), P_C(α,β)` directly (`λ(α)`, `λ(β)`, three bilinear
//!     forms). The floor for a party that only needs the value and has nobody
//!     to convince.
//!
//! Usage:
//!   cargo run --release --example bench_verifier_csv -- \
//!       --ns 1024,4096,16384,65536,262144 --ks 1,2,4,8,16,32,64,128 \
//!       --reps 9 --out results_verifier.csv

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use ark_bls12_381::Fr;
use ark_ff::UniformRand;
use ark_poly::{univariate::SparsePolynomial, EvaluationDomain, Radix2EvaluationDomain};
use ark_std::test_rng;

use amortized_proofs_masters_thesis::marlin_ahp::inner_lincheck::{
    InnerLincheck, InnerLincheckIndex,
};
use amortized_proofs_masters_thesis::pc::{self, Poly};
use amortized_proofs_masters_thesis::reductions::abc_fold::AbcFold;
use amortized_proofs_masters_thesis::reductions::discharge::prove_discharge;
use amortized_proofs_masters_thesis::reductions::fsmt_pcc::FsMtPcc;
use amortized_proofs_masters_thesis::reductions::pco_fold::PcoFold;
use amortized_proofs_masters_thesis::reductions::rok_p::RokP;
use amortized_proofs_masters_thesis::relations::abc::{
    leaf_instance, AbcParams, AbcStatement, AbcWitness,
};
use amortized_proofs_masters_thesis::relations::abc_leaf_pcc::{
    full_pcc_bundles, leaf_correctness_pcc,
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
    let mut ns = vec![1024usize, 4096, 16384, 65536, 262144];
    let mut ks = vec![1usize, 2, 4, 8, 16, 32, 64, 128];
    let mut reps = 9usize;
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
            "--out" => out = Some(it.next().expect("--out needs a path")),
            "--help" | "-h" => {
                eprintln!(
                    "bench_verifier_csv [--ns 1024,...] [--ks 1,2,...] [--reps 9] [--out f.csv]"
                );
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
// Setup + workload helpers (mirrors `bench_csv`, kept local so the two
// binaries can be re-tuned independently)
// ---------------------------------------------------------------------------

struct Setup {
    srs: pc::Srs,
    abc_params: AbcParams,
    index: InnerLincheckIndex,
    n: usize,
}

fn build_setup(n: usize, max_k: usize) -> Setup {
    let rng = &mut test_rng();
    let a: Vec<(usize, usize, Fr)> = (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
    let b: Vec<(usize, usize, Fr)> = (0..n).map(|i| (i, (i + 1) % n, Fr::rand(rng))).collect();
    let c: Vec<(usize, usize, Fr)> = (0..n).map(|i| (i, i, Fr::rand(rng))).collect();

    let joint_supp = {
        use std::collections::BTreeSet;
        let mut s: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &(i, j, _) in &a { s.insert((i, j)); }
        for &(i, j, _) in &b { s.insert((i, j)); }
        for &(i, j, _) in &c { s.insert((i, j)); }
        s.len()
    };
    let k_matrix = joint_supp.max(2).next_power_of_two();
    let max_deg = (n + max_k.saturating_sub(1)).max(k_matrix.saturating_sub(1));
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

/// One synthetic R_P claim `(α, β, y, η)` with `y` honestly computed — the
/// statement a local party would hand to a plain Marlin lincheck prover.
fn synthetic_claim(setup: &Setup, alpha: Fr, beta: Fr, eta: Fr) -> PStatement {
    let dom = Radix2EvaluationDomain::<Fr>::new(setup.n).unwrap();
    let lambda_alpha = dom.evaluate_all_lagrange_coefficients(alpha);
    let lambda_beta = dom.evaluate_all_lagrange_coefficients(beta);
    let eval = |m: &[(usize, usize, Fr)]| -> Fr {
        m.iter().map(|&(i, j, v)| v * lambda_alpha[i] * lambda_beta[j]).sum()
    };
    let p_a = eval(&setup.abc_params.matrix_a);
    let p_b = eval(&setup.abc_params.matrix_b);
    let p_c = eval(&setup.abc_params.matrix_c);
    let y = p_a + eta * p_b + eta * eta * p_c;
    PStatement { alpha, beta, y, eta }
}

fn random_leaves(
    setup: &Setup,
    k: usize,
) -> (Vec<(AbcStatement, AbcWitness)>, Vec<(Fr, Fr)>) {
    let rng = &mut test_rng();
    let mut leaves = Vec::with_capacity(k);
    let mut challenges = Vec::with_capacity(k);
    for _ in 0..k {
        let (alpha, beta) = (Fr::rand(rng), Fr::rand(rng));
        leaves.push(leaf_instance(&setup.abc_params, alpha, beta));
        challenges.push((alpha, beta));
    }
    (leaves, challenges)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

/// Median of `reps` timed calls, after one untimed warmup.
fn timed<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    black_box(f());
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let out = f();
        v.push(ms_since(t0));
        black_box(out);
    }
    median(v)
}

// ---------------------------------------------------------------------------
// K-independent costs (measured once per n)
// ---------------------------------------------------------------------------

struct PerN {
    /// The client's share of building its R_{A,B,C} leaf: the four sparse leaf
    /// polynomials and their commitments. Excludes the witness (`u`, `v`) and
    /// the delegated answers (`y_A`, `y_B`, `y_C`).
    stmt_prep: f64,
    /// `RokP::reduce` — the prover step from its own leaf to an R_P claim.
    local_rokp: f64,
    /// `InnerLincheck::prove` on that R_P claim.
    local_lincheck: f64,
    /// Reference: evaluating `P_A, P_B, P_C` at `(α, β)` directly, no proof.
    direct_eval: f64,
}

/// The client's half of `abc::leaf_instance`: the four sparse leaf polynomials
/// and their commitments — the part of the statement it can and must build
/// itself. Mirrors the shapes in `relations::abc::leaf_instance`; the witness
/// vectors and the bilinear forms are deliberately absent (see module docs).
fn client_statement_prep(setup: &Setup, alpha: Fr, beta: Fr) -> [pc::Comm; 4] {
    let dom = Radix2EvaluationDomain::<Fr>::new(setup.n).unwrap();
    let alpha_n_minus_1 = dom.evaluate_vanishing_polynomial(alpha);
    let beta_n_minus_1 = dom.evaluate_vanishing_polynomial(beta);
    let one = Fr::from(1u64);

    let n_u = SparsePolynomial::from_coefficients_vec(vec![
        (0, -alpha), (1, -alpha_n_minus_1), (setup.n, alpha),
    ]);
    let t_u = SparsePolynomial::from_coefficients_vec(vec![(0, -alpha), (1, one)]);
    let n_v = SparsePolynomial::from_coefficients_vec(vec![
        (0, -beta), (1, -beta_n_minus_1), (setup.n, beta),
    ]);
    let t_v = SparsePolynomial::from_coefficients_vec(vec![(0, -beta), (1, one)]);

    [
        pc::commit(&setup.srs, &Poly::Sparse(n_u)),
        pc::commit(&setup.srs, &Poly::Sparse(t_u)),
        pc::commit(&setup.srs, &Poly::Sparse(n_v)),
        pc::commit(&setup.srs, &Poly::Sparse(t_v)),
    ]
}

fn measure_per_n(setup: &Setup, reps: usize) -> PerN {
    let rng = &mut test_rng();
    let (alpha, beta) = (Fr::rand(rng), Fr::rand(rng));
    let eta = Fr::rand(rng);

    let stmt_prep = timed(reps, || client_statement_prep(setup, alpha, beta));
    let direct_eval = timed(reps, || synthetic_claim(setup, alpha, beta, eta));

    let (leaf_stmt, leaf_wit) = leaf_instance(&setup.abc_params, alpha, beta);
    let local_rokp = timed(reps, || {
        let mut t = Blake3Transcript::new(b"bench::local::rok");
        RokP::reduce(&setup.abc_params, &leaf_stmt, &leaf_wit, &mut t)
    });

    let claim = synthetic_claim(setup, alpha, beta, eta);
    let local_lincheck = timed(reps, || {
        let mut t = Blake3Transcript::new(b"bench::local::lin");
        InnerLincheck::prove(&setup.srs, &setup.index, &claim, &mut t)
    });

    PerN { stmt_prep, local_rokp, local_lincheck, direct_eval }
}

// ---------------------------------------------------------------------------
// Side-2 verifier, staged
// ---------------------------------------------------------------------------

struct Stages {
    fold: f64,
    discharge: f64,
    rokp: f64,
    lincheck: f64,
}

/// Runs the Side-2 prover **once**, then times one local party's (leaf 0)
/// verification `reps` times, split into its stages.
///
/// K = 1: no fold and no promises, so `fold = discharge = 0` — the client
/// delegates a single job and only checks RokP + the lincheck.
fn time_side2_verifier_stages(setup: &Setup, k: usize, reps: usize) -> Stages {
    let (leaves, challenges) = random_leaves(setup, k);
    let leaf_stmts: Vec<AbcStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();
    let pcc_params = PccParams { srs: setup.srs.clone() };

    // ---------------- Prover (Server) — once, untimed ----------------
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

    let discharge = if k == 1 {
        None
    } else {
        let full = full_pcc_bundles(&pcc_params, setup.n, &challenges, pcc_bundles);
        let pcc_stmts: Vec<Vec<PccStatement>> =
            full.iter().map(|b| b.iter().map(|(s, _)| s.clone()).collect()).collect();
        let pcc_wits: Vec<Vec<PccWitness>> =
            full.into_iter().map(|b| b.into_iter().map(|(_, w)| w).collect()).collect();
        let mut t_fsmt = Blake3Transcript::new(b"bench::s2::fsmt");
        let mut t_pcofold = Blake3Transcript::new(b"bench::s2::pcofold");
        Some(prove_discharge(&pcc_params, &pcc_stmts, &pcc_wits, &mut t_fsmt, &mut t_pcofold))
    };

    let mut t_rok = Blake3Transcript::new(b"bench::s2::rok");
    let (p_stmt, _pco_stmt, _pco_wit, rok_proof) =
        RokP::reduce(&setup.abc_params, &root_stmt, &root_wit, &mut t_rok);
    let mut t_lin = Blake3Transcript::new(b"bench::s2::lin");
    let lincheck_proof = InnerLincheck::prove(&setup.srs, &setup.index, &p_stmt, &mut t_lin);

    // ---------------- Verifier (local party, leaf 0) — reps + 1 warmup -------
    let mut fold_v = Vec::with_capacity(reps);
    let mut disc_v = Vec::with_capacity(reps);
    let mut rokp_v = Vec::with_capacity(reps);
    let mut lin_v = Vec::with_capacity(reps);

    for rep in 0..=reps {
        // (1) fold path verify, recovering this leaf's κ promises.
        let t0 = Instant::now();
        let promises_0 = if paths.is_empty() {
            Vec::new()
        } else {
            let mut t = Blake3Transcript::new(b"bench::s2::fold");
            AbcFold::verify(&setup.abc_params, 0, &leaf_stmts[0], &paths[0], &root_stmt, &mut t)
                .expect("AbcFold verify")
        };
        let fold_ms = ms_since(t0);

        // (2) promise discharge. Includes rebuilding this leaf's own
        //     encoding-correctness R_PCC — the verifier constructs it rather
        //     than receiving it — then FsMtPcc, PcoFold, and one KZG check.
        let t0 = Instant::now();
        if let Some(d) = &discharge {
            let mut promises = promises_0;
            promises.push(
                leaf_correctness_pcc(&pcc_params, setup.n, challenges[0].0, challenges[0].1).0,
            );
            let mut t_fsmtv = Blake3Transcript::new(b"bench::s2::fsmt");
            let leaf_pco_0 =
                FsMtPcc::verify(&pcc_params, 0, &promises, &d.fsmt_proofs[0], &mut t_fsmtv)
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
        let discharge_ms = ms_since(t0);

        // (3) RokP verify — last SNARK-agnostic stage.
        let t0 = Instant::now();
        let mut t_rok_v = Blake3Transcript::new(b"bench::s2::rok");
        let out = RokP::verify(&setup.abc_params, &root_stmt, &rok_proof, &mut t_rok_v)
            .expect("RokP verify");
        let rokp_ms = ms_since(t0);
        black_box(out);

        // (4) the one inner lincheck on the folded R_P claim.
        let t0 = Instant::now();
        let mut t_lin_v = Blake3Transcript::new(b"bench::s2::lin");
        assert!(InnerLincheck::verify(
            &setup.srs, &setup.index, &p_stmt, &lincheck_proof, &mut t_lin_v,
        ));
        let lincheck_ms = ms_since(t0);

        if rep > 0 {
            fold_v.push(fold_ms);
            disc_v.push(discharge_ms);
            rokp_v.push(rokp_ms);
            lin_v.push(lincheck_ms);
        }
    }

    Stages {
        fold: median(fold_v),
        discharge: median(disc_v),
        rokp: median(rokp_v),
        lincheck: median(lin_v),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();
    let max_k = *args.ks.iter().max().expect("--ks must be non-empty");

    let mut rows: Vec<String> = Vec::new();
    rows.push(
        "n,k,v_stmt_prep_ms,v_fold_ms,v_discharge_ms,v_rokp_ms,v_lincheck_ms,\
         v_amort_ms,v_total_ms,\
         p_local_rokp_ms,p_local_lincheck_ms,ref_direct_eval_ms"
            .to_string(),
    );

    for &n in &args.ns {
        let setup = build_setup(n, max_k);
        // These four are K-independent: measure once per n, repeat per row so
        // the CSV stays self-contained.
        let per_n = measure_per_n(&setup, args.reps);
        eprintln!(
            "n={n:>7}  stmt_prep={:.3}  local: rokp={:.3} lincheck={:.3}  \
             (direct eval, no proof: {:.3}) (ms)",
            per_n.stmt_prep, per_n.local_rokp, per_n.local_lincheck, per_n.direct_eval
        );

        for &k in &args.ks {
            let s = time_side2_verifier_stages(&setup, k, args.reps);
            let amort = per_n.stmt_prep + s.fold + s.discharge + s.rokp;
            let total = amort + s.lincheck;
            rows.push(format!(
                "{n},{k},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                per_n.stmt_prep, s.fold, s.discharge, s.rokp, s.lincheck,
                amort, total,
                per_n.local_rokp, per_n.local_lincheck, per_n.direct_eval,
            ));
            eprintln!(
                "n={n:>7} k={k:>4}  amort={amort:8.3}  total={total:8.3}  \
                 (prep {:.3} / fold {:.3} / disch {:.3} / rokp {:.3} | lin {:.3})",
                per_n.stmt_prep, s.fold, s.discharge, s.rokp, s.lincheck
            );
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
