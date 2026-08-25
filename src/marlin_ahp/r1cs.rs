//! A satisfiable R1CS instance for the Marlin outer sumcheck.
//!
//! The rest of this crate never needs a witness: `R_P` is a claim about the
//! *public* matrices, so `AbcParams` carries `(row, col, val)` triples with
//! random values and no assignment. Marlin's rounds 1-2, by contrast, are the
//! witness-dependent half of the prover and need a genuine `z` with
//! `Az ∘ Bz = Cz` — which random matrices almost surely do not admit.
//!
//! Layout follows `arithmetize.rs`: matrices index `z` directly over `H`,
//! rather than going through a `ConstraintSynthesizer` with public inputs
//! interleaved at multiples of `|H|/|X|`. There is a single public input
//! `z[0] = 1`, so `|X| = 1`, `domain_x = {1}` and `v_X(X) = X − 1`.
//!
//! # Density
//!
//! `density = d` puts `d` non-zeros per row across the three matrices jointly,
//! for `d·n` non-zeros in total. Rows are banded: row `i` holds entries at
//! columns `(i + o) mod n` for a set of offsets,
//!
//! ```text
//!   O_A = {0, 1, …, ⌈d/2⌉ − 1}      O_B = {⌈d/2⌉, …, d − 1}      O_C = {0}
//! ```
//!
//! `C` stays diagonal and `0 ∈ O_A`, so `supp(C) ⊆ supp(A)` and the joint
//! support is exactly `d·n`. `d = 2` reproduces the `A` diagonal / `B`
//! off-by-one shape the other benchmarks use, drawing from the rng in the same
//! order, so instances at `d = 2` are bit-identical to the earlier ones.
//!
//! Note that the indexer domain is `|K| = next_pow2(d·n)`, so `d = 3` and
//! `d = 4` share `|K| = 4n`, as do `d = 5` and `d = 6` at `8n`. Round 3's cost
//! tracks `|K|`, while `calculate_t` and the `Az`/`Bz` products track the true
//! `d·n` — the two move differently and both are reported.

use ark_bls12_381::Fr;
use ark_ff::{One, UniformRand, Zero};
use ark_std::rand::RngCore;

/// An R1CS instance together with a satisfying assignment.
#[derive(Clone, Debug)]
pub struct R1csInstance {
    pub matrix_a: Vec<(usize, usize, Fr)>,
    pub matrix_b: Vec<(usize, usize, Fr)>,
    pub matrix_c: Vec<(usize, usize, Fr)>,
    /// Assignment over `H`; `z[0] = 1` is the public input.
    ///
    /// This is an **input** to the prover — producing it is the work of running
    /// the computation being proven, not of proving it. `Az` and `Bz` are a
    /// different matter and are deliberately *not* cached here: see [`matvec`].
    ///
    /// [`matvec`]: R1csInstance::matvec
    pub z: Vec<Fr>,
    pub n: usize,
    /// Non-zeros per row across `(A, B, C)` jointly; `d·n` in total.
    pub density: usize,
}

/// Build a satisfiable instance of the given density (see the module docs for
/// the banded shape).
///
/// `A` and `B` get random values, but `C` is **derived** rather than drawn:
/// ```text
///   c_ii = (Az)_i · (Bz)_i / z_i          (z_i ≠ 0 by construction)
/// ```
/// so that `Az ∘ Bz = Cz` holds by construction. Deriving `C` is what makes the
/// instance satisfiable at all, and keeping `C` diagonal is what keeps the
/// joint support at exactly `d·n`.
///
/// (`(Az)_i = 0` has probability ~`1/|F|` and would store an explicit zero at a
/// supported position — harmless, and not worth a guard.)
pub fn satisfiable_instance(
    n: usize,
    density: usize,
    rng: &mut impl RngCore,
) -> R1csInstance {
    assert!(n.is_power_of_two() && n >= 2, "n must be a power of two ≥ 2");
    assert!(
        density >= 2,
        "density ≥ 2: A needs at least one offset and B a different one",
    );
    assert!(density <= n, "offsets must be distinct mod n");

    let a_offsets: Vec<usize> = (0..density.div_ceil(2)).collect();
    let b_offsets: Vec<usize> = (density.div_ceil(2)..density).collect();

    // z[0] = 1 is the public input; the rest is the witness, non-zero so the
    // division below is well defined.
    let mut z: Vec<Fr> = Vec::with_capacity(n);
    z.push(Fr::one());
    for _ in 1..n {
        let mut v = Fr::rand(rng);
        while v.is_zero() {
            v = Fr::rand(rng);
        }
        z.push(v);
    }

    // Row-major over (row, offset). At d = 2 there is one offset per matrix, so
    // this draws in the same order as the original diagonal/off-diagonal
    // generator and reproduces its instances exactly.
    let build = |offsets: &[usize], rng: &mut dyn RngCore| -> Vec<(usize, usize, Fr)> {
        let mut m = Vec::with_capacity(offsets.len() * n);
        for i in 0..n {
            for &o in offsets {
                m.push((i, (i + o) % n, Fr::rand(rng)));
            }
        }
        m
    };
    let matrix_a = build(&a_offsets, rng);
    let matrix_b = build(&b_offsets, rng);

    // Partial instance, so the products can go through the same `matvec` the
    // prover uses rather than a shape-specific shortcut.
    let mut inst = R1csInstance {
        matrix_a,
        matrix_b,
        matrix_c: Vec::new(),
        z,
        n,
        density,
    };
    let z_a = inst.matvec(&inst.matrix_a);
    let z_b = inst.matvec(&inst.matrix_b);
    inst.matrix_c = (0..n).map(|i| (i, i, z_a[i] * z_b[i] / inst.z[i])).collect();

    debug_assert!(inst.is_satisfied(), "generated instance must satisfy R1CS");
    debug_assert_eq!(inst.joint_support(), density * n);
    inst
}

impl R1csInstance {
    /// `M·z`.
    ///
    /// **Prover work, deliberately not cached.** These products exist only
    /// because the AHP needs the `ẑ_A`, `ẑ_B` oracles, so they belong inside
    /// round 1 and inside its timing. Caching them on the instance would hand
    /// the witness-dependent half a chunk of its `O(nnz)` work for free — an
    /// error that grows with density, which is exactly the axis
    /// `bench_marlin_split` sweeps. Upstream computes them in `prover_init`,
    /// which also runs the constraint synthesizer; we keep synthesis out (that
    /// is the circuit's cost, not the proof system's) and the products in.
    pub fn matvec(&self, m: &[(usize, usize, Fr)]) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); self.n];
        for &(i, j, v) in m {
            out[i] += v * self.z[j];
        }
        out
    }

    /// `Az ∘ Bz = Cz`.
    pub fn is_satisfied(&self) -> bool {
        let z_a = self.matvec(&self.matrix_a);
        let z_b = self.matvec(&self.matrix_b);
        let z_c = self.matvec(&self.matrix_c);
        (0..self.n).all(|i| z_a[i] * z_b[i] == z_c[i])
    }

    /// Size of the joint support of `(A, B, C)` — the indexer domain before
    /// rounding up to a power of two. `density · n` for the shape built above.
    pub fn joint_support(&self) -> usize {
        use std::collections::BTreeSet;
        let mut s: BTreeSet<(usize, usize)> = BTreeSet::new();
        for m in [&self.matrix_a, &self.matrix_b, &self.matrix_c] {
            for &(i, j, _) in m {
                s.insert((i, j));
            }
        }
        s.len()
    }

    /// The indexer domain size actually used: `next_pow2(joint support)`.
    pub fn k_size(&self) -> usize {
        self.joint_support().max(2).next_power_of_two()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::test_rng;

    #[test]
    fn instance_is_satisfiable() {
        for n in [8usize, 16, 32] {
            for d in 2..=6 {
                let inst = satisfiable_instance(n, d, &mut test_rng());
                assert!(inst.is_satisfied(), "n = {n}, d = {d}");
            }
        }
    }

    #[test]
    fn public_input_is_one() {
        let inst = satisfiable_instance(8, 2, &mut test_rng());
        assert_eq!(inst.z[0], Fr::one());
    }

    /// The whole point of keeping C diagonal and inside supp(A): the joint
    /// support — and hence |K| — must be exactly the requested density.
    #[test]
    fn joint_support_is_density_times_n() {
        for n in [8usize, 16, 32] {
            for d in 2..=8 {
                let inst = satisfiable_instance(n, d, &mut test_rng());
                assert_eq!(inst.joint_support(), d * n, "n = {n}, d = {d}");
            }
        }
    }

    /// `|K|` is padded to a power of two, so d = 3 and d = 4 share an indexer
    /// domain (as do 5 and 6). Pinned because it is why round 3's cost is a
    /// step function of density.
    #[test]
    fn k_size_is_padded_to_a_power_of_two() {
        let n = 16usize;
        let k = |d| satisfiable_instance(n, d, &mut test_rng()).k_size();
        assert_eq!(k(2), 2 * n);
        assert_eq!(k(3), 4 * n);
        assert_eq!(k(4), 4 * n);
        assert_eq!(k(5), 8 * n);
        assert_eq!(k(6), 8 * n);
        assert_eq!(k(8), 8 * n);
    }

    #[test]
    fn witness_entries_are_nonzero() {
        let inst = satisfiable_instance(32, 4, &mut test_rng());
        assert!(inst.z.iter().all(|v| !v.is_zero()));
    }

    /// Density 2 must reproduce the original diagonal / off-by-one shape, so
    /// that measurements taken before the density parameter existed stay
    /// comparable.
    #[test]
    fn density_two_is_the_original_shape() {
        let n = 16usize;
        let inst = satisfiable_instance(n, 2, &mut test_rng());
        for (i, &(row, col, _)) in inst.matrix_a.iter().enumerate() {
            assert_eq!((row, col), (i, i));
        }
        for (i, &(row, col, _)) in inst.matrix_b.iter().enumerate() {
            assert_eq!((row, col), (i, (i + 1) % n));
        }
        for (i, &(row, col, _)) in inst.matrix_c.iter().enumerate() {
            assert_eq!((row, col), (i, i));
        }
    }

    #[test]
    fn matvec_matches_dense_product() {
        let n = 16usize;
        let inst = satisfiable_instance(n, 5, &mut test_rng());
        let got = inst.matvec(&inst.matrix_a);
        let mut want = vec![Fr::zero(); n];
        for &(i, j, v) in &inst.matrix_a {
            want[i] += v * inst.z[j];
        }
        assert_eq!(got, want);
    }
}
