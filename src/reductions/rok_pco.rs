use ark_bls12_381::Fr;
use ark_ff::Zero;
use ark_poly::univariate::SparsePolynomial;

use crate::pc::Comm;
use crate::relations::pco::{PcoStatement, PcoWitness};
use crate::transcript::Blake3Transcript;

/// Non-interactive `Π_PCO : (R_PCO^x)^ℓ → R_PCO^x`.
///
/// Batches `ℓ` PCO statements at a common evaluation point `x` into a single
/// PCO statement, leveraging the polynomial commitment's additive
/// homomorphism: the new commitment `C' = Σᵢ rⁱ · Cᵢ` is the same linear
/// combination as the new polynomial `p'(X) = Σᵢ rⁱ · pᵢ(X)` and the new
/// value `y' = Σᵢ rⁱ · yᵢ`.
///
/// No proof message: `r` is the only round of communication and is squeezed
/// from the transcript after absorbing the input statements.
pub struct RokPco;

impl RokPco {
    /// Prover side. All inputs must share the same evaluation point `x`
    /// (paper's `R_PCO^x` precondition); panics otherwise.
    ///
    /// **Reference-only** — not reached by the benchmarks. `PcoFold` uses
    /// `fold_pair_stmt` + `verify` (both below); this witness-bearing `reduce`
    /// is kept for tests and as the readable batch reference.
    pub fn reduce(
        stmts: &[PcoStatement],
        wits: &[PcoWitness],
        transcript: &mut Blake3Transcript,
    ) -> (PcoStatement, PcoWitness) {
        assert_eq!(stmts.len(), wits.len(), "stmts and wits must agree in length");
        assert!(!stmts.is_empty(), "input must be non-empty");
        assert_shared_point(stmts);

        absorb_inputs(transcript, stmts);
        let r: Fr = transcript.squeeze_field(b"rok_pco::r");

        let mut p_prime = SparsePolynomial::<Fr>::zero();
        let mut c_prime = Comm::zero();
        let mut y_prime = Fr::from(0u64);
        let mut r_pow = Fr::from(1u64);
        for (s, w) in stmts.iter().zip(wits) {
            let scaled = &w.polynomial * r_pow;
            p_prime = &p_prime + &scaled;
            c_prime += s.commitment * r_pow;
            y_prime += s.value * r_pow;
            r_pow *= r;
        }

        let out_stmt = PcoStatement {
            commitment: c_prime,
            point: stmts[0].point,
            value: y_prime,
        };
        let out_wit = PcoWitness { polynomial: p_prime };
        (out_stmt, out_wit)
    }

    /// Verifier side. Reconstructs the batched statement deterministically.
    /// (No proof message; `r` is derived from the same transcript state.)
    pub fn verify(stmts: &[PcoStatement], transcript: &mut Blake3Transcript) -> PcoStatement {
        assert!(!stmts.is_empty(), "input must be non-empty");
        assert_shared_point(stmts);

        absorb_inputs(transcript, stmts);
        let r: Fr = transcript.squeeze_field(b"rok_pco::r");

        let mut c_prime = Comm::zero();
        let mut y_prime = Fr::from(0u64);
        let mut r_pow = Fr::from(1u64);
        for s in stmts {
            c_prime += s.commitment * r_pow;
            y_prime += s.value * r_pow;
            r_pow *= r;
        }

        PcoStatement {
            commitment: c_prime,
            point: stmts[0].point,
            value: y_prime,
        }
    }

    /// Fold a specific left/right statement pair, returning the folded
    /// statement together with the challenge used. The transcript operations
    /// match `reduce`/`verify` on the 2-element slice `[left, right]`, so a
    /// tree-folder can recover each leaf's root coefficient without ever
    /// materializing intermediate witness polynomials.
    pub fn fold_pair_stmt(
        left: &PcoStatement,
        right: &PcoStatement,
        transcript: &mut Blake3Transcript,
    ) -> (PcoStatement, Fr) {
        debug_assert_eq!(
            left.point, right.point,
            "RokPco::fold_pair_stmt inputs must share the same point"
        );
        let stmts = [left.clone(), right.clone()];
        absorb_inputs(transcript, &stmts);
        let r: Fr = transcript.squeeze_field(b"rok_pco::r");
        let out = PcoStatement {
            commitment: left.commitment + right.commitment * r,
            point: left.point,
            value: left.value + right.value * r,
        };
        (out, r)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_shared_point(stmts: &[PcoStatement]) {
    let x = stmts[0].point;
    for s in &stmts[1..] {
        assert_eq!(s.point, x, "all inputs of R_PCO^x must share the same point");
    }
}

fn absorb_inputs(t: &mut Blake3Transcript, stmts: &[PcoStatement]) {
    t.absorb_usize(b"n", stmts.len());
    for s in stmts {
        t.absorb(b"commitment", &s.commitment);
        t.absorb(b"point", &s.point);
        t.absorb(b"value", &s.value);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Relation;
    use crate::pc::{self, Poly, Srs};
    use crate::relations::pco::{PcoParams, PcoRelation};
    use ark_ff::UniformRand;
    use ark_poly::Polynomial;
    use ark_std::test_rng;

    fn make_inputs(
        srs: &Srs,
        x: Fr,
        polys: Vec<SparsePolynomial<Fr>>,
    ) -> (Vec<PcoStatement>, Vec<PcoWitness>) {
        let stmts: Vec<PcoStatement> = polys
            .iter()
            .map(|p| PcoStatement {
                commitment: pc::commit(srs, &Poly::Sparse(p.clone())),
                point: x,
                value: p.evaluate(&x),
            })
            .collect();
        let wits: Vec<PcoWitness> = polys
            .into_iter()
            .map(|p| PcoWitness { polynomial: p })
            .collect();
        (stmts, wits)
    }

    fn three_inputs() -> (Srs, Vec<PcoStatement>, Vec<PcoWitness>) {
        let rng = &mut test_rng();
        let srs = pc::setup(20, rng);
        let x = Fr::rand(rng);
        let polys = vec![
            SparsePolynomial::from_coefficients_vec(vec![
                (0, Fr::from(1u64)),
                (1, Fr::from(2u64)),
            ]),
            SparsePolynomial::from_coefficients_vec(vec![
                (0, Fr::from(3u64)),
                (2, Fr::from(4u64)),
            ]),
            SparsePolynomial::from_coefficients_vec(vec![
                (1, Fr::from(5u64)),
                (3, Fr::from(6u64)),
            ]),
        ];
        let (stmts, wits) = make_inputs(&srs, x, polys);
        (srs, stmts, wits)
    }

    /// Batched output `(stmt, wit)` must satisfy `PcoRelation`.
    #[test]
    fn reduce_roundtrip() {
        let (srs, stmts, wits) = three_inputs();
        let mut t = Blake3Transcript::new(b"test");
        let (out_stmt, out_wit) = RokPco::reduce(&stmts, &wits, &mut t);
        let pco_params = PcoParams { srs };
        assert!(PcoRelation::is_satisfied(&pco_params, &out_stmt, &out_wit));
    }

    /// Prover and verifier with identical labels reconstruct the same statement.
    #[test]
    fn prover_verifier_agree() {
        let (_srs, stmts, wits) = three_inputs();
        let mut t_p = Blake3Transcript::new(b"test");
        let mut t_v = Blake3Transcript::new(b"test");
        let (s_p, _) = RokPco::reduce(&stmts, &wits, &mut t_p);
        let s_v = RokPco::verify(&stmts, &mut t_v);
        assert_eq!(s_p, s_v);
    }

    /// The batched commitment must equal a fresh commit of the batched
    /// polynomial — pinning the homomorphism property explicitly.
    #[test]
    fn homomorphism_check() {
        let (srs, stmts, wits) = three_inputs();
        let mut t = Blake3Transcript::new(b"test");
        let (out_stmt, out_wit) = RokPco::reduce(&stmts, &wits, &mut t);
        let recomputed = pc::commit(&srs, &Poly::Sparse(out_wit.polynomial));
        assert_eq!(recomputed, out_stmt.commitment);
    }

    /// Different transcript labels yield different `r` and thus different
    /// batched outputs.
    #[test]
    fn different_label_different_outputs() {
        let (_srs, stmts, wits) = three_inputs();
        let mut t_a = Blake3Transcript::new(b"label_a");
        let mut t_b = Blake3Transcript::new(b"label_b");
        let (s_a, _) = RokPco::reduce(&stmts, &wits, &mut t_a);
        let (s_b, _) = RokPco::reduce(&stmts, &wits, &mut t_b);
        assert_ne!(s_a, s_b);
    }

    /// With ℓ = 1, the batched output equals the single input (since the only
    /// term has coefficient `r^0 = 1`).
    #[test]
    fn single_input_passthrough() {
        let rng = &mut test_rng();
        let srs = pc::setup(10, rng);
        let x = Fr::rand(rng);
        let p = SparsePolynomial::from_coefficients_vec(vec![
            (0, Fr::from(7u64)),
            (1, Fr::from(11u64)),
        ]);
        let (stmts, wits) = make_inputs(&srs, x, vec![p.clone()]);

        let mut t = Blake3Transcript::new(b"test");
        let (out_stmt, out_wit) = RokPco::reduce(&stmts, &wits, &mut t);
        assert_eq!(out_stmt, stmts[0]);
        assert_eq!(out_wit.polynomial, p);
    }

    /// Inputs at different evaluation points violate the `R_PCO^x`
    /// precondition and must trigger the assertion.
    #[test]
    #[should_panic(expected = "all inputs of R_PCO^x must share the same point")]
    fn mismatched_points_panics() {
        let rng = &mut test_rng();
        let srs = pc::setup(10, rng);
        let p = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(1u64))]);
        let c = pc::commit(&srs, &Poly::Sparse(p.clone()));
        let stmts = vec![
            PcoStatement { commitment: c, point: Fr::from(1u64), value: Fr::from(1u64) },
            PcoStatement { commitment: c, point: Fr::from(2u64), value: Fr::from(1u64) },
        ];
        let wits = vec![
            PcoWitness { polynomial: p.clone() },
            PcoWitness { polynomial: p },
        ];
        let mut t = Blake3Transcript::new(b"test");
        let _ = RokPco::reduce(&stmts, &wits, &mut t);
    }
}
