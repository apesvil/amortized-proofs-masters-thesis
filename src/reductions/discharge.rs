use ark_bls12_381::Fr;
use ark_ff::Zero;
use ark_poly::univariate::SparsePolynomial;

use crate::pc::{self, Opening, Poly};
use crate::relations::pcc::{PccParams, PccStatement, PccWitness};
use crate::relations::pco::{PcoStatement, PcoWitness};
use crate::transcript::Blake3Transcript;

use super::fsmt_pcc::{FsMtPcc, FsMtPccProof};
use super::pco_fold::{PcoFold, PcoFoldPathProof};

/// Output of the amortized promise discharge: everything the `K` local
/// verifiers need to check their fold promises, plus the single materialized
/// root witness and its KZG opening.
pub struct DischargeProof {
    /// Per-leaf FsMtPcc proof (feeds `FsMtPcc::verify`).
    pub fsmt_proofs: Vec<FsMtPccProof>,
    /// Per-leaf R_PCO statement at the shared point `x` (the FsMtPcc output).
    pub leaf_pco_stmts: Vec<PcoStatement>,
    /// Per-leaf PcoFold path proof (feeds `PcoFold::verify`).
    pub pco_paths: Vec<PcoFoldPathProof>,
    /// The single folded root R_PCO statement.
    pub root_stmt: PcoStatement,
    /// The single materialized root witness
    /// `Σ_i coeff_i · Σ_k r_i^k · p_{i,k}`.
    pub root_wit: PcoWitness,
    /// KZG opening of `root_stmt` at its point `x`.
    pub opening: Opening,
}

/// Amortized promise discharge with a single materialized witness.
///
/// Runs `FsMtPcc` (Π_PC ∘ Π_DT) and `PcoFold` on statements/commitments only —
/// the transcript-bearing work is unchanged, so every challenge, proof, and
/// statement is byte-identical to `FsMtPcc::reduce_amortized` followed by
/// `PcoFold::fold_amortized`. The `K` per-leaf witnesses and the fold tree's
/// intermediate node witnesses are never built; instead the single opened
/// polynomial
/// ```text
///   P_root(X) = Σ_i coeff_i · ( Σ_k r_i^k · p_{i,k}(X) )
/// ```
/// is assembled in one pass over the original sparse promise polynomials and
/// opened once. This removes the `O(K·(n+K))` witness-densification cost of the
/// per-leaf-then-fold path, leaving one `O(n+K)` opening plus an `O(κ·K²)`
/// sparse accumulation independent of `n`.
///
/// `fsmt_transcript` and `pco_transcript` are the two (independent) transcripts
/// the verifiers mirror with `FsMtPcc::verify` and `PcoFold::verify`.
pub fn prove_discharge(
    params: &PccParams,
    stmts: &[Vec<PccStatement>],
    wits: &[Vec<PccWitness>],
    fsmt_transcript: &mut Blake3Transcript,
    pco_transcript: &mut Blake3Transcript,
) -> DischargeProof {
    // Phase 1: FsMtPcc statement/proof production (no leaf witnesses built).
    let (fsmt_proofs, leaf_pco_stmts, recipes) =
        FsMtPcc::reduce_amortized_stmts(params, stmts, wits, fsmt_transcript);

    // Phase 2: fold statements only, recovering each leaf's root coefficient.
    let (root_stmt, pco_paths, coeffs) =
        PcoFold::fold_stmts(&leaf_pco_stmts, pco_transcript);

    // Phase 3: assemble the single root witness in one pass. The weight of the
    // sparse basis polynomial `p_{i,k}` is `coeff_i · r_i^k`.
    let big_d = params.srs.powers_g1.len();
    let mut acc = vec![Fr::zero(); big_d];
    for (recipe, &coeff) in recipes.iter().zip(&coeffs) {
        let mut w = coeff;
        for p in &recipe.polys {
            for &(deg, c) in p.iter() {
                acc[deg] += w * c;
            }
            w *= recipe.r;
        }
    }
    let root_poly = SparsePolynomial::from_coefficients_vec(
        acc.into_iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, c)| (i, c))
            .collect(),
    );

    let (opening, _value) =
        pc::prove(&params.srs, &Poly::Sparse(root_poly.clone()), root_stmt.point);
    let root_wit = PcoWitness { polynomial: root_poly };

    DischargeProof {
        fsmt_proofs,
        leaf_pco_stmts,
        pco_paths,
        root_stmt,
        root_wit,
        opening,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::prove_discharge;
    use crate::core::Relation;
    use crate::pc::{self, Poly, Srs};
    use crate::reductions::fsmt_pcc::FsMtPcc;
    use crate::reductions::pco_fold::PcoFold;
    use crate::relations::pcc::{
        Constraint, Monomial, PccParams, PccStatement, PccWitness,
    };
    use crate::relations::pco::{PcoParams, PcoRelation, PcoStatement, PcoWitness};
    use crate::transcript::Blake3Transcript;
    use ark_bls12_381::Fr;
    use ark_poly::univariate::SparsePolynomial;
    use ark_std::test_rng;

    /// One PCC instance with constraint `Y_0 − Y_1 = 0` over constants
    /// `p_0 = a, p_1 = a` (mirrors the `fsmt_pcc` test helper).
    fn make_inner(srs: &Srs, a: u64) -> (PccStatement, PccWitness) {
        let p0 = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(a))]);
        let p1 = SparsePolynomial::from_coefficients_vec(vec![(0, Fr::from(a))]);
        let c0 = pc::commit(srs, &Poly::Sparse(p0.clone()));
        let c1 = pc::commit(srs, &Poly::Sparse(p1.clone()));
        let stmt = PccStatement {
            commitments: vec![c0, c1],
            degrees: vec![1, 1],
            constraints: vec![Constraint {
                monomials: vec![
                    Monomial { coeff: Fr::from(1u64), x_deg: 0, y_terms: vec![(0, 1)] },
                    Monomial { coeff: -Fr::from(1u64), x_deg: 0, y_terms: vec![(1, 1)] },
                ],
            }],
        };
        let wit = PccWitness { polynomials: vec![p0, p1] };
        (stmt, wit)
    }

    fn k_leaves_nested(
        k: usize,
        kappa_inner: usize,
    ) -> (PccParams, Vec<Vec<PccStatement>>, Vec<Vec<PccWitness>>) {
        let rng = &mut test_rng();
        let srs = pc::setup(20, rng);
        let mut stmts = Vec::with_capacity(k);
        let mut wits = Vec::with_capacity(k);
        for i in 0..k {
            let mut bs = Vec::with_capacity(kappa_inner);
            let mut bw = Vec::with_capacity(kappa_inner);
            for j in 0..kappa_inner {
                let (s, w) = make_inner(&srs, (i * kappa_inner + j + 1) as u64);
                bs.push(s);
                bw.push(w);
            }
            stmts.push(bs);
            wits.push(bw);
        }
        (PccParams { srs }, stmts, wits)
    }

    /// `prove_discharge` yields the exact same root statement, root witness,
    /// and opening as the per-leaf `reduce_amortized` + `fold_amortized` +
    /// `pc::prove` pipeline — the refactor only changes *how* the witness is
    /// assembled, not the result.
    #[test]
    fn discharge_matches_manual_pipeline() {
        let (params, stmts, wits) = k_leaves_nested(4, 2);

        // Manual per-leaf pipeline.
        let mut t_f = Blake3Transcript::new(b"disc::fsmt");
        let outs = FsMtPcc::reduce_amortized(&params, &stmts, &wits, &mut t_f);
        let pco_leaves: Vec<(PcoStatement, PcoWitness)> =
            outs.iter().map(|(_, s, w)| (s.clone(), w.clone())).collect();
        let mut t_p = Blake3Transcript::new(b"disc::pco");
        let ((root_s, root_w), _paths) = PcoFold::fold_amortized(pco_leaves, &mut t_p);
        let (open_manual, _) =
            pc::prove(&params.srs, &Poly::Sparse(root_w.polynomial.clone()), root_s.point);

        // Single-witness discharge (fresh transcripts, same labels).
        let mut t_f2 = Blake3Transcript::new(b"disc::fsmt");
        let mut t_p2 = Blake3Transcript::new(b"disc::pco");
        let d = prove_discharge(&params, &stmts, &wits, &mut t_f2, &mut t_p2);

        assert_eq!(d.root_stmt, root_s);
        assert_eq!(d.root_wit.polynomial, root_w.polynomial);
        assert_eq!(d.opening, open_manual);
    }

    /// The full per-leaf verifier path accepts `prove_discharge`'s output.
    #[test]
    fn discharge_end_to_end_verifies() {
        let (params, stmts, wits) = k_leaves_nested(4, 3);
        let mut t_f = Blake3Transcript::new(b"disc::fsmt");
        let mut t_p = Blake3Transcript::new(b"disc::pco");
        let d = prove_discharge(&params, &stmts, &wits, &mut t_f, &mut t_p);

        // Root satisfies R_PCO and its opening verifies.
        let pco_params = PcoParams { srs: params.srs.clone() };
        assert!(PcoRelation::is_satisfied(&pco_params, &d.root_stmt, &d.root_wit));
        assert!(pc::verify(
            &params.srs,
            &d.root_stmt.commitment,
            d.root_stmt.point,
            d.root_stmt.value,
            &d.opening,
        ));

        // Each local verifier: FsMtPcc::verify → PcoFold::verify → shared root.
        for i in 0..stmts.len() {
            let mut t_fv = Blake3Transcript::new(b"disc::fsmt");
            let leaf_pco = FsMtPcc::verify(&params, i, &stmts[i], &d.fsmt_proofs[i], &mut t_fv)
                .expect("FsMtPcc verify");
            assert_eq!(leaf_pco, d.leaf_pco_stmts[i]);
            let mut t_pv = Blake3Transcript::new(b"disc::pco");
            assert!(PcoFold::verify(i, &leaf_pco, &d.pco_paths[i], &d.root_stmt, &mut t_pv));
        }
    }
}
