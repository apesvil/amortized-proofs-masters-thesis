use crate::relations::abc::{AbcParams, AbcStatement, AbcWitness};
use crate::relations::pcc::{PccStatement, PccWitness};
use crate::transcript::Blake3Transcript;

use super::rok_abc::{RokAbc, RokAbcProof};

/// `LfscPcc(Π_{ABC}, κ)` — the promise-bearing local-folding-scheme compiler
/// applied to `Π_{ABC} : R_{A,B,C}² → R_{A,B,C} × R_PCC`.
///
/// Folds `K = 2^κ` R_{A,B,C} leaves through a binary tree. Each fold node
/// produces one R_PCC promise. The output is:
/// - the root R_{A,B,C} instance,
/// - per leaf: a κ-step path proof (sibling + `RokAbcProof` at each level),
/// - per leaf: the κ R_PCC promises collected along its path
///   (kept as `Vec<PccStatement>`, not concatenated, so `FsMtPcc`'s nested
///    input shape consumes it directly).
pub struct AbcFold;

pub struct AbcFoldPathStep {
    pub sibling: AbcStatement,
    pub rok_abc_proof: RokAbcProof,
}

pub struct AbcFoldPathProof {
    /// Leaf → root order, length `κ`.
    pub steps: Vec<AbcFoldPathStep>,
}

impl AbcFold {
    pub fn prove(
        params: &AbcParams,
        leaves: Vec<(AbcStatement, AbcWitness)>,
        transcript: &mut Blake3Transcript,
    ) -> (
        (AbcStatement, AbcWitness),
        Vec<AbcFoldPathProof>,
        Vec<Vec<(PccStatement, PccWitness)>>,
    ) {
        let k = leaves.len();
        assert!(
            k.is_power_of_two() && k >= 2,
            "K must be a power of two ≥ 2"
        );
        let kappa = k.trailing_zeros() as usize;

        let mut node_left_stmts: Vec<Vec<AbcStatement>> = Vec::with_capacity(kappa);
        let mut node_right_stmts: Vec<Vec<AbcStatement>> = Vec::with_capacity(kappa);
        let mut node_proofs: Vec<Vec<RokAbcProof>> = Vec::with_capacity(kappa);
        let mut node_pccs: Vec<Vec<(PccStatement, PccWitness)>> = Vec::with_capacity(kappa);

        let mut current: Vec<(AbcStatement, AbcWitness)> = leaves;

        for level in 0..kappa {
            let pairs = current.len() / 2;
            let mut next: Vec<(AbcStatement, AbcWitness)> = Vec::with_capacity(pairs);
            let mut lefts: Vec<AbcStatement> = Vec::with_capacity(pairs);
            let mut rights: Vec<AbcStatement> = Vec::with_capacity(pairs);
            let mut proofs_lvl: Vec<RokAbcProof> = Vec::with_capacity(pairs);
            let mut pccs_lvl: Vec<(PccStatement, PccWitness)> = Vec::with_capacity(pairs);

            let mut iter = current.into_iter();
            for pair_idx in 0..pairs {
                let (left_stmt, left_wit) = iter.next().unwrap();
                let (right_stmt, right_wit) = iter.next().unwrap();

                let mut t_node = transcript.fork(b"abc_fold");
                t_node.absorb_usize(b"level", level);
                t_node.absorb_usize(b"pair", pair_idx);

                let (rok_proof, (s_out, w_out), (pcc_stmt, pcc_wit)) = RokAbc::reduce(
                    params,
                    &left_stmt,
                    &left_wit,
                    &right_stmt,
                    &right_wit,
                    &mut t_node,
                );

                lefts.push(left_stmt);
                rights.push(right_stmt);
                proofs_lvl.push(rok_proof);
                pccs_lvl.push((pcc_stmt, pcc_wit));
                next.push((s_out, w_out));
            }

            node_left_stmts.push(lefts);
            node_right_stmts.push(rights);
            node_proofs.push(proofs_lvl);
            node_pccs.push(pccs_lvl);
            current = next;
        }
        assert_eq!(current.len(), 1);
        let root = current.into_iter().next().unwrap();

        let mut paths: Vec<AbcFoldPathProof> = Vec::with_capacity(k);
        let mut bundles: Vec<Vec<(PccStatement, PccWitness)>> = Vec::with_capacity(k);
        for leaf_idx in 0..k {
            let mut steps: Vec<AbcFoldPathStep> = Vec::with_capacity(kappa);
            let mut bundle: Vec<(PccStatement, PccWitness)> = Vec::with_capacity(kappa);
            let mut cur = leaf_idx;
            for level in 0..kappa {
                let pair_idx = cur / 2;
                let position = cur & 1;
                let sibling = if position == 0 {
                    node_right_stmts[level][pair_idx].clone()
                } else {
                    node_left_stmts[level][pair_idx].clone()
                };
                steps.push(AbcFoldPathStep {
                    sibling,
                    rok_abc_proof: node_proofs[level][pair_idx].clone(),
                });
                let (pcc_s, pcc_w) = &node_pccs[level][pair_idx];
                bundle.push((pcc_s.clone(), pcc_w.clone()));
                cur = pair_idx;
            }
            paths.push(AbcFoldPathProof { steps });
            bundles.push(bundle);
        }

        (root, paths, bundles)
    }

    /// Per-leaf verifier.
    pub fn verify(
        _params: &AbcParams,
        index: usize,
        leaf: &AbcStatement,
        path: &AbcFoldPathProof,
        claimed_root: &AbcStatement,
        transcript: &mut Blake3Transcript,
    ) -> Option<Vec<PccStatement>> {
        let kappa = path.steps.len();
        let mut current = leaf.clone();
        let mut cur_idx = index;
        let mut pccs: Vec<PccStatement> = Vec::with_capacity(kappa);

        for (level, step) in path.steps.iter().enumerate() {
            let pair_idx = cur_idx / 2;
            let position = cur_idx & 1;
            let (left, right) = if position == 0 {
                (&current, &step.sibling)
            } else {
                (&step.sibling, &current)
            };

            let mut t_node = transcript.fork(b"abc_fold");
            t_node.absorb_usize(b"level", level);
            t_node.absorb_usize(b"pair", pair_idx);

            let (folded, pcc_promise) =
                RokAbc::verify(left, right, &step.rok_abc_proof, &mut t_node);

            current = folded;
            pccs.push(pcc_promise);
            cur_idx = pair_idx;
        }

        if &current != claimed_root {
            return None;
        }
        Some(pccs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Relation;
    use crate::pc;
    use crate::relations::abc::{leaf_instance, AbcRelation};
    use crate::relations::pcc::{PccParams, PccRelation};
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    fn k_leaves(k: usize, n: usize) -> (AbcParams, Vec<(AbcStatement, AbcWitness)>) {
        let rng = &mut test_rng();
        let a: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        let b: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, (i + 1) % n, Fr::from(1u64))).collect();
        let c: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
        let srs = pc::setup(40, rng);
        let params = AbcParams { srs, matrix_a: a, matrix_b: b, matrix_c: c, n };
        let leaves: Vec<(AbcStatement, AbcWitness)> = (0..k)
            .map(|_| leaf_instance(&params, Fr::rand(rng), Fr::rand(rng)))
            .collect();
        (params, leaves)
    }

    /// Like `k_leaves`, but also returns the `(α, β)` challenge behind each
    /// leaf (needed to build/reconstruct the leaf-correctness promise).
    fn k_leaves_with_challenges(
        k: usize,
        n: usize,
    ) -> (AbcParams, Vec<(AbcStatement, AbcWitness)>, Vec<(Fr, Fr)>) {
        let rng = &mut test_rng();
        let a: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        let b: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, (i + 1) % n, Fr::from(1u64))).collect();
        let c: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::rand(rng))).collect();
        let srs = pc::setup(40, rng);
        let params = AbcParams { srs, matrix_a: a, matrix_b: b, matrix_c: c, n };
        let mut leaves = Vec::with_capacity(k);
        let mut challenges = Vec::with_capacity(k);
        for _ in 0..k {
            let (alpha, beta) = (Fr::rand(rng), Fr::rand(rng));
            leaves.push(leaf_instance(&params, alpha, beta));
            challenges.push((alpha, beta));
        }
        (params, leaves, challenges)
    }

    #[test]
    fn fold_roundtrip_k4() {
        let (params, leaves) = k_leaves(4, 4);
        let mut t = Blake3Transcript::new(b"test");
        let (root, _paths, bundles) = AbcFold::prove(&params, leaves, &mut t);

        assert!(AbcRelation::is_satisfied(&params, &root.0, &root.1));

        let pcc_params = PccParams { srs: params.srs.clone() };
        for bundle in &bundles {
            assert_eq!(bundle.len(), 2);
            for (s, w) in bundle {
                assert!(PccRelation::is_satisfied(&pcc_params, s, w));
            }
        }
    }

    #[test]
    fn prover_verifier_agree() {
        let (params, leaves) = k_leaves(4, 4);
        let leaves_stmts: Vec<AbcStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let (root, paths, bundles) = AbcFold::prove(&params, leaves, &mut t_p);

        for (i, path) in paths.iter().enumerate() {
            let mut t_v = Blake3Transcript::new(b"test");
            let pccs = AbcFold::verify(&params, i, &leaves_stmts[i], path, &root.0, &mut t_v)
                .expect("honest verify must succeed");
            assert_eq!(pccs.len(), bundles[i].len());
            for (verifier_pcc, (prover_pcc, _)) in pccs.iter().zip(&bundles[i]) {
                assert_eq!(verifier_pcc, prover_pcc);
            }
        }
    }

    #[test]
    fn wrong_root_rejected() {
        let (params, leaves) = k_leaves(4, 4);
        let leaves_stmts: Vec<AbcStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let (mut root, paths, _bundles) = AbcFold::prove(&params, leaves, &mut t_p);
        root.0.y_a += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(AbcFold::verify(&params, 0, &leaves_stmts[0], &paths[0], &root.0, &mut t_v)
            .is_none());
    }

    #[test]
    fn tampered_sibling_rejected() {
        let (params, leaves) = k_leaves(4, 4);
        let leaves_stmts: Vec<AbcStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let (root, mut paths, _bundles) = AbcFold::prove(&params, leaves, &mut t_p);
        paths[2].steps[0].sibling.y_b += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(AbcFold::verify(&params, 2, &leaves_stmts[2], &paths[2], &root.0, &mut t_v)
            .is_none());
    }

    #[test]
    fn paired_leaves_share_level0_pcc() {
        let (params, leaves) = k_leaves(4, 4);
        let mut t = Blake3Transcript::new(b"test");
        let (_root, _paths, bundles) = AbcFold::prove(&params, leaves, &mut t);

        assert_eq!(bundles[0][0].0, bundles[1][0].0);
        assert_eq!(bundles[2][0].0, bundles[3][0].0);
        assert_ne!(bundles[0][0].0, bundles[2][0].0);
    }

    #[test]
    fn end_to_end_with_fsmt_pcc() {
        use crate::reductions::fsmt_pcc::FsMtPcc;
        use crate::relations::pco::{PcoParams, PcoRelation};

        let (params, leaves) = k_leaves(4, 4);
        let leaves_stmts: Vec<AbcStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t = Blake3Transcript::new(b"test");
        let (root, paths, bundles) = AbcFold::prove(&params, leaves, &mut t);

        assert!(AbcRelation::is_satisfied(&params, &root.0, &root.1));

        let pcc_stmts: Vec<Vec<PccStatement>> = bundles
            .iter()
            .map(|b| b.iter().map(|(s, _)| s.clone()).collect())
            .collect();
        let pcc_wits: Vec<Vec<PccWitness>> = bundles
            .into_iter()
            .map(|b| b.into_iter().map(|(_, w)| w).collect())
            .collect();

        let pcc_params = PccParams { srs: params.srs.clone() };

        let mut t_fs = Blake3Transcript::new(b"fsmt");
        let outputs =
            FsMtPcc::reduce_amortized(&pcc_params, &pcc_stmts, &pcc_wits, &mut t_fs);

        let pco_params = PcoParams { srs: params.srs.clone() };
        let x = outputs[0].1.point;
        for (_, pco_s, pco_w) in &outputs {
            assert_eq!(pco_s.point, x);
            assert!(PcoRelation::is_satisfied(&pco_params, pco_s, pco_w));
        }

        for (i, (proof, prover_pco, _)) in outputs.iter().enumerate() {
            let mut t_v = Blake3Transcript::new(b"test");
            let _ = AbcFold::verify(
                &params,
                i,
                &leaves_stmts[i],
                &paths[i],
                &root.0,
                &mut t_v,
            )
            .expect("AbcFold verify must succeed");

            let mut t_fsv = Blake3Transcript::new(b"fsmt");
            let pco_v =
                FsMtPcc::verify(&pcc_params, i, &pcc_stmts[i], proof, &mut t_fsv)
                    .expect("FsMtPcc verify must succeed");
            assert_eq!(*prover_pco, pco_v);
        }
    }

    /// Full sound Side-2 discharge including the per-leaf encoding-correctness
    /// promise: κ+1 promises per leaf go through `full_pcc_bundles` →
    /// `prove_discharge`, and every local verifier reconstructs its own
    /// leaf-correctness statement, matches the prover's bundle, and checks the
    /// FsMtPcc + PcoFold + KZG-opening chain.
    #[test]
    fn end_to_end_with_leaf_correctness() {
        use crate::reductions::discharge::prove_discharge;
        use crate::reductions::fsmt_pcc::FsMtPcc;
        use crate::reductions::pco_fold::PcoFold;
        use crate::relations::abc_leaf_pcc::{full_pcc_bundles, leaf_correctness_pcc};

        let (params, leaves, challenges) = k_leaves_with_challenges(4, 4);
        let leaf_stmts: Vec<AbcStatement> =
            leaves.iter().map(|(s, _)| s.clone()).collect();
        let pcc_params = PccParams { srs: params.srs.clone() };

        let mut t = Blake3Transcript::new(b"test");
        let (root, paths, bundles) = AbcFold::prove(&params, leaves, &mut t);
        assert!(AbcRelation::is_satisfied(&params, &root.0, &root.1));

        // Append each leaf's encoding-correctness promise → κ+1 per leaf.
        let full = full_pcc_bundles(&pcc_params, params.n, &challenges, bundles);
        let pcc_stmts: Vec<Vec<PccStatement>> = full
            .iter()
            .map(|b| b.iter().map(|(s, _)| s.clone()).collect())
            .collect();
        let pcc_wits: Vec<Vec<PccWitness>> = full
            .into_iter()
            .map(|b| b.into_iter().map(|(_, w)| w).collect())
            .collect();

        let mut t_f = Blake3Transcript::new(b"fsmt");
        let mut t_p = Blake3Transcript::new(b"pco");
        let d = prove_discharge(&pcc_params, &pcc_stmts, &pcc_wits, &mut t_f, &mut t_p);

        for i in 0..pcc_stmts.len() {
            let mut t_v = Blake3Transcript::new(b"test");
            let mut promises =
                AbcFold::verify(&params, i, &leaf_stmts[i], &paths[i], &root.0, &mut t_v)
                    .expect("AbcFold verify");
            promises.push(
                leaf_correctness_pcc(&pcc_params, params.n, challenges[i].0, challenges[i].1).0,
            );
            // Verifier's reconstructed bundle must match what the prover discharged.
            assert_eq!(promises, pcc_stmts[i]);

            let mut t_fv = Blake3Transcript::new(b"fsmt");
            let leaf_pco =
                FsMtPcc::verify(&pcc_params, i, &promises, &d.fsmt_proofs[i], &mut t_fv)
                    .expect("FsMtPcc verify");
            let mut t_pv = Blake3Transcript::new(b"pco");
            assert!(PcoFold::verify(i, &leaf_pco, &d.pco_paths[i], &d.root_stmt, &mut t_pv));
        }

        assert!(pc::verify(
            &params.srs,
            &d.root_stmt.commitment,
            d.root_stmt.point,
            d.root_stmt.value,
            &d.opening,
        ));
    }
}
