use crate::relations::m::{MParams, MStatement, MWitness};
use crate::relations::pcc::{PccStatement, PccWitness};
use crate::transcript::Blake3Transcript;

use super::rok_m::{RokM, RokMProof};

/// `LfscPcc(Π_M, κ)` — the promise-bearing local-folding-scheme compiler
/// applied to `Π_M : R_M² → R_M × R_PCC`.
///
/// Folds `K = 2^κ` `R_M` leaves through a binary tree of `Π_M`. Each
/// fold node produces one `R_PCC` "promise". The output is:
/// - the root `R_M` instance,
/// - per leaf: a κ-step path proof (sibling + `RokMProof` at each level),
/// - per leaf: the κ `R_PCC` promises collected along its path
///   (`R_PCC^κ` in the paper's notation — kept as a `Vec<PccStatement>`,
///    not concatenated, so `FsMtPcc`'s nested input shape consumes it directly).
pub struct MFold;

/// One step of a leaf's fold path: the sibling `R_M` statement at this
/// level and the `Π_M` proof that produced this level's fold.
pub struct MFoldPathStep {
    pub sibling: MStatement,
    pub rok_m_proof: RokMProof,
}

pub struct MFoldPathProof {
    /// Leaf → root order, length `κ`.
    pub steps: Vec<MFoldPathStep>,
}

impl MFold {
    /// Prover side. Returns
    ///   `(root, paths, bundles)`
    /// with `paths.len() == bundles.len() == K`, each `paths[i].steps.len()`
    /// and `bundles[i].len()` equal to `κ`.
    pub fn prove(
        params: &MParams,
        leaves: Vec<(MStatement, MWitness)>,
        transcript: &mut Blake3Transcript,
    ) -> (
        (MStatement, MWitness),
        Vec<MFoldPathProof>,
        Vec<Vec<(PccStatement, PccWitness)>>,
    ) {
        let k = leaves.len();
        assert!(
            k.is_power_of_two() && k >= 2,
            "K must be a power of two ≥ 2"
        );
        let kappa = k.trailing_zeros() as usize;

        // Per-level / per-pair bookkeeping. We store the *input* M statements
        // at each fold node (left / right) so the path-building step can
        // pick out each leaf's sibling at every level.
        let mut node_left_stmts: Vec<Vec<MStatement>> = Vec::with_capacity(kappa);
        let mut node_right_stmts: Vec<Vec<MStatement>> = Vec::with_capacity(kappa);
        let mut node_proofs: Vec<Vec<RokMProof>> = Vec::with_capacity(kappa);
        let mut node_pccs: Vec<Vec<(PccStatement, PccWitness)>> = Vec::with_capacity(kappa);

        // Drive the fold layer by layer.
        let mut current: Vec<(MStatement, MWitness)> = leaves;

        for level in 0..kappa {
            let pairs = current.len() / 2;
            let mut next: Vec<(MStatement, MWitness)> = Vec::with_capacity(pairs);
            let mut lefts: Vec<MStatement> = Vec::with_capacity(pairs);
            let mut rights: Vec<MStatement> = Vec::with_capacity(pairs);
            let mut proofs_lvl: Vec<RokMProof> = Vec::with_capacity(pairs);
            let mut pccs_lvl: Vec<(PccStatement, PccWitness)> = Vec::with_capacity(pairs);

            let mut iter = current.into_iter();
            for pair_idx in 0..pairs {
                let (left_stmt, left_wit) = iter.next().unwrap();
                let (right_stmt, right_wit) = iter.next().unwrap();

                let mut t_node = transcript.fork(b"m_fold");
                t_node.absorb_usize(b"level", level);
                t_node.absorb_usize(b"pair", pair_idx);

                let (rok_proof, (m_out, m_wit), (pcc_stmt, pcc_wit)) = RokM::reduce(
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
                next.push((m_out, m_wit));
            }

            node_left_stmts.push(lefts);
            node_right_stmts.push(rights);
            node_proofs.push(proofs_lvl);
            node_pccs.push(pccs_lvl);
            current = next;
        }
        assert_eq!(current.len(), 1);
        let root = current.into_iter().next().unwrap();

        // Build the per-leaf paths and PCC bundles by walking each leaf up
        // through its κ fold nodes.
        let mut paths: Vec<MFoldPathProof> = Vec::with_capacity(k);
        let mut bundles: Vec<Vec<(PccStatement, PccWitness)>> = Vec::with_capacity(k);
        for leaf_idx in 0..k {
            let mut steps: Vec<MFoldPathStep> = Vec::with_capacity(kappa);
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
                steps.push(MFoldPathStep {
                    sibling,
                    rok_m_proof: node_proofs[level][pair_idx].clone(),
                });
                let (pcc_s, pcc_w) = &node_pccs[level][pair_idx];
                bundle.push((pcc_s.clone(), pcc_w.clone()));
                cur = pair_idx;
            }
            paths.push(MFoldPathProof { steps });
            bundles.push(bundle);
        }

        (root, paths, bundles)
    }

    /// Per-leaf verifier. Returns the κ R_PCC promises along this leaf's
    /// path if (and only if) the reconstructed root matches `claimed_root`.
    pub fn verify(
        _params: &MParams,
        index: usize,
        leaf: &MStatement,
        path: &MFoldPathProof,
        claimed_root: &MStatement,
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

            let mut t_node = transcript.fork(b"m_fold");
            t_node.absorb_usize(b"level", level);
            t_node.absorb_usize(b"pair", pair_idx);

            let (folded_m, pcc_promise) =
                RokM::verify(left, right, &step.rok_m_proof, &mut t_node);

            current = folded_m;
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
    use crate::relations::m::{leaf_instance, MRelation};
    use crate::relations::pcc::{PccParams, PccRelation};
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    /// Build params (identity matrix `I_n`) and K random leaves.
    fn k_m_leaves(k: usize, n: usize) -> (MParams, Vec<(MStatement, MWitness)>) {
        let rng = &mut test_rng();
        let matrix: Vec<(usize, usize, Fr)> =
            (0..n).map(|i| (i, i, Fr::from(1u64))).collect();
        let srs = pc::setup(40, rng); // large enough for folded N polynomials up to deg n + K
        let params = MParams { srs, matrix, n };
        let leaves: Vec<(MStatement, MWitness)> = (0..k)
            .map(|_| leaf_instance(&params, Fr::rand(rng), Fr::rand(rng)))
            .collect();
        (params, leaves)
    }

    /// Round-trip K = 4: root must satisfy `R_M`; every entry of every
    /// per-leaf PCC bundle must satisfy `R_PCC`.
    #[test]
    fn fold_roundtrip_k4() {
        let (params, leaves) = k_m_leaves(4, 4);
        let mut t = Blake3Transcript::new(b"test");
        let (root, _paths, bundles) = MFold::prove(&params, leaves, &mut t);

        assert!(MRelation::is_satisfied(&params, &root.0, &root.1));

        let pcc_params = PccParams { srs: params.srs.clone() };
        for bundle in &bundles {
            assert_eq!(bundle.len(), 2); // κ = log2(4) = 2
            for (s, w) in bundle {
                assert!(PccRelation::is_satisfied(&pcc_params, s, w));
            }
        }
    }

    /// Per-leaf verify reconstructs the κ PCC statements the prover emitted.
    #[test]
    fn prover_verifier_agree() {
        let (params, leaves) = k_m_leaves(4, 4);
        let leaves_stmts: Vec<MStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let (root, paths, bundles) = MFold::prove(&params, leaves, &mut t_p);

        for (i, path) in paths.iter().enumerate() {
            let mut t_v = Blake3Transcript::new(b"test");
            let pccs = MFold::verify(&params, i, &leaves_stmts[i], path, &root.0, &mut t_v)
                .expect("honest verify must succeed");
            assert_eq!(pccs.len(), bundles[i].len());
            for (verifier_pcc, (prover_pcc, _)) in pccs.iter().zip(&bundles[i]) {
                assert_eq!(verifier_pcc, prover_pcc);
            }
        }
    }

    /// Verifier with a tampered claimed root → `None`.
    #[test]
    fn wrong_root_rejected() {
        let (params, leaves) = k_m_leaves(4, 4);
        let leaves_stmts: Vec<MStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let (mut root, paths, _bundles) = MFold::prove(&params, leaves, &mut t_p);

        // Tamper the root's value.
        root.0.value += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(MFold::verify(&params, 0, &leaves_stmts[0], &paths[0], &root.0, &mut t_v)
            .is_none());
    }

    /// Mutating any sibling in the path proof causes the fold chain to
    /// reach the wrong root → reject.
    #[test]
    fn tampered_sibling_rejected() {
        let (params, leaves) = k_m_leaves(4, 4);
        let leaves_stmts: Vec<MStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t_p = Blake3Transcript::new(b"test");
        let (root, mut paths, _bundles) = MFold::prove(&params, leaves, &mut t_p);

        // Tamper a sibling at level 0 of leaf 2's path.
        paths[2].steps[0].sibling.value += Fr::from(1u64);

        let mut t_v = Blake3Transcript::new(b"test");
        assert!(MFold::verify(&params, 2, &leaves_stmts[2], &paths[2], &root.0, &mut t_v)
            .is_none());
    }

    /// Leaves 0 and 1 are paired at level 0 — their first PCC promise in the
    /// bundle should be identical (same fold node).
    #[test]
    fn paired_leaves_share_level0_pcc() {
        let (params, leaves) = k_m_leaves(4, 4);
        let mut t = Blake3Transcript::new(b"test");
        let (_root, _paths, bundles) = MFold::prove(&params, leaves, &mut t);

        assert_eq!(bundles[0][0].0, bundles[1][0].0);
        // Leaves 2 and 3 also share their level-0 PCC.
        assert_eq!(bundles[2][0].0, bundles[3][0].0);
        // Leaves 0 and 2 don't (different fold node at level 0).
        assert_ne!(bundles[0][0].0, bundles[2][0].0);
    }

    /// End-to-end: K leaves → MFold → bundles feed straight into FsMtPcc's
    /// nested input shape; every leaf produces a valid PCO statement.
    #[test]
    fn end_to_end_with_fsmt_pcc() {
        use crate::reductions::fsmt_pcc::FsMtPcc;
        use crate::relations::pco::{PcoParams, PcoRelation};

        let (m_params, leaves) = k_m_leaves(4, 4);
        let leaves_stmts: Vec<MStatement> = leaves.iter().map(|(s, _)| s.clone()).collect();

        let mut t = Blake3Transcript::new(b"test");
        let (root, paths, bundles) = MFold::prove(&m_params, leaves, &mut t);

        // Sanity: root M satisfies its relation.
        assert!(MRelation::is_satisfied(&m_params, &root.0, &root.1));

        // Split per-leaf (stmt, wit) bundles into the two nested vecs that
        // FsMtPcc expects.
        let pcc_stmts: Vec<Vec<PccStatement>> = bundles
            .iter()
            .map(|b| b.iter().map(|(s, _)| s.clone()).collect())
            .collect();
        let pcc_wits: Vec<Vec<PccWitness>> = bundles
            .into_iter()
            .map(|b| b.into_iter().map(|(_, w)| w).collect())
            .collect();

        let pcc_params = PccParams { srs: m_params.srs.clone() };

        let mut t_fs = Blake3Transcript::new(b"fsmt");
        let outputs =
            FsMtPcc::reduce_amortized(&pcc_params, &pcc_stmts, &pcc_wits, &mut t_fs);

        let pco_params = PcoParams { srs: m_params.srs.clone() };
        let x = outputs[0].1.point;
        for (_, pco_s, pco_w) in &outputs {
            assert_eq!(pco_s.point, x);
            assert!(PcoRelation::is_satisfied(&pco_params, pco_s, pco_w));
        }

        // Each leaf can also be independently verified through MFold then FsMtPcc.
        for (i, (proof, prover_pco, _)) in outputs.iter().enumerate() {
            let mut t_v = Blake3Transcript::new(b"test");
            let _ = MFold::verify(
                &m_params,
                i,
                &leaves_stmts[i],
                &paths[i],
                &root.0,
                &mut t_v,
            )
            .expect("MFold verify must succeed");

            let mut t_fsv = Blake3Transcript::new(b"fsmt");
            let pco_v =
                FsMtPcc::verify(&pcc_params, i, &pcc_stmts[i], proof, &mut t_fsv)
                    .expect("FsMtPcc verify must succeed");
            assert_eq!(*prover_pco, pco_v);
        }
    }
}
