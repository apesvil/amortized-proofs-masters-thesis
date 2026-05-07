use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_ec::VariableBaseMSM;
use ark_ff::Zero;

// Standard MSM over a contiguous slice of SRS bases: Σᵢ coeffs[i] · srs[i].
pub(crate) fn msm_dense(srs: &[G1Affine], coeffs: &[Fr]) -> G1Projective {
    if coeffs.is_empty() {
        return G1Projective::zero();
    }
    G1Projective::msm(&srs[..coeffs.len()], coeffs).expect("msm failed")
}

// Sparse MSM: pulls only the bases at the specified degrees, then runs an
// MSM of size k = terms.len() instead of size max_deg + 1. Duplicate degrees are
// summed naturally because they appear as separate (base, scalar) pairs.
pub(crate) fn msm_sparse(srs: &[G1Affine], terms: &[(usize, Fr)]) -> G1Projective {
    if terms.is_empty() {
        return G1Projective::zero();
    }
    let bases: Vec<G1Affine> = terms.iter().map(|(d, _)| srs[*d]).collect();
    let scalars: Vec<Fr> = terms.iter().map(|(_, c)| *c).collect();
    G1Projective::msm(&bases, &scalars).expect("msm failed")
}
