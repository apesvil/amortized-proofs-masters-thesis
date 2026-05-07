use ark_bls12_381::{Fr, G1Affine, G2Affine};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{One, PrimeField, UniformRand};
use ark_std::rand::RngCore;

/// Structured reference string: SRS = { [τⁱg]_{i=0..d} in G1, h and τh in G2 }.
#[derive(Clone)]
pub struct Srs {
    pub(crate) powers_g1: Vec<G1Affine>, // [g, τg, τ²g, ..., τᵈg]
    pub(crate) g2: G2Affine,             // h
    pub(crate) tau_g2: G2Affine,         // τ·h
}

/// Generates the SRS for polynomials up to `max_degree`.
/// In production, `τ` comes from a trusted setup ceremony; here it is sampled locally.
pub fn setup(max_degree: usize, rng: &mut impl RngCore) -> Srs {
    let tau = Fr::rand(rng);
    let g = G1Affine::generator();
    let h = G2Affine::generator();

    let mut powers_g1 = Vec::with_capacity(max_degree + 1);
    let mut tau_i = Fr::one();
    for _ in 0..=max_degree {
        powers_g1.push(g.mul_bigint(tau_i.into_bigint()).into_affine());
        tau_i *= tau;
    }

    Srs {
        powers_g1,
        g2: h,
        tau_g2: h.mul_bigint(tau.into_bigint()).into_affine(),
    }
}
