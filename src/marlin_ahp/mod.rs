//! Marlin's AHP inner sumcheck ("lincheck"), used to discharge our R_P
//! statement `(α, β, y, η)` with `y = P_A(α,β) + η·P_B(α,β) + η²·P_C(α,β)`.
//!
//! The arithmetization (`arithmetize.rs`) is **vendored** from
//! `arkworks-rs/marlin` (`src/ahp/constraint_systems.rs`, MIT/Apache-2.0).
//! `bivariate.rs` carries the `UnnormalizedBivariateLagrangePoly` trait it
//! depends on (also vendored, from the same crate's `src/ahp/mod.rs`).
//! The protocol itself (`inner_lincheck.rs`) is implemented fresh on top of
//! our existing KZG + Blake3 transcript, following Marlin §5.3.

//! `outer_sumcheck.rs` ports Marlin's rounds 1-2 — the witness-dependent half
//! of the prover — from the same upstream crate, so that a full Marlin proving
//! time can be measured in one place and split into its witness-dependent and
//! witness-independent parts. `r1cs.rs` supplies the satisfying assignment
//! those rounds need.

pub mod arithmetize;
pub mod bivariate;
pub mod inner_lincheck;
pub mod outer_sumcheck;
pub mod r1cs;

#[allow(unused_imports)] // public surface, no internal consumer in this binary crate
pub use arithmetize::{arithmetize, MatrixArithmetization, MatrixEvals};
#[allow(unused_imports)] // public surface, no internal consumer in this binary crate
pub use bivariate::UnnormalizedBivariateLagrangePoly;
#[allow(unused_imports)] // public surface, no internal consumer in this binary crate
pub use inner_lincheck::{InnerLincheck, InnerLincheckIndex, InnerLincheckProof};
#[allow(unused_imports)] // public surface, no internal consumer in this binary crate
pub use outer_sumcheck::{
    repo_p_statement, OuterClaim, OuterFirstRound, OuterProof, OuterSumcheck,
};
#[allow(unused_imports)] // public surface, no internal consumer in this binary crate
pub use r1cs::{satisfiable_instance, R1csInstance};
