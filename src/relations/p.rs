use ark_bls12_381::Fr;

/// Public statement of R_P (page 27 of the updated paper):
/// the implicit claim is `y = λ(α)ᵀ·M·λ(β)` for a fixed matrix `M`.
///
/// The paper formalizes R_P as `(pp = ε, x = (α, β, y), w = ε)`. We provide
/// only the statement struct here — no `PParams`, no `PWitness`, and no
/// `is_satisfied` impl — because R_P is the project's benchmark boundary:
/// it will be discharged by different SNARKs depending on the experiment,
/// and each backend wants its own verification routine.
#[derive(Clone, Debug, PartialEq)]
pub struct PStatement {
    pub alpha: Fr,
    pub beta: Fr,
    pub y: Fr,
}
