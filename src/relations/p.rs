use ark_bls12_381::Fr;

/// Public statement of R_P (page 27 of the updated paper, extended to the
/// three-matrix R1CS setting). The implicit claim is
/// ```text
///   y = P_A(α, β) + η·P_B(α, β) + η²·P_C(α, β)
/// ```
/// where `P_X(X, Y) := λ(X)ᵀ·X·λ(Y)` is the bivariate polynomial encoding
/// of the R1CS matrix `X` for `X ∈ {A, B, C}`. The discharging SNARK
/// receives `(α, β, y, η)` and verifies that identity directly.
///
/// The paper formalizes R_P as `(pp = ε, x, w = ε)`. We provide only the
/// statement struct here — no `PParams`, no `PWitness`, and no
/// `is_satisfied` impl — because R_P is the project's benchmark boundary:
/// it will be discharged by different SNARKs depending on the experiment,
/// and each backend wants its own verification routine.
#[derive(Clone, Debug, PartialEq)]
pub struct PStatement {
    pub alpha: Fr,
    pub beta: Fr,
    pub y: Fr,
    pub eta: Fr,
}
