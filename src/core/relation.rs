/// A cryptographic relation R ⊆ { (params, statement, witness) }.
pub trait Relation {
    type Params;
    type Statement;
    type Witness;

    fn is_satisfied(
        params: &Self::Params,
        stmt: &Self::Statement,
        wit: &Self::Witness,
    ) -> bool;
}
