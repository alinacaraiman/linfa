use linfa::{Float, Label};
use linfa_trees::DecisionTreeParams;

pub struct RandomForestParams<F: Float, L: Label> {
    pub(crate) n_estimators: usize,
    pub(crate) tree_hyperparameters: DecisionTreeParams<F, L>,
    pub(crate) max_features: u64,
    pub(crate) bootstrap: bool,
    pub max_n_rows: Option<usize>,
}

impl<F: Float, L: Label> RandomForestParams<F, L> {
    pub(crate) fn new(
        n_estimators: usize,
        tree_hyperparameters: DecisionTreeParams<F, L>,
        bootstrap: bool,
        max_n_rows: Option<usize>,
    ) -> Self {
        Self {
            n_estimators,
            tree_hyperparameters,
            max_features: 42,
            bootstrap,
            max_n_rows,
        }
    }
}
