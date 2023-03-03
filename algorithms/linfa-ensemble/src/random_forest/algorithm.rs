use std::collections::HashMap;

use linfa::{
    dataset::{AsSingleTargets, Labels},
    traits::Fit,
    Dataset, Float, Label,
};
use linfa_trees::{DecisionTree, DecisionTreeParams};
use ndarray::{Array, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::RandomForestParams;

#[derive(Clone, Debug)]
pub struct RandomForest<F, L>
where
    F: Float,
    L: Label,
{
    pub(crate) trees: Vec<DecisionTree<F, L>>,
}

impl<F: Float, L: Label> RandomForest<F, L> {
    pub fn params(
        n_estimators: usize,
        tree_hyperparameters: DecisionTreeParams<F, L>,
        bootstrap: bool,
        max_n_rows: Option<usize>,
    ) -> RandomForestParams<F, L> {
        RandomForestParams::new(n_estimators, tree_hyperparameters, bootstrap, max_n_rows)
    }
}

impl<F: Float, L: Label, D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>
    Fit<ArrayBase<D, Ix2>, T, linfa::Error> for RandomForestParams<F, L>
{
    type Object = RandomForest<F, L>;

    fn fit(
        &self,
        dataset: &linfa::DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::Object, linfa::Error> {
        let mut trees: Vec<DecisionTree<F, L>> = Vec::with_capacity(self.n_estimators);
        let max_n_rows = self.max_n_rows.unwrap_or(dataset.records.nrows());

        let y = dataset.as_single_targets();

        for _ in 0..self.n_estimators {
            let rnd_idx = Array::random((1, max_n_rows), Uniform::new(0, dataset.records.nrows()))
                .into_raw_vec();
            let xsample = dataset.records.select(Axis(0), &rnd_idx);
            let ysample = y.select(Axis(0), &rnd_idx);
            let dsample = Dataset::new(xsample, ysample);
            let tree = self.tree_hyperparameters.fit(&dsample)?;
            trees.push(tree);
        }

        Ok(RandomForest { trees })
    }
}

impl<F: Float, L: Label> RandomForest<F, L> {
    pub fn feature_importances(&self) -> HashMap<usize, usize> {
        let mut importances: HashMap<usize, usize> = HashMap::new();
        for tree in &self.trees {
            for feat in tree.features() {
                *importances.entry(feat).or_insert(0) += 1
            }
        }

        importances
    }
}
