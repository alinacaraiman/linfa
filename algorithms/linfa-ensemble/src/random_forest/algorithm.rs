use std::collections::HashMap;

use linfa::{
    dataset::{AsSingleTargets, Labels},
    traits::{Fit, PredictInplace, Predict},
    Dataset, Float, Label,
};
use linfa_trees::{DecisionTree, DecisionTreeParams};
use ndarray::{Array, ArrayBase, Axis, Data, Ix2, Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::RandomForestParams;

#[derive(Clone, Debug)]
pub struct RandomForest<F, L>
where
    F: Float,
    L: Label,
{
    pub(crate) trees: Vec<DecisionTree<F, L>>,
    n_estimators: usize,
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

        Ok(RandomForest { trees , n_estimators: self.n_estimators })
    }
}

// impl<F: Float, L: Label + ndarray_rand::rand_distr::num_traits::Zero, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>> for RandomForest<F, L> {
//     fn predict_inplace<'a>(&'a self, x: &'a ArrayBase<D, Ix2>, y: &mut Array1<L>) {
//         let mut predictions: Array2<L> = Array2::zeros((self.n_estimators, x.nrows()));
//         for (row, target) in x.rows().into_iter().zip(y.iter_mut()) {

//         }

//         for i in 0..self.n_estimators {
//             let pred = self.trees[i].predict(x);

//             for j in 0..pred.len() {
//                 predictions[[i, j]] = pred[j].clone();
//             }
//         }

//         let mut result: Vec<L> = Vec::with_capacity(x.nrows());

//         let flattened: Vec<Vec<L>> = self
//             .trees
//             .iter()
//             .map(|tree| tree.predict(x).to_vec())
//             .collect();
//         for j in 0..predictions.ncols() {
//             // hashmap to store most common prediction across trees
//             let mut counter_stats: HashMap<L, u64> = HashMap::new();
//             for i in 0..self.n_estimators {
//                 *counter_stats.entry(predictions[[i,j]]).or_insert(0) += 1;
//             }

//             let final_pred = counter_stats
//                 .iter()
//                 .max_by(|a, b| a.1.cmp(&b.1))
//                 .map(|(k, _v)| k)
//                 .unwrap();

//             result.push(*final_pred);
//         }
//     }

//     fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
//         Array1::default(x.nrows())
//     }
// }

impl<F: Float, L: Label> RandomForest<F, L> {
    pub fn feature_importances(&self) -> Vec<usize> {
        let mut importances: HashMap<usize, usize> = HashMap::new();
        for st in &self.trees {
            // features in the single tree
            let st_feats = st.features();
            for f in st_feats.iter() {
                *importances.entry(*f).or_insert(0) += 1
            }
        }

        let mut top_feats: Vec<_> = importances.into_iter().collect();
        top_feats.sort_by(|a, b| b.1.cmp(&a.1));

        top_feats.iter().map(|(a, _)| *a).collect()
    }
}
