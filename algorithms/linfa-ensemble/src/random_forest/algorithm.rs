use linfa::{
    dataset::{AsSingleTargets, Labels},
    traits::Fit,
    Dataset, Float, Label,
};
use linfa_trees::{DecisionTree, DecisionTreeParams};
use ndarray::{Array, ArrayBase, Axis, Data, Ix2, Array0, Array1};
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

impl<F: Float, L: Label> RandomForest<F, L> 
where f64: PartialEq<F>{
    pub fn feature_importances(&self, n_features: usize) -> Vec<F> {
        let mut importances: Array1<F> = Array1::zeros(n_features);
        for tree in &self.trees {
            // features in the single tree
           let tree_importance = tree.feature_importance();
            for i in 0..n_features {
                importances[i] += tree_importance[i];
            }
        }

        let mut sorted_imp = importances.to_vec();
        sorted_imp.sort_by(|a, b| b.partial_cmp(a).unwrap());
        importances.iter().map(|&x| x / importances.sum()).collect()
    }
}

#[cfg(test)]
mod tests {
    use linfa::{traits::Fit, Dataset};
    use linfa_trees::DecisionTreeParams;
    use ndarray::{Array, Array2};
    use ndarray_rand::rand::{rngs::StdRng, SeedableRng, Rng};

    use crate::RandomForest;

    #[test]
    fn test_feat_importance() {
        // Generate a random dataset with 10 features and 100 data points    
        let mut rng = StdRng::seed_from_u64(0);
        let mut features = Array2::zeros((10, 100));
        for i in 0..10 {
            for j in 0..100 {
                println!("i:{i}, j:{j}");
                features[[i, j]] = rng.gen_range(0.1..1.0);
            }
        }
        let target = Array::from((0..100).map(|_| rng.gen_range(0..2)).collect::<Vec<usize>>());
        let dt_params = DecisionTreeParams::<f64, usize>::new().max_depth(Some(3));
        let rf_params = RandomForest::params(100, dt_params, true, None);
        let rf = rf_params.fit(&Dataset::new(features, target)).unwrap();

        let fi = rf.feature_importances(10);
        assert_eq!(fi.iter().fold(0., |acc, x| acc + x) +- 2e-10 , 1.0 +- 2e-10);
    }
}
