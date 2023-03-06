use thiserror::Error;

#[derive(Error, Debug)]
pub enum RandomForestError {
    #[error("Invalid hyperparameter: {0}")]
    InvalidParams(#[from] RandomForestParamsError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}

#[derive(Error, Debug)]
pub enum RandomForestParamsError {
    #[error("n_estimators cannot be 0")]
    NEstimators,
}
