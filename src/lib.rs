#![doc = include_str!("../README.md")]

mod cxx;
pub mod headers;
pub(crate) mod model;
pub mod reader;

pub use crate::cxx::LoadMethod;

use headers::InvalidCounts;
pub use model::{Model, State, WordIdx};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("The model you are trying to load is of higher order as the current compilation allows. Set the env var `KENLM_MAX_ORDER={max_order:?}` at build time to use this model with order {model_order:?}")]
    IncompatibleMaxOrder {
        max_order: usize,
        model_order: usize,
    },
    #[error("This model not have a vocabulary, cannot enumerate it to copy into rust-land.")]
    ModelHasNoVocab,
    #[error("Decoding the fixed width parameter header failed, likely the model file is broken or incompatible.")]
    ParamHeaderFormatError,
    #[error("Decoding the count header failed, likely the model file is broken or incompatible.")]
    CountHeaderError(#[from] InvalidCounts),
    #[error("Decoding the sanity header failed, likely the model file is broken or incompatible.")]
    SanityFormatError,
    #[error("The sanity header did not match the reference header. Likely the model is broken or incompatible.")]
    SanityMismatch,
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}
