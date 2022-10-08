use std::hash::Hash;

pub mod arpa;

#[derive(Debug, Clone)]
pub struct ProbBackoff {
    pub log_prob: f32,
    pub backoff: f32,
}

pub trait NGramRep: std::fmt::Debug + Clone + PartialEq + Eq + Hash {}

impl NGramRep for String {}
impl NGramRep for Vec<usize> {}
impl NGramRep for Vec<u32> {}
impl NGramRep for Vec<u16> {}
impl NGramRep for Vec<u8> {}
impl NGramRep for () {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NGram<T>(T)
where
    T: NGramRep; // TODO: this sensible?

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sequence {
    String(String),
    Ints(Vec<u32>),
}

#[derive(Debug, Clone)]
pub struct ProbBackoffNgram<T>
where
    T: NGramRep,
{
    pub ngram: NGram<T>,
    pub prob_backoff: ProbBackoff,
}

#[derive(Debug, Clone)]
pub struct ProbNgram<T>
where
    T: NGramRep,
{
    pub ngram: NGram<T>,
    pub prob: f32,
}
