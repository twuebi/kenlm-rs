use std::hash::Hash;
use std::ops::Deref;

pub mod arpa;

#[derive(Debug, Clone, Copy)]
pub struct ProbBackoff {
    pub log_prob: f32,
    pub backoff: f32,
}

pub trait NGramRep: ToByteVec {}

pub trait ToByteVec {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_word_chunks().into_iter().flatten().collect()
    }
    fn to_word_chunks(&self) -> Vec<Vec<u8>>;
}

impl ToByteVec for String {
    fn to_word_chunks(&self) -> Vec<Vec<u8>> {
        self.split_ascii_whitespace()
            .map(|a| a.as_bytes().iter().cloned().collect())
            .collect()
    }
}

macro_rules! impl_byte_vec_for_int {
    (
        $(
            $dtype:ident
        );*
    ) => {

          $(impl ToByteVec for Vec<$dtype> {
            fn to_word_chunks(&self) -> Vec<Vec<u8>> {
                self.iter().rev().map(|usize| usize.to_le_bytes().to_vec()).collect()
            }
          }
          )+
        }

}

impl_byte_vec_for_int! {usize}
impl_byte_vec_for_int! {u32}
impl_byte_vec_for_int! {u16}
impl_byte_vec_for_int! {u8}

impl ToByteVec for () {
    fn to_word_chunks(&self) -> Vec<Vec<u8>> {
        vec![]
    }
}

impl NGramRep for String {}
impl NGramRep for Vec<usize> {}
impl NGramRep for Vec<u32> {}
impl NGramRep for Vec<u16> {}
impl NGramRep for Vec<u8> {}
impl NGramRep for () {}

impl<T> Deref for NGram<T>
where
    T: NGramRep,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NGram<T: NGramRep>(T);

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
