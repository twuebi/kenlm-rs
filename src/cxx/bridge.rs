// autocxx generates some stuff that makes clippy angry
#![allow(clippy::all)]

use ::std::cell::RefCell;
use ::std::mem::size_of;
use ::std::rc::Rc;

use autocxx::subclass::CppSubclassDefault;
use autocxx::{prelude::*, subclass::is_subclass};

include_cpp! {
    #include "lm/virtual_interface.hh"
    #include "lm/config.hh"
    #include "lm/state.hh"
    #include "util/mmap.hh"
    #include "lm/enumerate_vocab.hh"
    #include "util/string_piece.hh"
    #include "lm/max_order.hh"
    #include "lm/binary_format.hh"
    #include "lm/facade.hh"

    safety!(unsafe)
    generate!("lm::ngram::SizeOfSanity")
    generate!("lm::ngram::ModelMaxOrder")
    generate_pod!("lm::ngram::FixedWidthParameters")
    generate_pod!("lm::ngram::State")

    generate!("util::LoadMethod")
    generate!("lm::base::Model")
    generate!("lm::base::Vocabulary")
    generate!("lm::base::LoadVirtualPtr")

    generate!("lm::ngram::Config")
    generate!("lm::base::Config_Create")
    generate!("lm::ngram::Config_set_load_method")
    generate!("lm::ngram::Config_set_enumerate_callback")
    generate!("lm::WordIndex")
    generate!("StringPiece")
    subclass!("lm::EnumerateVocab", VocabFetchCallback)
}

pub(crate) use ffi::*;
use lm::EnumerateVocab_methods;
use lm::WordIndex;

impl Clone for lm::ngram::State {
    fn clone(&self) -> Self {
        Self {
            words: self.words,
            backoff: self.backoff,
            length: self.length,
        }
    }
}

#[is_subclass(superclass("EnumerateVocab"))]
#[derive(Default)]
pub struct VocabFetchCallback {
    pub vocab: Vec<String>,
}

impl EnumerateVocab_methods for VocabFetchCallback {
    fn Add(&mut self, index: WordIndex, string: &StringPiece) {
        // make clippy happy
        let _ = index;
        let string = string
            .as_string()
            .as_ref()
            // safety: this should ever only be none if the kenlm
            //         vocab contains a null ptr which means a bug
            //         over there. Since this is called from C++
            //         and kenlm dictates its signature no Result
            //         here either.
            .unwrap()
            .to_string();

        self.vocab.push(string);
    }
}

pub fn size_of_sanity_header() -> u64 {
    u64::from(ffi::lm::ngram::SizeOfSanity())
}

pub fn get_max_order() -> u8 {
    u8::from(ffi::lm::ngram::ModelMaxOrder())
}

pub fn get_size_of_fixed_width_params() -> usize {
    size_of::<ffi::lm::ngram::FixedWidthParameters>()
}

pub fn get_vocab_call_back() -> Rc<RefCell<VocabFetchCallback>> {
    VocabFetchCallback::default_rust_owned()
}
