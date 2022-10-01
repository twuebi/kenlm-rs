pub mod bridge;
use std::{
    cell::{RefCell, RefMut},
    rc::Rc,
};

use autocxx::{prelude::*, subclass::CppSubclassDefault};
use bridge::VocabCallback;

pub struct Model {
    inner: UniquePtr<bridge::lm::base::Model>,
    pub vocab: Option<Vec<String>>,
}

impl Model {
    /// Initializes the model from [file_name], initialization happens in C++ land. Setting
    /// [store_vocab] adds a callback to the model config object which copies the vocab into
    /// this Rust-land struct.
    /// For some models, this may lead to duplication in memory, in others, e.g. trie-format,
    /// this leads to increased memory usage, dependent on the model size this can use quite
    /// a lot of memory.
    /// If you run out of memory or don't need the vocab, consider passing false here.
    pub fn new(file_name: &str, store_vocab: bool) -> Self {
        let mut builder = ModelBuilder { vocab: None };

        cxx::let_cxx_string!(file_name = &file_name);

        let inner = {
            let mut config: UniquePtr<bridge::lm::ngram::Config> =
                bridge::lm::base::Config_Create();
            bridge::lm::ngram::Config_set_load_method(
                config.as_mut().unwrap(),
                bridge::util::LoadMethod::LAZY,
            );

            if store_vocab {
                builder = builder.with_vocab(bridge::VocabCallback::default_rust_owned());
                let mut callback_ref = builder.borrow_vocab_mut().unwrap();
                let callback_pin_mut = callback_ref.pin_mut();
                bridge::lm::ngram::Config_set_enumerate_callback(
                    config.as_mut().unwrap(),
                    callback_pin_mut,
                );
            };
            bridge::lm::base::LoadVirtualPtr(&file_name, &config)
        };
        builder.build(inner)
    }

    pub fn state_size(&self) -> usize {
        self.inner.StateSize()
    }

    /// Get the index of a word in the language model, returns None if
    /// the vocab does not contain the word.
    pub fn get_word_idx(&self, word: &str) -> Option<c_uint> {
        let vocab = self.inner.BaseVocabulary();
        cxx::let_cxx_string!(input = &word);
        let idx = vocab.Index1(&input);
        if idx == vocab.NotFound() {
            return None;
        }
        Some(idx)
    }

    /// If you use this function swap in_state and out_state between calls.
    /// You could also create a new out_state every time but that would be
    /// wasteful. See below for an example or go and check score_sentence.
    ///
    /// Or take a look at https://github.com/kpu/kenlm/blob/master/python/score_sentence.cc
    ///
    /// let mut mem1 = self.get_sentence_state();
    /// let mut mem2 = self.get_sentence_state();
    ///
    /// if bos {
    ///     mem1 = self.begin_sentence_bos(mem1);
    /// } else {
    ///     mem1 = self.begin_sentence_null(mem1);
    /// }
    /// let mut score = 0f32;
    /// for w in sentence {
    ///     let out = self.score_word_given_state(&mut mem1, &mut mem2, w);
    ///     std::mem::swap(&mut mem1, &mut mem2);
    ///     score += out;
    /// }
    pub fn score_word_given_state(
        &self,
        in_state: &mut StateWrapper,
        out_state: &mut StateWrapper,
        word: &str,
    ) -> f32 {
        let vocab = self.inner.BaseVocabulary();
        cxx::let_cxx_string!(input = &word);
        let index = vocab.IndexString(&input);
        self.score_index_given_state(in_state, out_state, index)
    }

    pub fn score_index_given_state(
        &self,
        in_state: &mut StateWrapper,
        out_state: &mut StateWrapper,
        index: c_uint,
    ) -> f32 {
        let in_state = in_state.0.pin_mut();
        let s = std::pin::Pin::<&mut bridge::lm::ngram::State>::into_inner(in_state);
        let ptr = s as *mut bridge::lm::ngram::State;
        let raw1 = ptr as *mut autocxx::c_void;

        let out_state = out_state.0.pin_mut();
        let s2 = std::pin::Pin::<&mut bridge::lm::ngram::State>::into_inner(out_state);
        let ptr2 = s2 as *mut bridge::lm::ngram::State;
        let raw2 = ptr2 as *mut autocxx::c_void;
        unsafe { self.inner.BaseScore(raw1, index, raw2) }
    }

    pub fn score_sentence(&self, sentence: &[&str], bos: bool, eos: bool) -> f32 {
        let vocab = self.inner.BaseVocabulary();

        let mut mem1 = self.new_state();
        let mut mem2 = self.new_state();
        if bos {
            self.fill_state_with_bos_context(&mut mem1);
        } else {
            self.fill_state_with_null_context(&mut mem1);
        }

        let mut score = 0f32;

        for w in sentence {
            let out = self.score_word_given_state(&mut mem1, &mut mem2, w);
            std::mem::swap(&mut mem1, &mut mem2);
            score += out;
        }

        if eos {
            let out = self.score_index_given_state(&mut mem1, &mut mem2, vocab.EndSentence());
            score += out;
        }

        score
    }

    pub fn new_state(&self) -> StateWrapper {
        StateWrapper::new(self)
    }

    pub fn fill_state_with_bos_context(&self, state: &mut StateWrapper) {
        let in_state = state.0.pin_mut();
        let s = std::pin::Pin::<&mut bridge::lm::ngram::State>::into_inner(in_state);
        let ptr = s as *mut bridge::lm::ngram::State;
        let raw = ptr as *mut autocxx::c_void;
        unsafe { self.inner.BeginSentenceWrite(raw) }
    }

    pub fn fill_state_with_null_context(&self, state: &mut StateWrapper) {
        let in_state = state.0.pin_mut();
        let s = std::pin::Pin::<&mut bridge::lm::ngram::State>::into_inner(in_state);
        let ptr = s as *mut bridge::lm::ngram::State;
        let raw = ptr as *mut autocxx::c_void;
        unsafe { self.inner.NullContextWrite(raw) }
    }
}

struct ModelBuilder {
    vocab: Option<Rc<RefCell<VocabCallback>>>,
}

impl ModelBuilder {
    fn with_vocab(mut self, vocab: Rc<RefCell<VocabCallback>>) -> Self {
        self.vocab = Some(vocab);
        self
    }

    fn borrow_vocab_mut(&'_ self) -> Option<RefMut<'_, VocabCallback>> {
        self.vocab.as_deref().map(|voc| voc.borrow_mut())
    }

    fn build(self, inner: UniquePtr<bridge::lm::base::Model>) -> Model {
        if let Some(voc) = self.vocab {
            let mut vocab = vec![];
            std::mem::swap(&mut voc.borrow_mut().vocab, &mut vocab);
            Model {
                inner,
                vocab: Some(vocab),
            }
        } else {
            Model { inner, vocab: None }
        }
    }
}

#[derive(Debug)]
pub struct StateWrapper(UniquePtr<bridge::lm::ngram::State>);

impl StateWrapper {
    fn new(model: &Model) -> Self {
        let size = std::mem::size_of::<bridge::lm::ngram::State>();
        let model_size = model.state_size();
        if size != model_size {
            eprintln!("size of bridge::lm::ngram::State: {size} does not match size returned by StateSize: {model_size}");
        }
        let state = bridge::lm::ngram::State::new().within_unique_ptr();
        Self(state)
    }
}

/// Panics if Self::0 contains a null-pointer
impl Clone for StateWrapper {
    fn clone(&self) -> Self {
        Self(self.0.as_ref().unwrap().clone().within_unique_ptr())
    }
}

impl std::fmt::Debug for bridge::lm::ngram::State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("words", &self.words)
            .field("backoff", &self.backoff)
            .field("length", &self.length)
            .finish()
    }
}
