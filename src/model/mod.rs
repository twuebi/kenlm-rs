mod builder;

use std::ops::Deref;

use crate::headers::FixedParameterHeader;
use crate::Error;
use autocxx::prelude::*;

use crate::cxx::{bridge, CxxModel};

use self::builder::ModelBuilder;

pub struct Model {
    inner: CxxModel,
    vocab: Option<Vec<String>>,
    fixed_parameters: FixedParameterHeader,
}

impl Model {
    /// Initializes the model from `file_name`, stores vocab if `store_vocab` is true.
    ///
    /// Initializes the model from `file_name`, initialization happens in C++ land. Setting
    /// `store_vocab` adds a callback to the model config object which copies the vocab into
    /// this Rust-land struct. A vocab can only be stored if the supplied model has one, hence
    /// this constructor will return an error if the [FixedParameterHeader] of `file_name`
    /// contains `has_vocabulary=false` and you supply `store_vocab=true`.
    ///
    /// For some models, loading a vocab may lead to duplication in memory, in others, e.g.
    /// trie-format, this may lead to increased memory usage, dependent on the model size this
    /// can use quite a lot of memory.
    /// If you run out of memory or don't need the vocab, consider not storing the vocab here.
    pub fn new(file_name: &str, store_vocab: bool) -> Result<Self, Error> {
        ModelBuilder::new(file_name)
            .store_vocab(store_vocab)
            .build()
    }

    pub fn state_size(&self) -> usize {
        self.inner.StateSize()
    }

    pub fn get_fixed_parameter_header(&self) -> &FixedParameterHeader {
        &self.fixed_parameters
    }

    /// Get the index of a word in the language model
    ///
    /// returns None if the vocab does not contain the word.
    pub fn get_word_idx_opt(&self, word: &str) -> Option<WordIdx> {
        let vocab = self.inner.BaseVocabulary();
        cxx::let_cxx_string!(input = &word);
        let idx = vocab.Index1(&input);
        if idx == vocab.NotFound() {
            return None;
        }
        Some(WordIdx(idx))
    }

    /// Get the index of a word in the language model
    ///
    /// returns vocab.NotFound() if the vocab does not contain the word.
    pub fn get_word_idx(&self, word: &str) -> WordIdx {
        let vocab = self.inner.BaseVocabulary();
        cxx::let_cxx_string!(input = &word);
        let idx = vocab.Index1(&input);
        WordIdx(idx)
    }

    /// Score a word (suffix) given a state (prefix).
    ///
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
        self.score_index_given_state(in_state, out_state, WordIdx(index))
    }

    pub fn score_index_given_state(
        &self,
        in_state: &mut StateWrapper,
        out_state: &mut StateWrapper,
        index: WordIdx,
    ) -> f32 {
        let in_state = in_state.0.pin_mut();
        let s = std::pin::Pin::<&mut bridge::lm::ngram::State>::into_inner(in_state);
        let ptr = s as *mut bridge::lm::ngram::State;
        let raw1 = ptr as *mut autocxx::c_void;

        let out_state = out_state.0.pin_mut();
        let s2 = std::pin::Pin::<&mut bridge::lm::ngram::State>::into_inner(out_state);
        let ptr2 = s2 as *mut bridge::lm::ngram::State;
        let raw2 = ptr2 as *mut autocxx::c_void;
        unsafe { self.inner.BaseScore(raw1, index.0, raw2) }
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
            let out =
                self.score_index_given_state(&mut mem1, &mut mem2, WordIdx(vocab.EndSentence()));
            score += out;
        }

        score
    }

    pub fn new_state(&self) -> StateWrapper {
        let mut state = StateWrapper::new_for_model(self);
        // better safe than sorry i guess?
        self.fill_state_with_null_context(&mut state);
        state
    }

    pub fn get_vocab(&self) -> Option<&[String]> {
        self.vocab.as_deref()
    }

    pub fn get_order(&self) -> u8 {
        self.inner.Order()
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

#[derive(Debug, Clone, Copy)]
pub struct WordIdx(c_uint);

impl Deref for WordIdx {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0 .0
    }
}

#[derive(Debug)]
pub struct StateWrapper(UniquePtr<bridge::lm::ngram::State>);

impl StateWrapper {
    fn new_for_model(model: &Model) -> Self {
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

#[cfg(test)]
mod test {
    use super::{Error, Model};
    pub const TEST_SENTENCE: &[&str] = &[
        "i", "have", "a", "good", "deal", "of", "will", "you", "remember", "and", "what", "i",
        "have", "set", "my", "mind", "upon", "no", "doubt", "i", "shall", "some", "day", "achieve",
    ];

    pub const TEST_WITH_OOV: &[&str] = &[
        "i", "have", "a", "good", "deal", "of", "will", "you", "remember", "and", "what", "i",
        "have", "set", "my", "mind", "upon", "no", "doubt", "i", "shall", "some", "day", "achieve",
        "toast",
    ];

    #[test]
    fn loads() {
        let _model = Model::new("test_data/test.bin", false).expect("should exist");
    }

    #[test]
    fn loads_probing_model() {
        let _model = Model::new("test_data/carol_probing_bigram.bin", false).expect("should exist");
    }

    #[test]
    fn loads_trie_model() {
        let _model = Model::new("test_data/carol_probing_bigram.bin", false).expect("should exist");
    }

    #[test]
    fn does_not_load() {
        let model = Model::new("no-file-to-be-found", false);
        match model {
            Ok(_) => panic!("There should be no file called 'no-file-to-be-found' around here."),
            Err(err) => assert!(matches!(err, Error::FileNotFound(_))),
        }
    }

    #[test]
    fn does_not_enumerate_vocab_without_vocab_in_binary() {
        let model = Model::new("test_data/test_no_vocab.bin", true);
        match model {
            Ok(_) => panic!("There should be no file called 'no-file-to-be-found' around here."),
            Err(err) => assert!(matches!(err, super::Error::ModelHasNoVocab), "{err}"),
        }
    }

    #[test]
    fn loads_without_vocab() {
        let model = Model::new("test_data/test.bin", false).expect("should exist");
        assert!(model.get_vocab().is_none())
    }

    #[test]
    fn loads_with_vocab() {
        let model = Model::new("test_data/test.bin", true).expect("should exist");

        assert_eq!(
            model.get_vocab().unwrap(),
            &[
                "<unk>".to_string(),
                "<s>".to_string(),
                "a".to_string(),
                "will".to_string(),
                "remember".to_string(),
                "set".to_string(),
                "what".to_string(),
                "day".to_string(),
                "mind".to_string(),
                "you".to_string(),
                "</s>".to_string(),
                "deal".to_string(),
                "of".to_string(),
                "have".to_string(),
                "and".to_string(),
                "my".to_string(),
                "some".to_string(),
                "no".to_string(),
                "upon".to_string(),
                "doubt".to_string(),
                "i".to_string(),
                "shall".to_string(),
                "achieve".to_string(),
                "good".to_string()
            ]
        )
    }

    #[test]
    fn score_works() {
        let model = Model::new("test_data/test.bin", true).expect("should exist");
        let mut in_state = model.new_state();
        let mut out_state = model.new_state();
        let score = model.score_word_given_state(&mut in_state, &mut out_state, &"some");
        approx::assert_abs_diff_eq!(-1.3708712f32, score, epsilon = f32::EPSILON);
    }

    #[test]
    fn score_sentence_works() {
        let model = Model::new("test_data/test.bin", false).expect("should exist");
        let score = model.score_sentence(&["some"], false, false);
        approx::assert_abs_diff_eq!(-1.3708712f32, score, epsilon = f32::EPSILON);
    }

    #[test]
    fn score_longer_sentence_works() {
        let model = Model::new("test_data/test.bin", false).expect("should exist");
        let score = model.score_sentence(TEST_SENTENCE, false, false);
        approx::assert_abs_diff_eq!(-4.874725f32, score, epsilon = f32::EPSILON);
    }

    #[test]
    fn score_longer_sentence_bos_eos_with_oov_works() {
        let model = Model::new("test_data/test.bin", false).expect("should exist");
        let score = model.score_sentence(TEST_WITH_OOV, true, true);
        approx::assert_abs_diff_eq!(-7.4208074f32, score, epsilon = f32::EPSILON);
    }

    #[test]
    fn score_longer_sentence_with_oov_works() {
        let model = Model::new("test_data/test.bin", false).expect("should exist");
        let score = model.score_sentence(TEST_WITH_OOV, false, false);
        approx::assert_abs_diff_eq!(-7.1395426f32, score, epsilon = f32::EPSILON);
    }

    #[test]
    fn score_longer_sentence_bos_with_oov_works() {
        let model = Model::new("test_data/test.bin", false).expect("should exist");
        let score = model.score_sentence(TEST_WITH_OOV, true, false);
        approx::assert_abs_diff_eq!(-6.0499362f32, score, epsilon = f32::EPSILON);
    }

    #[test]
    fn score_sentence_bos_works() {
        let model = Model::new("test_data/test.bin", false).expect("should exist");
        let score = model.score_sentence(&["some"], true, false);
        approx::assert_abs_diff_eq!(-1.6719012f32, score, epsilon = f32::EPSILON);
    }

    #[test]
    fn score_sentence_bos_eos_works() {
        let model = Model::new("test_data/test.bin", false).expect("should exist");
        let score = model.score_sentence(&["some"], true, true);
        approx::assert_abs_diff_eq!(-3.3438025f32, score, epsilon = f32::EPSILON);
    }
    struct Example {
        input_word: &'static str,
        word_idx: u32,
        prefix_length: usize,
        score: f32,
    }

    #[test]
    fn states_behave_as_expected() {
        let model = Model::new("test_data/test.bin", true).expect("should exist");
        let mut in_state = model.new_state();
        let mut out_state = model.new_state();
        let expectation = [
            Example {
                input_word: "some",
                word_idx: 16,
                prefix_length: 1,
                score: -1.3708712,
            },
            Example {
                input_word: "game",
                word_idx: 0,
                prefix_length: 0,
                score: -1.9637879,
            },
            Example {
                input_word: "told",
                word_idx: 0,
                prefix_length: 0,
                score: -1.6627579,
            },
            Example {
                input_word: "me",
                word_idx: 0,
                prefix_length: 0,
                score: -1.6627579,
            },
            Example {
                input_word: "that",
                word_idx: 0,
                prefix_length: 0,
                score: -1.6627579,
            },
            Example {
                input_word: "i",
                word_idx: 20,
                prefix_length: 1,
                score: -1.0744861,
            },
            Example {
                input_word: "have",
                word_idx: 13,
                prefix_length: 2,
                score: -0.45023733,
            },
            Example {
                input_word: "a",
                word_idx: 2,
                prefix_length: 2,
                score: -0.41381443,
            },
            Example {
                input_word: "good",
                word_idx: 23,
                prefix_length: 2,
                score: -0.11881906,
            },
            Example {
                input_word: "deal",
                word_idx: 11,
                prefix_length: 2,
                score: -0.11881906,
            },
            Example {
                input_word: "of",
                word_idx: 12,
                prefix_length: 2,
                score: -0.11881906,
            },
            Example {
                input_word: "will",
                word_idx: 3,
                prefix_length: 2,
                score: -0.11881906,
            },
            Example {
                input_word: "you",
                word_idx: 9,
                prefix_length: 2,
                score: -0.11881906,
            },
            Example {
                input_word: "remember",
                word_idx: 4,
                prefix_length: 2,
                score: -0.11881906,
            },
        ];
        for Example {
            input_word,
            word_idx,
            prefix_length,
            score: expected_score,
        } in expectation
        {
            let lookup_word_idx = model.get_word_idx(input_word);
            let c = lookup_word_idx.0 .0;
            assert_eq!(c, word_idx);
            let score = model.score_word_given_state(&mut in_state, &mut out_state, input_word);
            assert_eq!(out_state.0.Length() as usize, prefix_length);
            assert_eq!(out_state.0.words[0].0, word_idx);
            std::mem::swap(&mut in_state, &mut out_state);
            approx::assert_abs_diff_eq!(expected_score, score, epsilon = f32::EPSILON);
        }
    }
}
