use std::{cell::RefCell, ops::Deref, rc::Rc};

use ::cxx::UniquePtr;

use crate::Error;

use self::bridge::VocabFetchCallback;

pub(crate) mod bridge;

pub struct CxxModel(UniquePtr<bridge::lm::base::Model>);

impl Deref for CxxModel {
    type Target = UniquePtr<bridge::lm::base::Model>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl CxxModel {
    pub fn load_from_file_with_config(filename: &str, config: &Config) -> Self {
        cxx::let_cxx_string!(file_name = filename);
        Self(bridge::lm::base::LoadVirtualPtr(&file_name, &config.inner))
    }
}

pub struct Config {
    inner: UniquePtr<bridge::lm::ngram::Config>,
    vocab_callback: Option<Rc<RefCell<VocabFetchCallback>>>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            inner: bridge::lm::base::Config_Create(),
            vocab_callback: None,
        }
    }
}

impl Config {
    pub fn set_load_method(&mut self, method: LoadMethod) -> Result<(), Error> {
        bridge::lm::ngram::Config_set_load_method(
            self.inner
                .as_mut()
                // If this is null, then this is a bug and no Error will help here.
                .unwrap(),
            bridge::util::LoadMethod::from(method),
        );
        Ok(())
    }

    pub fn add_vocab_fetch_callback(&mut self) {
        let callback = bridge::get_vocab_call_back();
        let cb = callback.clone();
        let mut callback_ref = callback.borrow_mut();
        let callback_pin_mut = callback_ref.pin_mut();
        bridge::lm::ngram::Config_set_enumerate_callback(
            // There should always be a config here given that Default creates one.
            self.inner.as_mut().unwrap(),
            callback_pin_mut,
        );
        self.vocab_callback = Some(cb);
    }

    pub fn get_vocab(&mut self) -> Option<Vec<String>> {
        if let Some(voc) = self.vocab_callback.as_ref() {
            let mut vocab = vec![];
            std::mem::swap(&mut voc.borrow_mut().vocab, &mut vocab);
            return Some(vocab);
        }
        None
    }
}

#[derive(Debug, Copy, Clone)]
pub enum LoadMethod {
    Lazy,
    PopulateOrRead,
    PopulateOrLazy,
    Read,
    ParallelRead,
}

impl From<LoadMethod> for bridge::util::LoadMethod {
    fn from(method: LoadMethod) -> Self {
        match method {
            LoadMethod::Lazy => Self::LAZY,
            LoadMethod::PopulateOrRead => Self::POPULATE_OR_READ,
            LoadMethod::Read => Self::READ,
            LoadMethod::ParallelRead => Self::PARALLEL_READ,
            LoadMethod::PopulateOrLazy => Self::POPULATE_OR_LAZY,
        }
    }
}
