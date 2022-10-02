use crate::headers::{CountHeader, FixedParameterHeader, SanityHeader};
use crate::{headers, Error};

use crate::cxx::bridge::get_max_order;

use super::Model;

pub(crate) struct ModelBuilder {
    vocab: bool,
    file_name: String,
}

impl ModelBuilder {
    pub(crate) fn new(file_name: &str) -> Self {
        Self {
            vocab: false,
            file_name: file_name.into(),
        }
    }

    pub(crate) fn store_vocab(mut self, store_vocab: bool) -> Self {
        self.vocab = store_vocab;
        self
    }

    pub(crate) fn get_fd(&self) -> Result<std::fs::File, Error> {
        std::fs::File::open(&self.file_name)
            .map_err(|_| Error::FileNotFound(self.file_name.to_string()))
    }

    fn verify_sanity(&self, sanity_header: SanityHeader) -> Result<(), Error> {
        if sanity_header != SanityHeader::REFERENCE {
            eprintln!(
                "Sanity header does not match the reference: \n{sanity_header:?} \nvs\n{:?}",
                SanityHeader::REFERENCE
            );

            return Err(Error::SanityFormatError);
        }
        Ok(())
    }

    pub(crate) fn verify(&self, fixed_params: &FixedParameterHeader) -> Result<(), Error> {
        if get_max_order() < fixed_params.order {
            return Err(Error::IncompatibleMaxOrder {
                max_order: get_max_order().into(),
                model_order: fixed_params.order.into(),
            });
        }
        if self.vocab && !fixed_params.has_vocabulary() {
            return Err(Error::ModelHasNoVocab);
        }
        Ok(())
    }

    pub(crate) fn build(self) -> Result<Model, Error> {
        let mut fd = self.get_fd()?;

        let sanity_header = SanityHeader::from_file(&mut fd)?;
        self.verify_sanity(sanity_header)?;
        let fixed_params = headers::FixedParameterHeader::from_file(&mut fd)?;
        self.verify(&fixed_params)?;
        let counts = CountHeader::from_file(&mut fd, &fixed_params)?;
        let mut config = crate::cxx::Config::default();
        let inner = {
            config.set_load_method(crate::cxx::LoadMethod::Lazy)?;

            if self.vocab {
                config.add_vocab_fetch_callback();
            };
            crate::cxx::CxxModel::load_from_file_with_config(&self.file_name, &config)
            // bridge::lm::base::LoadVirtualPtr(&file_name, &config)
        };

        Ok(Model {
            inner,
            vocab: config.get_vocab(),
            fixed_parameters: fixed_params,
            count_header: counts,
        })
    }
}
