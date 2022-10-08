use std::io::{BufReader, Seek, SeekFrom};

use crate::headers::{Counts, FixedParameters, Sanity};
use crate::reader::arpa::ArpaReader;
use crate::{headers, Error, LoadMethod};

use crate::cxx::bridge::get_max_order;

use super::Model;

pub(crate) struct ModelBuilder {
    vocab: bool,
    file_name: String,
    load_method: LoadMethod,
}

impl ModelBuilder {
    pub(crate) fn new(file_name: &str) -> Self {
        Self {
            vocab: false,
            file_name: file_name.into(),
            load_method: LoadMethod::Lazy,
        }
    }

    pub(crate) fn with_load_method(mut self, load_method: LoadMethod) -> Self {
        self.load_method = load_method;
        self
    }

    pub(crate) fn store_vocab(mut self, store_vocab: bool) -> Self {
        self.vocab = store_vocab;
        self
    }

    fn verify_sanity(&self, sanity_header: Sanity) -> Result<(), Error> {
        if sanity_header != Sanity::REFERENCE {
            eprintln!(
                "Sanity header does not match the reference: \n{sanity_header:?} \nvs\n{:?}",
                Sanity::REFERENCE
            );

            return Err(Error::SanityFormatError);
        }
        Ok(())
    }

    pub(crate) fn verify(&self, fixed_params: &FixedParameters) -> Result<(), Error> {
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

    pub(crate) fn verify_arpa(&self, counts: &Counts) -> Result<(), Error> {
        if (get_max_order() as usize) < counts.order().into() {
            return Err(Error::IncompatibleMaxOrder {
                max_order: get_max_order().into(),
                model_order: counts.order().into(),
            });
        }
        Ok(())
    }

    pub(crate) fn build(self) -> Result<Model, Error> {
        let mut fd = std::fs::File::open(&self.file_name)
            .map_err(|_| Error::FileNotFound(self.file_name.to_string()))?;
        let mut config = crate::cxx::Config::default();
        config.set_load_method(self.load_method)?;
        if self.vocab {
            config.add_vocab_fetch_callback();
        };

        if let Ok(arpa_reader) = ArpaReader::new(BufReader::new(&mut fd)) {
            self.verify_arpa(arpa_reader.counts())?;
            let inner = crate::cxx::CxxModel::load_from_file_with_config(&self.file_name, &config);
            Ok(Model {
                inner,
                vocab: config.get_vocab(),
                fixed_parameters: None,
                count_header: arpa_reader.counts().clone(),
            })
        } else {
            fd.seek(SeekFrom::Start(0))?;
            let sanity_header = Sanity::from_file(&mut fd)?;
            self.verify_sanity(sanity_header)?;
            let fixed_params = headers::FixedParameters::from_file(&mut fd)?;
            self.verify(&fixed_params)?;
            let count_header = Counts::from_kenlm_binary(&mut fd, &fixed_params)?;

            let inner = crate::cxx::CxxModel::load_from_file_with_config(&self.file_name, &config);
            Ok(Model {
                inner,
                vocab: config.get_vocab(),
                fixed_parameters: Some(fixed_params),
                count_header,
            })
        }
    }
}
