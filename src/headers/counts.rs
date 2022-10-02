use byteorder::{LittleEndian, ReadBytesExt};

use crate::Error;

use super::FixedParameterHeader;

/// KenLM Model Count Header
///
/// This struct is stored in binary KenLM models following byte 108. It stores the number
/// of ngrams per ngram length.
#[derive(Clone, Debug, PartialEq)]
pub struct CountHeader {
    /// Order of the NGram model
    pub counts: Vec<usize>,
}

impl CountHeader {
    pub(crate) fn from_file(
        fd: &mut std::fs::File,
        fixed_params: &FixedParameterHeader,
    ) -> Result<Self, Error> {
        Ok(Self {
            counts: (0..fixed_params.order)
                .map(|_| fd.read_u64::<LittleEndian>().map(|c| c as usize))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}
