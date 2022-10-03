use std::io::Read;

use zerocopy::{AsBytes, FromBytes};

use crate::cxx::bridge;
use crate::Error;

/// KenLM Model Header
///
/// `FixedParameters` is stored in bytes 88-108 in binary KenLM models. It stores general
/// information about the model. It is implemented here since we have to perform some
/// validation of the model & load-configuration before dispatching to C++ to avoid
/// violent crashes upon C++ runtime exceptions.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, FromBytes)]
pub struct FixedParameters {
    /// Order of the NGram model
    pub order: u8,
    /// Probing multiplier for the probing storage model
    pub probing_multiplier: f32,
    /// The model type, see src/cxx/lm/model_type.hh for further info
    ///
    /// PROBING = 0,
    /// REST_PROBING = 1,
    /// TRIE = 2,
    /// QUANT_TRIE = 3,
    /// ARRAY_TRIE = 4,
    /// QUANT_ARRAY_TRIE = 5
    pub model_type: u32,
    /// Does this binary store a vocabulary?
    pub has_vocabulary: u8, // this is actually a bool but FromBytes doesn't like those
    /// undocumented
    pub search_version: u32,
}

impl FixedParameters {
    pub(crate) fn from_file(fd: &mut std::fs::File) -> Result<Self, Error> {
        let mut buf = vec![0u8; bridge::get_size_of_fixed_width_params()];
        fd.read_exact(&mut buf)?;
        FixedParameters::read_from(buf.as_bytes()).ok_or(Error::ParamHeaderFormatError)
    }

    pub fn has_vocabulary(&self) -> bool {
        self.has_vocabulary != 0
    }

    #[cfg(test)]
    fn from_file_manually_parsed(fd: &mut std::fs::File) -> Result<Self, Error> {
        use std::io::{Seek, SeekFrom};

        use byteorder::{LittleEndian, ReadBytesExt};
        let order = fd.read_u8()?;
        fd.seek(SeekFrom::Current(3))?; // skip padding
        let probing_multiplier = fd.read_f32::<LittleEndian>()?;
        let model_type = fd.read_u32::<LittleEndian>()?;

        let has_vocabulary = fd.read_u8()? != 0;
        fd.seek(SeekFrom::Current(3))?; // skip padding

        let search_version = fd.read_u32::<LittleEndian>()?;

        Ok(FixedParameters {
            order,
            probing_multiplier,
            model_type,
            has_vocabulary: has_vocabulary as u8,
            search_version,
        })
    }
}

#[cfg(test)]
mod test {
    use super::FixedParameters;

    #[test]
    fn test_loads_expected() {
        let mut fd = std::fs::File::open("test_data/fixed_params.bin").unwrap();
        let from_bytes = FixedParameters::from_file(&mut fd).unwrap();
        let mut fd = std::fs::File::open("test_data/fixed_params.bin").unwrap();
        let manually = FixedParameters::from_file_manually_parsed(&mut fd).unwrap();
        let expected = FixedParameters {
            order: 3,
            probing_multiplier: 1.5,
            model_type: 2,
            has_vocabulary: 1,
            search_version: 1,
        };
        assert_eq!(from_bytes, manually);
        assert_eq!(expected, manually);
    }
}
