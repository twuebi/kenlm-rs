use std::io::Read;

use zerocopy::{AsBytes, FromBytes};

use crate::cxx::bridge;
use crate::Error;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, FromBytes)]
pub struct FixedParameterHeader {
    pub order: u8,
    pub probing_multiplier: f32,
    pub model_type: u32,
    pub has_vocabulary: u8, // this is actually a bool but FromBytes doesn't like those
    pub search_version: u32,
}

impl FixedParameterHeader {
    pub(crate) fn from_file(fd: &mut std::fs::File) -> Result<Self, Error> {
        let mut buf = vec![0u8; bridge::get_size_of_fixed_width_params()];
        fd.read(&mut buf)?;
        FixedParameterHeader::read_from(buf.as_bytes()).ok_or(Error::ParamHeaderFormatError)
    }

    pub fn has_vocabulary(&self) -> bool {
        self.has_vocabulary != 0
    }

    #[allow(dead_code)]
    pub(crate) fn from_file_manually_parsed(fd: &mut std::fs::File) -> Result<Self, Error> {
        let mut buf = vec![0u8; bridge::get_size_of_fixed_width_params()];
        fd.read(&mut buf)?;
        let order = buf[0];
        // Padding: buf[1], buf[2], buf[3]
        // Probing multiplier: buf[4], buf[5], buf[6], buf[7]
        let probing_multiplier = f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        // Model type: buf[8], buf[9], buf[10], buf[11]
        let model_type = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        // has_vocabulary
        let has_vocabulary = buf[12] != 0;
        // Padding buf[13], buf[14], buf[15]
        let search_version = buf[16] as u32;
        // Padding buf[17], buf[18], buf[19]

        Ok(FixedParameterHeader {
            order,
            probing_multiplier,
            model_type,
            has_vocabulary: has_vocabulary as u8,
            search_version,
        })
    }
}
