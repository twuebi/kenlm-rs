use std::io::Read;

use zerocopy::FromBytes;

use crate::{cxx::bridge::size_of_sanity_header, Error};

use super::align8;

/// KenLM Model Sanity check
///
/// `Sanity` is stored in bytes 0-88 in binary KenLM models. As its name says, it's used
/// for sanity-checks. We implement it in rust-land to perform the validation here so that
/// we can avoid violent crashes upon C++ runtime exceptions.
#[repr(C)]
#[derive(Debug, PartialEq, FromBytes)]
pub(crate) struct Sanity {
    magic: [u8; MAGIC_BYTES.len()],
    padding: [u8; PADDING],
    float_zero: f32,
    float_one: f32,
    float_minus_half: f32,
    word_idx_one: u32,
    word_idx_max: u32,
    usize_sanity: u64,
}

const MAGIC_BYTES: [u8; 52] = *b"mmap lm http://kheafield.com/code format version 5\n\0";
const PADDING: usize = align8(MAGIC_BYTES.len()) - MAGIC_BYTES.len();

impl Sanity {
    // see src/cxx/lm/binary_format.hh & src/cxx/lm/binary_format.cc
    pub const REFERENCE: Sanity = Self {
        magic: MAGIC_BYTES,
        padding: [0; PADDING],
        float_zero: 0f32,
        float_one: 1f32,
        float_minus_half: -0.5f32,
        word_idx_one: 1u32,
        word_idx_max: u32::MAX,
        usize_sanity: 1,
    };

    pub(crate) fn from_file(fd: &mut std::fs::File) -> Result<Sanity, Error> {
        let mut header_bytes = vec![0; size_of_sanity_header() as usize];
        fd.read_exact(&mut header_bytes)?;
        Sanity::read_from(header_bytes.as_slice()).ok_or(Error::SanityFormatError)
    }
}

#[cfg(test)]
mod test {
    use super::Sanity;
    #[test]
    fn test_reference_expected() {
        let expected = Sanity {
            magic: *b"mmap lm http://kheafield.com/code format version 5\n\0",
            padding: *b"\0\0\0\0",
            float_zero: 0f32,
            float_one: 1f32,
            float_minus_half: -0.5f32,
            word_idx_one: 1,
            word_idx_max: u32::MAX,
            usize_sanity: 1,
        };
        assert_eq!(Sanity::REFERENCE, expected);
    }

    #[test]
    fn test_loads_expected() {
        let mut fd = std::fs::File::open("test_data/sanity.bin").unwrap();
        let from_bytes = Sanity::from_file(&mut fd).unwrap();
        let expected = Sanity::REFERENCE;
        assert_eq!(from_bytes, expected);
    }
}
