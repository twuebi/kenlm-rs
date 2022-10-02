pub(crate) mod fixed_width_params;
pub(crate) mod sanity;
pub use fixed_width_params::FixedParameterHeader;
pub(crate) use sanity::SanityHeader;

#[cfg(test)]
mod test {
    use crate::headers::FixedParameterHeader;

    #[test]
    fn loads_both() {
        let expected_fixed = FixedParameterHeader {
            order: 3,
            probing_multiplier: 1.5,
            model_type: 2,
            has_vocabulary: 1,
            search_version: 1,
        };

        let mut fd = std::fs::File::open("test_data/sanity_and_fixed.bin").unwrap();
        let sanity = super::SanityHeader::from_file(&mut fd).unwrap();
        assert_eq!(sanity, super::SanityHeader::REFERENCE);
        let fixed = FixedParameterHeader::from_file(&mut fd).unwrap();
        assert_eq!(fixed, expected_fixed);
    }

    #[test]
    fn loads_from_full_model_file() {
        let expected_fixed = FixedParameterHeader {
            order: 3,
            probing_multiplier: 1.5,
            model_type: 2,
            has_vocabulary: 1,
            search_version: 1,
        };

        let mut fd = std::fs::File::open("test_data/carol.bin").unwrap();
        let sanity = super::SanityHeader::from_file(&mut fd).unwrap();
        assert_eq!(sanity, super::SanityHeader::REFERENCE);
        let fixed = FixedParameterHeader::from_file(&mut fd).unwrap();
        assert_eq!(fixed, expected_fixed);
    }

    #[test]
    fn loads_from_other_full_model_file() {
        let expected_fixed = FixedParameterHeader {
            order: 2,
            probing_multiplier: 1.5,
            model_type: 0,
            has_vocabulary: 1,
            search_version: 0,
        };

        let mut fd = std::fs::File::open("test_data/carol_probing_bigram.bin").unwrap();
        let sanity = super::SanityHeader::from_file(&mut fd).unwrap();
        assert_eq!(sanity, super::SanityHeader::REFERENCE);
        let fixed = FixedParameterHeader::from_file(&mut fd).unwrap();
        assert_eq!(fixed, expected_fixed);
    }
}
