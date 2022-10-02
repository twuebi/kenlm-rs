mod counts;
pub(crate) mod fixed_width_params;
pub(crate) mod sanity;
pub use counts::CountHeader;
pub use fixed_width_params::FixedParameterHeader;
pub(crate) use sanity::SanityHeader;

#[cfg(test)]
mod test {
    use crate::headers::{counts::CountHeader, FixedParameterHeader};

    #[test]
    fn loads_all() {
        let expected_fixed = FixedParameterHeader {
            order: 3,
            probing_multiplier: 1.5,
            model_type: 2,
            has_vocabulary: 1,
            search_version: 1,
        };

        let mut fd = std::fs::File::open("test_data/sanity_fixed_and_counts.bin").unwrap();
        let sanity = super::SanityHeader::from_file(&mut fd).unwrap();
        assert_eq!(sanity, super::SanityHeader::REFERENCE);
        let fixed = FixedParameterHeader::from_file(&mut fd).unwrap();
        assert_eq!(fixed, expected_fixed);
        let counts = CountHeader::from_file(&mut fd, &fixed).unwrap();
        assert_eq!(
            counts,
            CountHeader {
                //.. why??
                counts: vec![24, 24, 24]
            }
        );
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
        let counts = CountHeader::from_file(&mut fd, &fixed).unwrap();
        assert_eq!(
            counts,
            CountHeader {
                counts: vec![4415, 18349, 25612]
            }
        );
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
        let counts = CountHeader::from_file(&mut fd, &fixed).unwrap();
        assert_eq!(
            counts,
            CountHeader {
                counts: vec![4415, 18349]
            }
        );
    }
}

#[allow(dead_code)]
fn total_header_size(order: usize) -> usize {
    align8(
        dbg!(std::mem::size_of::<FixedParameterHeader>())
            + dbg!(std::mem::size_of::<SanityHeader>())
            + order * std::mem::size_of::<u64>(),
    )
}

pub(crate) const fn align8(size: usize) -> usize {
    let size = size as isize;
    (((((size) - 1) / 8) + 1) * 8) as usize
}
