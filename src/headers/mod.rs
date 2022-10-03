mod counts;
pub(crate) mod fixed_width_params;
pub(crate) mod sanity;
pub use counts::Counts;
pub use fixed_width_params::FixedParameters;
pub(crate) use sanity::Sanity;

#[cfg(test)]
mod test {
    use crate::headers::{counts::Counts, FixedParameters};

    use super::total_header_size;

    #[test]
    fn loads_all() {
        let expected_fixed = FixedParameters {
            order: 3,
            probing_multiplier: 1.5,
            model_type: 2,
            has_vocabulary: 1,
            search_version: 1,
        };

        let mut fd = std::fs::File::open("test_data/sanity_fixed_and_counts.bin").unwrap();
        let sanity = super::Sanity::from_file(&mut fd).unwrap();
        assert_eq!(sanity, super::Sanity::REFERENCE);
        let fixed = FixedParameters::from_file(&mut fd).unwrap();
        assert_eq!(fixed, expected_fixed);
        let counts = Counts::from_file(&mut fd, &fixed).unwrap();
        assert_eq!(
            counts,
            Counts {
                //.. why??
                counts: vec![24, 24, 24]
            }
        );
    }

    #[test]
    fn loads_from_full_model_file() {
        let expected_fixed = FixedParameters {
            order: 3,
            probing_multiplier: 1.5,
            model_type: 2,
            has_vocabulary: 1,
            search_version: 1,
        };

        let mut fd = std::fs::File::open("test_data/carol.bin").unwrap();
        let sanity = super::Sanity::from_file(&mut fd).unwrap();
        assert_eq!(sanity, super::Sanity::REFERENCE);
        let fixed = FixedParameters::from_file(&mut fd).unwrap();
        assert_eq!(fixed, expected_fixed);
        let counts = Counts::from_file(&mut fd, &fixed).unwrap();
        assert_eq!(
            counts,
            Counts {
                counts: vec![4415, 18349, 25612]
            }
        );
    }

    #[test]
    fn loads_from_other_full_model_file() {
        let expected_fixed = FixedParameters {
            order: 2,
            probing_multiplier: 1.5,
            model_type: 0,
            has_vocabulary: 1,
            search_version: 0,
        };

        let mut fd = std::fs::File::open("test_data/carol_probing_bigram.bin").unwrap();
        let sanity = super::Sanity::from_file(&mut fd).unwrap();
        assert_eq!(sanity, super::Sanity::REFERENCE);
        let fixed = FixedParameters::from_file(&mut fd).unwrap();
        assert_eq!(fixed, expected_fixed);
        let counts = Counts::from_file(&mut fd, &fixed).unwrap();
        assert_eq!(
            counts,
            Counts {
                counts: vec![4415, 18349]
            }
        );
    }

    #[test]
    fn test_total_header_size() {
        assert_eq!(total_header_size(6), 160);
        assert_eq!(total_header_size(2), 128);
    }
}

#[cfg(test)]
fn total_header_size(order: usize) -> usize {
    align8(
        std::mem::size_of::<FixedParameters>()
            + std::mem::size_of::<Sanity>()
            + order * std::mem::size_of::<u64>(),
    )
}

pub(crate) const fn align8(size: usize) -> usize {
    let size = size as isize;
    (((((size) - 1) / 8) + 1) * 8) as usize
}
