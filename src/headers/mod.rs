mod counts;
pub(crate) mod fixed_width_params;
pub(crate) mod sanity;
pub use counts::{Counts, InvalidCounts, NGramCardinality};
pub use fixed_width_params::FixedParameters;
pub(crate) use sanity::Sanity;

#[cfg(test)]
mod test {
    use std::num::NonZeroUsize;

    use crate::headers::{counts::Counts, FixedParameters, NGramCardinality};

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
        let counts = Counts::from_kenlm_binary(&mut fd, &fixed).unwrap();
        assert_eq!(
            counts,
            Counts::from_count_vec(vec![
                NGramCardinality {
                    cardinality: 24,
                    order: NonZeroUsize::try_from(1).unwrap()
                },
                NGramCardinality {
                    cardinality: 24,
                    order: NonZeroUsize::try_from(2).unwrap()
                },
                NGramCardinality {
                    cardinality: 24,
                    order: NonZeroUsize::try_from(3).unwrap()
                }
            ])
            .unwrap()
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
        let counts = Counts::from_kenlm_binary(&mut fd, &fixed).unwrap();
        assert_eq!(
            counts,
            Counts::from_count_vec(vec![
                NGramCardinality::try_from_order_and_cardinality(1, 4415).unwrap(),
                NGramCardinality::try_from_order_and_cardinality(2, 18349).unwrap(),
                NGramCardinality::try_from_order_and_cardinality(3, 25612).unwrap()
            ])
            .unwrap()
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
        let counts = Counts::from_kenlm_binary(&mut fd, &fixed).unwrap();
        assert_eq!(
            counts,
            Counts::from_count_vec(vec![
                NGramCardinality::try_from_order_and_cardinality(1, 4415).unwrap(),
                NGramCardinality::try_from_order_and_cardinality(2, 18349).unwrap()
            ])
            .unwrap()
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
