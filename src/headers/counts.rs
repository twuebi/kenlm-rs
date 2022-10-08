use std::num::{NonZeroUsize, TryFromIntError};

use byteorder::{LittleEndian, ReadBytesExt};
use itertools::Itertools;

use crate::Error;

use super::FixedParameters;

/// CountHeader
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counts {
    counts: Vec<NGramCardinality>,
}

impl Counts {
    pub fn from_count_vec(mut counts: Vec<NGramCardinality>) -> Result<Self, InvalidCounts> {
        counts.sort_by(|c1, c2| c1.order.cmp(&c2.order));
        if counts.iter().map(|m| m.order).unique().count() != counts.len() {
            return Err(InvalidCounts);
        }
        if counts.is_empty() {
            return Err(InvalidCounts);
        }
        Ok(Self { counts })
    }

    pub(crate) fn from_kenlm_binary(
        fd: &mut std::fs::File,
        fixed_params: &FixedParameters,
    ) -> Result<Self, Error> {
        let counts = (0..fixed_params.order)
            .map(|order| {
                fd.read_u64::<LittleEndian>().map(|c| NGramCardinality {
                    cardinality: c as usize,
                    // int + 1
                    order: NonZeroUsize::try_from((order + 1) as usize).unwrap(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_count_vec(counts)?)
    }

    pub fn get(&self, idx: NonZeroUsize) -> Option<&NGramCardinality> {
        let usz: usize = idx.into();
        // index is order - 1
        self.counts.get(usz - 1)
    }

    pub fn order(&self) -> NonZeroUsize {
        self.highest_order_count().order
    }

    pub fn counts(&self) -> &[NGramCardinality] {
        &self.counts
    }

    pub fn highest_order_minus_one_counts(&self) -> &[NGramCardinality] {
        // Again, it is impossible to construct this struct with an empty counts vec
        &self.counts[..self.counts.len() - 1]
    }

    pub fn highest_order_count(&self) -> &NGramCardinality {
        // it is impossible to construct this struct with an empty counts vec
        self.counts.last().as_ref().unwrap()
    }
}

#[derive(thiserror::Error, Debug)]
#[error("")]
pub struct InvalidCounts;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]

pub struct NGramCardinality {
    pub order: NonZeroUsize,
    pub cardinality: usize,
}

impl NGramCardinality {
    pub fn try_from_order_and_cardinality(
        order: usize,
        cardinality: usize,
    ) -> Result<Self, TryFromIntError> {
        Ok(Self {
            order: NonZeroUsize::try_from(order)?,
            cardinality,
        })
    }
}
