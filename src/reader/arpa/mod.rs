use itertools::Itertools;
use std::{io::BufRead, num::NonZeroUsize};

use crate::headers::{Counts, InvalidCounts, NGramCardinality};

use super::{NGram, ProbBackoff, ProbBackoffNgram, ProbNgram};
#[cfg(test)]
mod test;

#[derive(thiserror::Error, Debug)]
pub enum ArpaReadError {
    #[error("The /data/ header is missing")]
    DataHeaderMissing,
    #[error("NGram Count Section could not be parsed.")]
    NgramCountsBroken,
    #[error("NGram counts are missing in the \\data\\ section")]
    NgramCountsMissing,
    #[error("A NGram section with backoff is malformed.")]
    BackOffSectionError,
    #[error("A NGram section is missing its header.")]
    NGramSectionHeaderMissing,
    #[error("A NGram section mismatch. Got: {0}; Expected: {1}")]
    NGramSectionHeaderMismatch(String, String),
    #[error("actual NGram count does not match the header description.")]
    NgramCountsMismatch,
    #[error("Decoding the count header failed")]
    CountHeaderError(#[from] InvalidCounts),
    #[error("A boundary between sections is missing. An empty line is expected")]
    SectionBoundaryMissing,
    #[error("The no-backoff section is malformed.")]
    NoBackoffSectionError,
    #[error("An IO error occured while reading the arpa file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Tried reading a section while being in the wrong state")]
    InvalidReaderState,
}

pub enum ArpaSection {
    Backoff(Vec<ProbBackoffNgram>),
    NoBackoff(Vec<ProbNgram>),
}

pub struct ArpaReader<B> {
    reader: B,
    counts: Counts,
    cur_section: NonZeroUsize,
}

impl<B> ArpaReader<B>
where
    B: BufRead,
{
    const ARPA_DATA_HEADER: &str = "\\data\\";
    const ARPA_NGRAM_KEY: &str = "ngram ";

    pub fn new(mut reader: B) -> Result<Self, ArpaReadError> {
        let counts = Self::read_count_header(&mut reader)?;
        Ok(Self {
            counts,
            reader,
            cur_section: NonZeroUsize::try_from(1).unwrap(),
        })
    }

    pub fn order(&self) -> NonZeroUsize {
        self.counts.order()
    }

    fn read_count_header(reader: &mut B) -> Result<Counts, ArpaReadError> {
        let mut reader = reader.lines();
        match reader.next().transpose()?.as_deref() {
            Some(Self::ARPA_DATA_HEADER) => {}
            _ => {
                return Err(ArpaReadError::DataHeaderMissing);
            }
        }

        let mut counts = vec![];
        while let Some(line) = reader.next().transpose()? {
            if line.trim().is_empty() {
                break;
            }

            if let Some(suffix) = line.strip_prefix(Self::ARPA_NGRAM_KEY) {
                counts.push(NGramCardinality::try_from_ngram_line_suffix(suffix)?);
            }
        }
        if counts.is_empty() {
            return Err(ArpaReadError::NgramCountsMissing);
        }
        let counts = counts.into_iter().collect();
        Ok(Counts::from_count_vec(counts)?)
    }

    fn next_backoff_section(&mut self) -> Result<Option<Vec<ProbBackoffNgram>>, ArpaReadError> {
        if self.cur_section >= self.order() {
            return Ok(None);
        }
        let count = if let Some(cnt) = self.counts.get(self.cur_section) {
            cnt
        } else {
            return Ok(None);
        };

        let mut reader = (&mut self.reader).lines();
        if let Some(next_line) = reader.next().transpose()? {
            matches_ngram_section_header(&next_line, count.order)?
        } else {
            return Err(ArpaReadError::NGramSectionHeaderMissing);
        };

        let prob_backoff_ngrams = (&mut reader)
            .take(count.cardinality)
            .map(|s| s.map_err(|_| ArpaReadError::BackOffSectionError))
            .map(|s| ProbBackoffNgram::try_from_arpa_line(&s?))
            .collect::<Result<Vec<ProbBackoffNgram>, ArpaReadError>>()?;

        if prob_backoff_ngrams.len() != count.cardinality {
            return Err(ArpaReadError::NgramCountsMismatch);
        }
        if let Some(line) = reader.next().transpose()? {
            if !line.trim().is_empty() {
                return Err(ArpaReadError::SectionBoundaryMissing);
            }
        }
        self.cur_section = self.cur_section.saturating_add(1);
        Ok(Some(prob_backoff_ngrams))
    }

    fn read_no_backoff_section(&mut self) -> Result<Vec<ProbNgram>, ArpaReadError> {
        if self.cur_section != self.order() {
            return Err(ArpaReadError::InvalidReaderState);
        }

        let mut reader = (&mut self.reader).lines();
        let counts = self.counts.highest_order_count();

        if let Some(line) = reader.next().transpose()? {
            matches_ngram_section_header(&line, counts.order)?;
        } else {
            return Err(ArpaReadError::NGramSectionHeaderMissing);
        }
        let prob_backoff_ngrams = (&mut reader)
            .take(counts.cardinality)
            .map(|s| s.map_err(|_| ArpaReadError::BackOffSectionError))
            .map(|s| ProbNgram::try_from_arpa_line(&s?))
            .collect::<Result<Vec<ProbNgram>, ArpaReadError>>()?;
        if prob_backoff_ngrams.len() != counts.cardinality {
            return Err(ArpaReadError::NgramCountsMismatch);
        }
        if let Some(Ok(line)) = reader.next() {
            if !line.trim().is_empty() {
                return Err(ArpaReadError::SectionBoundaryMissing);
            }
        }
        self.cur_section = self.cur_section.saturating_add(1);
        Ok(prob_backoff_ngrams)
    }

    pub fn counts(&self) -> &Counts {
        &self.counts
    }

    pub fn into_sections(
        &mut self,
    ) -> Result<(Vec<Vec<ProbBackoffNgram>>, Vec<ProbNgram>), ArpaReadError> {
        let mut backoffs = vec![];
        while let Some(backoff) = self.next_backoff_section()? {
            backoffs.push(backoff)
        }
        Ok((backoffs, self.read_no_backoff_section()?))
    }
}

impl ProbNgram {
    fn try_from_arpa_line(line: &str) -> Result<Self, ArpaReadError> {
        let mut pieces = line.split_ascii_whitespace();
        let log_prob = if let Some(Ok(log_prob)) = pieces.next().map(str::parse::<f32>) {
            log_prob
        } else {
            return Err(ArpaReadError::NoBackoffSectionError);
        };

        let ngram = pieces.join(" ");

        Ok(Self {
            ngram: NGram(ngram),
            prob: log_prob,
        })
    }
}

impl ProbBackoffNgram {
    fn try_from_arpa_line(line: &str) -> Result<Self, ArpaReadError> {
        let mut pieces = line.split_ascii_whitespace();
        let log_prob = if let Some(Ok(log_prob)) = pieces.next().map(str::parse::<f32>) {
            log_prob
        } else {
            return Err(ArpaReadError::BackOffSectionError);
        };
        let mut pieces = pieces.rev();
        let backoff = if let Some(Ok(backoff)) = pieces.next().map(str::parse::<f32>) {
            backoff
        } else {
            return Err(ArpaReadError::BackOffSectionError);
        };

        let ngram = pieces.rev().join(" ");

        Ok(Self {
            ngram: NGram(ngram),
            prob_backoff: ProbBackoff { log_prob, backoff },
        })
    }
}

fn matches_ngram_section_header(line: &str, order: NonZeroUsize) -> Result<(), ArpaReadError> {
    let order = order.get();
    let expected_header = format!("\\{}-grams:", order);
    if expected_header != line {
        return Err(ArpaReadError::NGramSectionHeaderMismatch(
            line.to_string(),
            expected_header,
        ));
    }
    Ok(())
}

pub fn read_arpa<B>(
    buf_read: B,
) -> Result<(Vec<Vec<ProbBackoffNgram>>, Vec<ProbNgram>), ArpaReadError>
where
    B: BufRead,
{
    let mut reader = ArpaReader::new(buf_read)?;
    reader.into_sections()
}

impl NGramCardinality {
    fn try_from_ngram_line_suffix(suffix: &str) -> Result<Self, ArpaReadError> {
        let mut suffix_pieces = suffix.split('=');
        let order = if let Some(Ok(order)) = suffix_pieces.next().map(|c| c.parse::<usize>()) {
            order
        } else {
            return Err(ArpaReadError::NgramCountsBroken);
        };
        let cardinality =
            if let Some(Ok(cardinality)) = suffix_pieces.next().map(|c| c.parse::<usize>()) {
                cardinality
            } else {
                return Err(ArpaReadError::NgramCountsBroken);
            };
        NGramCardinality::try_from_order_and_cardinality(order, cardinality)
            .map_err(|_| ArpaReadError::NgramCountsBroken)
    }
}
