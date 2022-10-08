use itertools::Itertools;
use std::{
    io::{BufRead, BufReader, Lines},
    num::NonZeroUsize,
};

use crate::headers::Counts;

use super::{NGram, ProbBackoff, ProbBackoffNgram, ProbNgram};
#[cfg(test)]
mod test;

#[derive(thiserror::Error, Debug)]
pub enum ArpaReadError {
    #[error("The /data/ header is missing")]
    DataHeaderMissing,
    #[error("Order key could not be parsed")]
    NgramOrder,
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
    #[error("A boundary between sections is missing. An empty line is expected")]
    SectionBoundaryMissing,
    #[error("The no-backoff section is malformed.")]
    NoBackoffSectionError,
}

const ARPA_DATA_HEADER: &str = "\\data\\";
const ARPA_NGRAM_KEY: &str = "ngram ";
const ARPA_NGRAM_SECION_HEADERS: [&str; 7] = [
    "\\1-grams:",
    "\\2-grams:",
    "\\3-grams:",
    "\\4-grams:",
    "\\5-grams:",
    "\\6-grams:",
    "\\7-grams:",
];

pub struct ArpaReader<'a, B>(&'a mut Lines<B>);

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
    if (order.get() <= 7 && ARPA_NGRAM_SECION_HEADERS[order.get() - 1] != line)
        || (order.get() > 7 && format!("\\{}-grams", order) != line)
    {
        return Err(ArpaReadError::NGramSectionHeaderMismatch(
            line.to_string(),
            if order.get() < 7 {
                ARPA_NGRAM_SECION_HEADERS[order.get() - 1].to_string()
            } else {
                format!("\\{}-grams", order)
            },
        ));
    }
    Ok(())
}

pub fn read_arpa_header<B: BufRead>(reader: &mut Lines<B>) -> Result<Counts, ArpaReadError> {
    if let Some(Ok(line)) = dbg!(reader.next()) {
        if line != ARPA_DATA_HEADER {
            return Err(ArpaReadError::DataHeaderMissing);
        }
    }
    let mut counts = vec![];

    while let Some(Ok(line)) = reader.next() {
        if line.trim().is_empty() {
            break;
        }

        if let Some(suffix) = line.strip_prefix(ARPA_NGRAM_KEY) {
            let mut suffix_pieces = suffix.split('=');
            let order = if let Some(Ok(order)) = suffix_pieces.next().map(|c| c.parse::<usize>()) {
                order
            } else {
                return Err(ArpaReadError::NgramOrder);
            };
            let cardinality =
                if let Some(Ok(cardinality)) = suffix_pieces.next().map(|c| c.parse::<usize>()) {
                    cardinality
                } else {
                    return Err(ArpaReadError::NgramOrder);
                };
            counts.push(NGramCardinality { order, cardinality });
        }
    }
    if counts.is_empty() {
        return Err(ArpaReadError::NgramCountsMissing);
    }

    let counts = counts
        .into_iter()
        .sorted_by(|cnt, cnt2| cnt.order.cmp(&cnt2.order))
        .map(
            |NGramCardinality {
                 order: _,
                 cardinality,
             }| cardinality,
        )
        .collect::<_>();

    Ok(Counts { counts })
}

fn read_backoff_section<B: BufRead>(
    reader: &mut Lines<B>,
    order: NonZeroUsize,
    count: usize,
) -> Result<Vec<ProbBackoffNgram>, ArpaReadError> {
    if let Some(Ok(line)) = dbg!(reader.next()) {
        matches_ngram_section_header(&line, order)?;
    } else {
        return Err(ArpaReadError::NGramSectionHeaderMissing);
    }
    let prob_backoff_ngrams = reader
        .take(count)
        .map(|s| s.map_err(|_| ArpaReadError::BackOffSectionError))
        .map(|s| ProbBackoffNgram::try_from_arpa_line(&s?))
        .collect::<Result<Vec<ProbBackoffNgram>, ArpaReadError>>()?;
    if prob_backoff_ngrams.len() != count {
        return Err(ArpaReadError::NgramCountsMismatch);
    }
    if let Some(Ok(line)) = reader.next() {
        if !line.trim().is_empty() {
            return Err(ArpaReadError::SectionBoundaryMissing);
        }
    }
    Ok(prob_backoff_ngrams)
}

fn read_no_backoff_section<B: BufRead>(
    reader: &mut Lines<B>,
    order: NonZeroUsize,
    count: usize,
) -> Result<Vec<ProbNgram>, ArpaReadError> {
    if let Some(Ok(line)) = dbg!(reader.next()) {
        matches_ngram_section_header(&line, order)?;
    } else {
        return Err(ArpaReadError::NGramSectionHeaderMissing);
    }
    let prob_backoff_ngrams = reader
        .take(count)
        .map(|s| s.map_err(|_| ArpaReadError::BackOffSectionError))
        .map(|s| ProbNgram::try_from_arpa_line(&s?))
        .collect::<Result<Vec<ProbNgram>, ArpaReadError>>()?;
    if prob_backoff_ngrams.len() != count {
        return Err(ArpaReadError::NgramCountsMismatch);
    }
    if let Some(Ok(line)) = reader.next() {
        if !line.trim().is_empty() {
            return Err(ArpaReadError::SectionBoundaryMissing);
        }
    }
    Ok(prob_backoff_ngrams)
}

pub fn read_arpa(
    file: &str,
) -> Result<(Vec<Vec<ProbBackoffNgram>>, Vec<ProbNgram>), ArpaReadError> {
    let reader = std::fs::File::open(file).unwrap();
    let buf_read = BufReader::new(reader);
    let mut rdr = buf_read.lines();
    let counts = dbg!(read_arpa_header(&mut rdr)?);
    let order = counts.counts.len();
    let backoff_sections = read_backoff_sections(&mut rdr, &counts)?;
    let no_backoff_section = read_no_backoff_section(
        &mut rdr,
        NonZeroUsize::try_from(order).unwrap(),
        counts.counts[order - 1],
    )?;

    Ok((backoff_sections, no_backoff_section))
}

fn read_backoff_sections<B: BufRead>(
    rdr: &mut Lines<B>,
    counts: &Counts,
) -> Result<Vec<Vec<ProbBackoffNgram>>, ArpaReadError> {
    // exclude the last section since it's without backoffs
    let mut sections = vec![];
    for (order, count) in counts.counts[..counts.counts.len() - 1].iter().enumerate() {
        // this should only fail for the off-chance that order + 1 wraps which is.. unlikely
        let order = NonZeroUsize::try_from(order + 1).unwrap();
        let section = read_backoff_section(rdr, order, *count)?;
        sections.push(section)
    }
    Ok(sections)
}

pub struct NGramCardinality {
    order: usize,
    cardinality: usize,
}
