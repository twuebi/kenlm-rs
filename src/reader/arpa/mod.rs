use fst::raw::Output;
use fst::Map;
use itertools::Itertools;
use std::collections::HashMap;
use std::io::Lines;
use std::{io::BufRead, num::NonZeroUsize};

use crate::headers::{Counts, InvalidCounts, NGramCardinality};
use crate::mapping::{BidirectionalMapping, Mappings};
use crate::reader::NGramRep;

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
    #[error("An IO error occurred while reading the arpa file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Tried reading a section while being in the wrong state")]
    InvalidReaderState,
}

#[derive(Debug, Clone)]
pub enum Scores {
    Longest(f32),
    Middle(ProbBackoff),
}

impl Scores {
    fn backoff(&self) -> f32 {
        match self {
            Scores::Longest(_) => 0f32,
            Scores::Middle(mid) => mid.backoff,
        }
    }
    fn score(&self) -> f32 {
        match self {
            Scores::Longest(score) => *score,
            Scores::Middle(mid) => mid.log_prob,
        }
    }
}

pub trait NGramIndexer<T> {
    fn find(&self, middle: T) -> Vec<Scores>;
    fn find_with_state(&self, state: State, new_word: u32) -> State;
}

#[derive(Debug)]
pub struct State {
    scores: Vec<Scores>,
    words: Vec<u32>,
    score: f32,
    score_history: Vec<f32>,
    ngram_length: usize,
    word_history: Vec<u32>,
    backoffs: Vec<f32>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            score: 0f32,
            scores: vec![],
            words: vec![],
            ngram_length: 0,
            score_history: vec![],
            word_history: vec![],
            backoffs: vec![],
        }
    }
}

impl<T> NGramIndexer<T> for FstIndexer
where
    T: NGramRep + std::fmt::Debug,
{
    fn find(&self, middle: T) -> Vec<Scores> {
        let fst = self.fst.as_fst();
        let mut cur_node = fst.root().addr();
        let mut output = Output::zero();
        let mut scores = vec![];
        'outer: for w in (middle.to_word_chunks()) {
            for b in w.iter() {
                let next_node = fst.node(cur_node);

                match next_node.find_input(*b).map(|t| next_node.transition(t)) {
                    None => {
                        break 'outer;
                    }
                    Some(next) => {
                        cur_node = next.addr;
                        let node = fst.node(cur_node);
                        output = output.cat(next.out);
                        if node.is_final() {
                            scores.push(
                                self.scores[output.cat(node.final_output()).value() as usize]
                                    .clone(),
                            )
                        }
                    }
                }
            }
        }
        scores
    }

    fn find_with_state(&self, state: State, new_word: u32) -> State {
        let mut out_state = State::default();

        let (_, new) = state
            .words
            .split_at(((state.words.len()) - state.ngram_length).max(0));
        let (_, new_scores) =
            dbg!(&state.scores).split_at(((state.words.len()) - state.ngram_length).max(0));
        let mut new = new.to_vec();
        let new_scores = new_scores.to_vec();
        new.push(new_word);
        out_state.word_history = state.word_history;
        out_state.word_history.push(new_word);
        // p(troll | a) = p(troll) * b(a)
        let r = dbg!(self.find(new.clone()));
        let sc = r.iter().rev().cloned().next().unwrap();
        out_state.backoffs = r
            .iter()
            .filter(|s| matches!(s, Scores::Middle(_)))
            .map(|s| s.backoff())
            .collect();
        out_state.ngram_length = dbg!(out_state.backoffs.len());
        let score = if dbg!(r.len()) != (dbg!(self.order)) {
            tracing::debug!("short: {} {}", r.len(), self.order);
            if sc.backoff() == 0.00 {
                tracing::debug!("no extension");
                out_state.words = vec![];
            }
            sc.score()
                + dbg!(&state.backoffs)
                    .iter()
                    .skip(r.len().saturating_sub(1))
                    .sum::<f32>()
        } else {
            sc.score()
        };
        out_state.score = score + state.score;
        out_state.scores = new_scores;
        out_state.scores.push(sc);
        out_state.words = new;

        (out_state)
    }
}

pub struct FstIndexer {
    pub fst: Map<Vec<u8>>,
    pub scores: Vec<Scores>,
    pub order: usize,
}

impl FstIndexer {
    fn new<T: NGramRep>(sections: ArpaFileSections<T>) -> Self {
        let ArpaFileSections {
            counts,
            backoffs,
            no_backoff,
        } = sections;
        let mut scores =
            Vec::with_capacity(counts.counts().iter().map(|cnt| cnt.cardinality).sum());

        let iter = backoffs
            .into_iter()
            .flat_map(|nob| nob.into_iter())
            .map(|it| (Scores::Middle(it.prob_backoff), it.ngram.to_bytes()))
            .chain(
                no_backoff
                    .into_iter()
                    .map(|nob| (Scores::Longest(nob.prob), nob.ngram.to_bytes())),
            )
            .enumerate()
            .map(|(idx, (score, key))| {
                // eprintln!("{key:?}, {idx} {score:?} {}", scores.len() + 1);
                scores.push(score);
                (key, idx as u64)
            })
            .sorted_by(|a, b| a.0.cmp(&b.0));
        let fst = fst::map::Map::from_iter(iter).unwrap();
        Self {
            fst,
            scores,
            order: counts.order().get(),
        }
    }
}

pub struct HashMapIndexer {
    pub map: HashMap<Vec<u32>, Scores>,
}
//
// impl NGramIndexer<&[u32]> for HashMapIndexer {
//     fn find(&self, query: &[u32]) -> Vec<Scores> {
//         if query.is_empty() {
//             return vec![self.map.get(&[0]).unwrap().clone()]
//         }
//         match self.map.get(query) {
//             None => { self.find(&query[1..]) },
//             Some(scr) => vec![scr.clone()],
//         }
//     }
//
//     fn find_with_state(&self, mut state: State, new_word: u32) -> (State) {
//         state.words.push(new_word);
//         state.ngram_length = (state.ngram_length + 1).min(2);
//
//         let scores = self.find(&state.words);
//
//     }
// }

pub struct ArpaModel<T>
where
    T: NGramRep,
{
    pub vocab: Mappings,
    pub sections: ArpaFileSections<T>,
}

pub struct ArpaFileSections<T>
where
    T: NGramRep,
{
    pub counts: Counts,
    pub backoffs: Vec<Vec<ProbBackoffNgram<T>>>,
    pub no_backoff: Vec<ProbNgram<T>>,
}

pub trait NGramProcessor {
    type Output;

    fn process_ngram<'a>(&mut self, pieces: impl Iterator<Item = &'a str>) -> Self::Output;
    fn into_mapping(self) -> Mappings;
}

pub struct StringProcessor;

impl NGramProcessor for StringProcessor {
    type Output = String;

    fn process_ngram<'a>(&mut self, mut pieces: impl Iterator<Item = &'a str>) -> Self::Output {
        pieces.join(" ")
    }

    fn into_mapping(self) -> Mappings {
        Mappings::NoOp
    }
}

pub struct IntVocabProcessor(Mappings);

impl NGramProcessor for IntVocabProcessor {
    type Output = Vec<u32>;

    fn process_ngram<'a>(&mut self, pieces: impl Iterator<Item = &'a str>) -> Self::Output {
        pieces
            .map(|pc| self.0.insert_or_get_index(pc.to_string()))
            .collect()
    }

    fn into_mapping(self) -> Mappings {
        self.0
    }
}

pub struct NoOpProc;

impl NGramProcessor for NoOpProc {
    type Output = ();

    fn process_ngram<'a>(&mut self, _: impl Iterator<Item = &'a str>) -> Self::Output {}

    fn into_mapping(self) -> Mappings {
        Mappings::NoOp
    }
}

/// Arpa reader
///
/// This struct consumes a [BufRead] and tries to parse its contents into a
/// structured representation of the arpa format.
///
/// An arpa file contains multiple sections, each section ends with an empty
/// line and has a heading which has `\` as the first.
///
/// The header of the first section is `\data\` and is expected on the first
/// line. The `\data\` heading is followed by n lines of the format
/// `ngram <order>=count` where `<order>` is within `1..=n` and n is the order
/// of the ngram model described by the arpa file.
///
/// The data section is followed by `n` ngram sections. Each n-gram section has
/// a heading of the format `\<order>-grams:`. The n-gram sections are expected
/// to be sorted in ascending order. Each n-gram section has exactly as many
/// rows as described in the count header line describing the current section.
///
/// There are `n-1` n-gram sections with backoff values, that is they have three
/// columns, `log_prob ngram backoff`. They are split on whitespace, the first
/// and last elements are parsed to floats while the middle elements are treated
/// as a whitespace separated ngram.
///
/// The last section of the file is an n-gram section where n is equal to the
/// order of the model. This section does not have backoff values, it is expected
/// to have two columns, `log_prob` and `ngram`. It is again split on whitespace,
/// the first element is parsed to float, the rest is treated as a white-space
/// separated n-gram.
pub struct ArpaReader<B, T> {
    reader: Lines<B>,
    counts: Counts,
    cur_section: NonZeroUsize,
    ngram_processor: T,
}

impl<B, T> ArpaReader<B, T>
where
    B: BufRead,
    T: NGramProcessor,
    <T as NGramProcessor>::Output: NGramRep,
{
    const ARPA_DATA_HEADER: &'_ str = "\\data\\";
    const ARPA_NGRAM_KEY: &'_ str = "ngram ";

    /// Constructs the ArpaReader, parses the header
    ///
    /// Constructs the ArpaReader and validates it by parsing the count header
    /// describing the file.
    pub fn new(mut reader: B, ngram_processor: T) -> Result<Self, ArpaReadError> {
        let counts = Self::read_count_header(&mut reader)?;

        Ok(Self {
            counts,
            reader: reader.lines(),
            cur_section: NonZeroUsize::try_from(1).unwrap(),
            ngram_processor,
        })
    }

    /// Returns the order of the model
    ///
    /// Returns the order of the model described by the arpa file.
    pub fn order(&self) -> NonZeroUsize {
        self.counts.order()
    }

    /// Returns the order of the model
    ///
    /// Returns the order of the model described by the arpa file.
    pub fn counts(&self) -> &Counts {
        &self.counts
    }

    /// Parse the n-gram sections
    ///
    /// Consumes the remainder of the reader and parses it according to the count-header of the file
    /// returns a tuple where the first element are the backoff sections in ascending ngram order,
    /// the second element is the highest order section which has no backoff values.
    pub fn into_arpa_model(mut self) -> Result<ArpaModel<T::Output>, ArpaReadError> {
        let mut backoffs = vec![];
        while let Some(backoff) = self.next_backoff_section()? {
            backoffs.push(backoff)
        }
        let no_backoff = self.read_no_backoff_section()?;
        let Self { counts, .. } = self;
        Ok(ArpaModel {
            vocab: self.ngram_processor.into_mapping(),
            sections: ArpaFileSections {
                counts,
                backoffs,
                no_backoff,
            },
        })
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

    fn next_backoff_section(
        &mut self,
    ) -> Result<Option<Vec<ProbBackoffNgram<T::Output>>>, ArpaReadError> {
        if self.cur_section >= self.order() {
            return Ok(None);
        }
        let count = if let Some(cnt) = self.counts.get(self.cur_section) {
            *cnt
        } else {
            return Ok(None);
        };

        if let Some(next_line) = self.reader.next().transpose()? {
            matches_ngram_section_header(&next_line, count.order)?
        } else {
            return Err(ArpaReadError::NGramSectionHeaderMissing);
        };

        let cardinality = count.cardinality;
        let mut prob_backoff_ngrams = Vec::with_capacity(cardinality);
        for _ in 0..cardinality {
            if let Some(line) = self.reader.next() {
                prob_backoff_ngrams.push(self.try_back_off_from_arpa_line(line?)?);
            }
        }

        if prob_backoff_ngrams.len() != count.cardinality {
            return Err(ArpaReadError::NgramCountsMismatch);
        }
        if let Some(line) = self.reader.next().transpose()? {
            if !line.trim().is_empty() {
                return Err(ArpaReadError::SectionBoundaryMissing);
            }
        }
        self.cur_section = self.cur_section.saturating_add(1);
        Ok(Some(prob_backoff_ngrams))
    }

    fn read_no_backoff_section(&mut self) -> Result<Vec<ProbNgram<T::Output>>, ArpaReadError> {
        if self.cur_section != self.order() {
            return Err(ArpaReadError::InvalidReaderState);
        }

        let counts = self.counts.highest_order_count();

        if let Some(line) = self.reader.next().transpose()? {
            matches_ngram_section_header(&line, counts.order)?;
        } else {
            return Err(ArpaReadError::NGramSectionHeaderMissing);
        }
        let cardinality = counts.cardinality;
        let mut prob_no_backoff_ngrams = Vec::with_capacity(cardinality);
        for _ in 0..counts.cardinality {
            if let Some(line) = self.reader.next() {
                prob_no_backoff_ngrams.push(self.try_no_backoff_from_arpa_line(line?)?);
            }
        }

        if prob_no_backoff_ngrams.len() != cardinality {
            return Err(ArpaReadError::NgramCountsMismatch);
        }
        if let Some(Ok(line)) = self.reader.next() {
            if !line.trim().is_empty() {
                return Err(ArpaReadError::SectionBoundaryMissing);
            }
        }
        self.cur_section = self.cur_section.saturating_add(1);
        Ok(prob_no_backoff_ngrams)
    }

    fn try_back_off_from_arpa_line(
        &mut self,
        line: String,
    ) -> Result<ProbBackoffNgram<T::Output>, ArpaReadError> {
        let mut pieces = line.split_ascii_whitespace();
        let log_prob = next_float(&mut pieces, ArpaReadError::BackOffSectionError)?;
        let mut pieces = pieces.rev();

        let backoff = next_float(&mut pieces, ArpaReadError::BackOffSectionError)?;

        let ngram = self.ngram_processor.process_ngram(pieces.rev());

        Ok(ProbBackoffNgram {
            ngram: NGram(ngram),
            prob_backoff: ProbBackoff { log_prob, backoff },
        })
    }

    fn try_no_backoff_from_arpa_line(
        &mut self,
        line: String,
    ) -> Result<ProbNgram<T::Output>, ArpaReadError> {
        let mut pieces = line.split_ascii_whitespace();
        let log_prob = next_float(&mut pieces, ArpaReadError::NoBackoffSectionError)?;

        let ngram = self.ngram_processor.process_ngram(pieces);

        Ok(ProbNgram {
            ngram: NGram(ngram),
            prob: log_prob,
        })
    }
}

fn next_float<'a>(
    mut pieces: impl Iterator<Item = &'a str>,
    error: ArpaReadError,
) -> Result<f32, ArpaReadError> {
    pieces
        .next()
        .map(str::parse::<f32>)
        .transpose()
        .ok()
        .flatten()
        .ok_or(error)
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

pub fn read_arpa<B, T>(
    buf_read: B,
    ngram_processor: T,
) -> Result<ArpaModel<T::Output>, ArpaReadError>
where
    B: BufRead,
    T: NGramProcessor,
    <T as NGramProcessor>::Output: NGramRep,
{
    ArpaReader::new(buf_read, ngram_processor)?.into_arpa_model()
}

impl NGramCardinality {
    fn try_from_ngram_line_suffix(suffix: &str) -> Result<Self, ArpaReadError> {
        let mut suffix_pieces = suffix.split('=');
        let mut parse_next_usize = || {
            if let Some(Ok(cardinality)) = suffix_pieces.next().map(|c| c.parse::<usize>()) {
                Ok(cardinality)
            } else {
                Err(ArpaReadError::NgramCountsBroken)
            }
        };
        let order = parse_next_usize()?;
        let cardinality = parse_next_usize()?;
        NGramCardinality::try_from_order_and_cardinality(order, cardinality)
            .map_err(|_| ArpaReadError::NgramCountsBroken)
    }
}
