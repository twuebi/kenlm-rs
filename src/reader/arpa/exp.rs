use std::{
    io::{BufRead, BufReader, Lines},
    num::NonZeroUsize,
};

use fst::raw::{Node, Output};
use itertools::Itertools;

use crate::{headers::Counts, vocab::BidiMapping};

#[derive(Debug, Clone)]
pub struct NgramWithProbBackoff {
    pub ngram: Vec<u32>,
    pub log_prob: f32,
    pub backoff: f32,
}

#[derive(Debug, Clone)]
pub struct NgramWithProb {
    pub ngram: Vec<u32>,
    pub log_prob: f32,
}

pub trait AsNgram {
    fn as_ngram(&self) -> &[u32];
}

impl AsNgram for NgramWithProb {
    fn as_ngram(&self) -> &[u32] {
        &self.ngram
    }
}

impl AsNgram for NgramWithProbBackoff {
    fn as_ngram(&self) -> &[u32] {
        &self.ngram
    }
}

impl NgramWithProbBackoff {
    fn try_from_backoff_arpa_line(
        line: &str,
        bidi: &mut BidiMapping,
    ) -> Result<Self, ArpaReadError> {
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
        let ngram = pieces
            .rev()
            .map(|pc| {
                let idx = bidi.insert_or_get(pc.to_string());
                idx
            })
            .collect::<Vec<_>>();
        Ok(NgramWithProbBackoff {
            ngram,
            log_prob,
            backoff,
        })
    }
}

impl NgramWithProb {
    fn try_from_non_backoff_arpa_line(
        line: &str,
        bidi: &mut BidiMapping,
    ) -> Result<Self, ArpaReadError> {
        let mut pieces = line.split_ascii_whitespace();
        let log_prob = if let Some(Ok(log_prob)) = pieces.next().map(str::parse::<f32>) {
            log_prob
        } else {
            return Err(ArpaReadError::BackOffSectionError);
        };
        let ngram = pieces
            .rev()
            .map(|pc| {
                let idx = bidi.insert_or_get(pc.to_string());
                idx
            })
            .collect::<Vec<_>>();
        Ok(Self { ngram, log_prob })
    }
}

use super::{matches_ngram_section_header, read_arpa_header, ArpaReadError};

fn read_backoff_section<B: BufRead>(
    fd: &mut Lines<B>,
    order: NonZeroUsize,
    count: usize,
    bidi: &mut BidiMapping,
) -> Result<Vec<NgramWithProbBackoff>, ArpaReadError> {
    if let Some(Ok(line)) = dbg!(fd.next()) {
        matches_ngram_section_header(&line, order)?;
    } else {
        return Err(ArpaReadError::NGramSectionHeaderMissing);
    }
    let prob_backoff_ngrams = fd
        .take(count)
        .map(|s| s.map_err(|_| ArpaReadError::BackOffSectionError))
        .map(|s| NgramWithProbBackoff::try_from_backoff_arpa_line(&s?, bidi))
        .collect::<Result<Vec<NgramWithProbBackoff>, ArpaReadError>>()?;
    if prob_backoff_ngrams.len() != count {
        return Err(ArpaReadError::NgramCountsMismatch);
    }
    if let Some(Ok(line)) = fd.next() {
        if !line.trim().is_empty() {
            return Err(ArpaReadError::SectionBoundaryMissing);
        }
    }
    Ok(prob_backoff_ngrams)
}

fn read_no_backoff_section<B: BufRead>(
    fd: &mut Lines<B>,
    order: NonZeroUsize,
    count: usize,
    bidi: &mut BidiMapping,
) -> Result<Vec<NgramWithProb>, ArpaReadError> {
    if let Some(Ok(line)) = dbg!(fd.next()) {
        matches_ngram_section_header(&line, order)?;
    } else {
        return Err(ArpaReadError::NGramSectionHeaderMissing);
    }
    let prob_backoff_ngrams = fd
        .take(count)
        .map(|s| s.map_err(|_| ArpaReadError::BackOffSectionError))
        .map(|s| NgramWithProb::try_from_non_backoff_arpa_line(&s?, bidi))
        .collect::<Result<Vec<NgramWithProb>, ArpaReadError>>()?;
    if prob_backoff_ngrams.len() != count {
        return Err(ArpaReadError::NgramCountsMismatch);
    }
    if let Some(Ok(line)) = fd.next() {
        if !line.trim().is_empty() {
            return Err(ArpaReadError::SectionBoundaryMissing);
        }
    }
    Ok(prob_backoff_ngrams)
}

pub fn read_arpa(
    file: &str,
) -> Result<
    (
        Vec<Vec<NgramWithProbBackoff>>,
        Vec<NgramWithProb>,
        BidiMapping,
        usize,
    ),
    ArpaReadError,
> {
    let fd = std::fs::File::open(file).unwrap();
    let buf_read = BufReader::new(fd);
    let mut rdr = buf_read.lines();
    let counts = dbg!(read_arpa_header(&mut rdr)?);
    let order = counts.counts.len();
    let mut bidi = BidiMapping::default();
    let backoff_sections = read_backoff_sections(&mut rdr, &counts, &mut bidi)?;
    let no_backoff_section = read_no_backoff_section(
        &mut rdr,
        NonZeroUsize::try_from(order).unwrap(),
        counts.counts[order - 1],
        &mut bidi,
    )?;
    // let iter = backoff_sections
    //     .into_iter()
    //     .flat_map(|item| item.into_iter().map(|i| (i.ngram, i.log_prob, i.backoff)))
    //     .sorted_by(|a, b| a.0.cmp(&b.0));
    Ok((backoff_sections, no_backoff_section, bidi, order))
}
#[derive(Clone, Debug, Copy)]
struct ProbBackoff {
    log_prob: f32,
    backoff: f32,
}

struct Model {
    unigrams: Vec<ProbBackoff>,
    lookups: Vec<Scores>,
    n_longest: usize,
    n_middle: usize,
    fst: fst::Map<Vec<u8>>,
    mapping: BidiMapping,
    order: usize,
}

#[derive(Debug, Clone, Copy)]
enum Scores {
    Longest(f32),
    Middle(ProbBackoff),
}
fn vec_into_bytes(vec: &[u32]) -> Vec<u8> {
    vec.into_iter()
        .map(|v| v.to_le_bytes())
        .flatten()
        .collect_vec()
}

fn consume_backoff_coll(vec: &mut Vec<Scores>, item: &NgramWithProbBackoff) -> Vec<u8> {
    vec_into_bytes(&item.ngram)
}

impl Model {
    fn from_arpa(file_name: &str) -> Self {
        let (backoff, longest, mapping, order) = read_arpa(file_name).unwrap();

        let order_minus_one = backoff.len();
        let mut unis = (&backoff[0]).clone();
        let n_unis = unis.len();
        let n_rest: usize = backoff[1..].iter().map(|c| c.len()).sum();
        let mut lookups = vec![];
        let things = backoff[1..]
            .into_iter()
            .flat_map(|s| s.iter())
            .for_each(|item| {
                lookups.push(Scores::Middle(ProbBackoff {
                    log_prob: item.log_prob,
                    backoff: item.backoff,
                }));
            });
        longest.iter().for_each(|c| {
            lookups.push(Scores::Longest(c.log_prob));
        });
        let n_longest: usize = longest.len();
        let search = backoff[1..]
            .into_iter()
            .flat_map(|s| s.iter().map(|c| vec_into_bytes(c.as_ngram())))
            .chain(longest.into_iter().map(|c| vec_into_bytes(c.as_ngram())))
            .enumerate()
            .sorted_by(|c, c2| c.1.cmp(&c2.1))
            .map(|(a, b)| (b, a as u64));
        let fst = fst::Map::from_iter(search).unwrap();
        unis.sort_by(|c, c2| c.ngram.cmp(&c2.ngram));

        Self {
            unigrams: unis
                .into_iter()
                .map(|s| ProbBackoff {
                    log_prob: s.log_prob,
                    backoff: s.backoff,
                })
                .collect(),
            lookups,
            n_longest,
            n_middle: n_rest,
            fst,
            mapping,
            order,
        }
    }

    fn uni_lookup(&self, unigram: &str) -> &ProbBackoff {
        let idx = self.mapping.get_index(unigram).unwrap();
        &self.unigrams[idx as usize]
    }

    fn fetch_scores(&self, output: fst::raw::Output) -> &Scores {
        let val = output.value() as usize;
        &self.lookups[val]
    }

    fn lookup_next(
        &self,
        state_addr: usize,
        output: fst::raw::Output,
        new_word: u32,
    ) -> Option<(usize, fst::raw::Output, Option<fst::raw::Output>)> {
        let state = self.fst.as_fst().node(state_addr);
        let bytes = new_word.to_le_bytes();
        let (node, output, is_final) =
            bytes
                .into_iter()
                .fold(Some((state, output, false)), |state, byte| {
                    state
                        .map(|(s, output, _is_final)| {
                            s.find_input(byte).map(|addr| {
                                let trans = s.transition(addr);
                                let state = self.fst.as_fst().node(trans.addr);
                                let out = output.cat(trans.out);
                                (state, out, state.is_final())
                            })
                        })
                        .flatten()
                })?;
        Some((
            node.addr(),
            output,
            if is_final {
                Some(output.cat(node.final_output()))
            } else {
                None
            },
        ))
    }

    fn score(&self, context: &[u32], word: u32) -> Vec<Option<Scores>> {
        let mut addr = self.fst.as_fst().root().addr();
        let mut output = Output::zero();
        let mut state_collector = vec![None; self.order];
        let uni = self.unigrams[context[0] as usize];
        state_collector[0] = Some(Scores::Middle(uni));
        for (w, widx) in context.into_iter().enumerate() {
            if let Some((new_addr, new_out, final_output)) =
                dbg!(self.lookup_next(addr, output, *widx))
            {
                output = new_out;
                addr = new_addr;
                if let Some(finished) = final_output {
                    let scores = self.fetch_scores(finished);
                    state_collector[w] = Some(*scores);
                } else {
                    state_collector[..w]
                        .iter_mut()
                        .for_each(|thing| *thing = None);
                }
            } else {
                if w != 0 {
                    addr = self.fst.as_fst().root().addr();
                    output = Output::zero();
                    state_collector = vec![None; self.order];
                }
            }
        }
        if let Some((_, _, Some(final_output))) = dbg!(self.lookup_next(addr, output, word)) {
            state_collector[self.order - 1] = Some(*self.fetch_scores(final_output));
        }
        state_collector
    }
    // template <class Search, class VocabularyT>
    // FullScoreReturn GenericModel<Search, VocabularyT>::FullScore(const State &in_state, const WordIndex new_word, State &out_state) const
    // {
    //   FullScoreReturn ret = ScoreExceptBackoff(in_state.words, in_state.words + in_state.length, new_word, out_state);
    //   for (const float *i = in_state.backoff + ret.ngram_length - 1; i < in_state.backoff + in_state.length; ++i)
    //   {
    //     ret.prob += *i;
    //   }
    //   return ret;
    // }
    // typename Search::UnigramPointer uni(search_.LookupUnigram(new_word, node, ret.independent_left, ret.extend_left));
    // out_state.backoff[0] = uni.Backoff();
    // ret.prob = uni.Prob();
    // ret.rest = uni.Rest();
}

fn read_backoff_sections<B: BufRead>(
    rdr: &mut Lines<B>,
    counts: &Counts,
    bidi: &mut BidiMapping,
) -> Result<Vec<Vec<NgramWithProbBackoff>>, ArpaReadError> {
    // exclude the last section since it's without backoffs
    let mut sections = vec![];
    for (order, count) in counts.counts[..counts.counts.len() - 1].iter().enumerate() {
        // this should only fail for the off-chance that order + 1 wraps which is.. unlikely
        let order = NonZeroUsize::try_from(order + 1).unwrap();
        let section = read_backoff_section(rdr, order, *count, bidi)?;
        sections.push(section)
    }
    Ok(sections)
}

#[cfg(test)]
mod test {
    use super::Model;

    #[test]
    fn test() {
        let model = Model::from_arpa("test_data/arpa/lm_small.arpa");
        let context = vec![
            model.mapping.get_index("i").unwrap(),
            model.mapping.get_index("have").unwrap(),
        ];
        let new_word = model.mapping.get_index("a").unwrap();

        dbg!(model.score(&context, new_word));
        panic!();
    }
}