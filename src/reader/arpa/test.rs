#[cfg(test)]
use std::{
    fs,
    io::{BufRead, BufReader},
};

use approx::assert_abs_diff_eq;

use crate::{headers::Counts, reader::arpa::read_arpa};

use super::{read_arpa_header, ArpaReadError, NGram, ProbBackoff, ProbBackoffNgram, ProbNgram};

fn compare_expectation(thing: ProbBackoff, expectation: ProbBackoff) {
    approx::assert_abs_diff_eq!(thing.backoff, expectation.backoff);
    approx::assert_abs_diff_eq!(thing.log_prob, expectation.log_prob);
}

fn check_probbackoff_for_order(thing: &[ProbBackoffNgram], expectation: Vec<ProbBackoffNgram>) {
    thing
        .iter()
        .cloned()
        .zip(expectation.into_iter())
        .for_each(|(a, b)| {
            compare_expectation(a.prob_backoff, b.prob_backoff);
            assert_eq!(a.ngram, b.ngram);
        })
}

fn check_prob_for_order(thing: &[ProbNgram], expectation: Vec<ProbNgram>) {
    thing
        .iter()
        .cloned()
        .zip(expectation.into_iter())
        .for_each(|(a, b)| {
            assert_abs_diff_eq!(a.prob, b.prob);
            assert_eq!(a.ngram, b.ngram);
        })
}

#[test]
fn test_reads() {
    let (with_backoff, no_backoff) = read_arpa("test_data/arpa/lm_small.arpa").unwrap();
    assert_eq!(with_backoff.len(), 2);
    let uni_expect = get_unigrams();
    check_probbackoff_for_order(&with_backoff[0], uni_expect);
    let bi_expect = get_bigrams();
    check_probbackoff_for_order(&with_backoff[1], bi_expect);
    let tri_expect = get_trigrams();
    check_prob_for_order(&no_backoff, tri_expect);
}

#[test]
fn test_no_data_header() {
    let fd = fs::File::open("test_data/arpa/arpa_no_data_header.arpa").unwrap();
    let buf_read = BufReader::new(fd);
    let mut lines = buf_read.lines();
    let err = read_arpa_header(&mut lines);
    match err {
        Ok(_) => panic!("returned Ok when it should have been `Err(DataHeaderMissing)`"),
        Err(err) => assert!(matches!(err, ArpaReadError::DataHeaderMissing)),
    }
}

#[test]
fn test_no_ngram_counts() {
    let fd = fs::File::open("test_data/arpa/arpa_no_counts.arpa").unwrap();
    let buf_read = BufReader::new(fd);
    let mut lines = buf_read.lines();
    let err = read_arpa_header(&mut lines);
    match err {
        Ok(_) => panic!("returned Ok when it should have been `Err(NgramCountsMissing)`"),
        Err(err) => assert!(matches!(err, ArpaReadError::NgramCountsMissing)),
    }
}

#[test]
fn test_header() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let buf_read = BufReader::new(fd);
    let mut lines = buf_read.lines();
    let counts = read_arpa_header(&mut lines).unwrap();
    assert_eq!(
        Counts {
            counts: vec![4415, 18349]
        },
        counts
    )
}

macro_rules! prob_backoff_ngram {
    (
        $(
            $log_prob:expr, $ngram:expr, $backoff:expr
        );*
    ) => {

            vec![
                $(
                    ProbBackoffNgram {
                        prob_backoff: ProbBackoff {
                            log_prob: $log_prob,
                            backoff: $backoff,
                        },
                        ngram: NGram ($ngram.to_string()),
                    }
                ),*
            ]
        }

}

macro_rules! prob_ngram {
    (
        $(
            $log_prob:expr, $ngram:expr
        );*
    ) => {

            vec![
                $(
                    ProbNgram {
                        prob: $log_prob,
                        ngram: NGram ($ngram.to_string()),
                    }
                ),*
            ]
        }

}

#[allow(clippy::approx_constant)]
pub fn get_trigrams() -> Vec<ProbNgram> {
    prob_ngram!(-0.21873854 ,"a a </s>";
    -0.10757457,	"you remember i";
    -0.18978158, "<s> i have";
    -0.1770414,	"remember i a";
    -0.10225761,	"i have a";
    -0.2051335,	"i a a";
    -0.21873854,	"have a good";
    -0.112957425,	"a good deal";
    -0.112957425,	"good deal of";
    -0.112957425,	"deal of will";
    -0.112957425,	"of will you";
    -0.112957425,	"will you remember")
}

#[allow(clippy::approx_constant)]
pub fn get_bigrams() -> Vec<ProbBackoffNgram> {
    prob_backoff_ngram!(-0.68063426	,"a </s>",	-0.0;
    -0.250891	,"<s> i",	-0.30103;
    -0.250891	,"remember i",	-0.30103;
    -0.5346796	,"i have",	-0.30103;
    -0.4809342	,"i a",	-0.30103;
    -0.23625793	,"have a",	-0.30103;
    -0.6071514	,"a a",	-0.30103;
    -0.68063426	,"a good",	-0.30103;
    -0.26603433	,"good deal",	-0.30103;
    -0.26603433	,"deal of",	-0.30103;
    -0.26603433	,"of will",	-0.30103;
    -0.26603433	,"will you",	-0.30103;
    -0.26603433	,"you remember",	-0.30103)
}

#[allow(clippy::approx_constant)]
pub fn get_unigrams() -> Vec<ProbBackoffNgram> {
    prob_backoff_ngram!(-1.3424227,	"<unk>", -0.0;
    -0.0,           "<s>", -0.30103;
    -1.0761548,	"</s>", -0.0;
    -0.91229796,	"i", -0.30103;
    -1.0761548,	"have", -0.30103;
    -0.7936082,	"a", -0.30103;
    -1.0761548,	"good", -0.30103;
    -1.0761548,	"deal", -0.30103;
    -1.0761548,	"of", -0.30103;
    -1.0761548,	"will", -0.30103;
    -1.0761548,	"you", -0.30103;
    -1.0761548,	"remember", -0.30103)
}
