mod expectation;

#[cfg(test)]
use std::{
    fs,
    io::{BufRead, BufReader},
};

use approx::assert_abs_diff_eq;

use crate::{headers::Counts, reader::arpa::read_arpa};

use super::{read_arpa_header, ArpaReadError, ProbBackoff, ProbBackoffNgram, ProbNgram};

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
    let uni_expect = expectation::get_unigrams();
    check_probbackoff_for_order(&with_backoff[0], uni_expect);
    let bi_expect = expectation::get_bigrams();
    check_probbackoff_for_order(&with_backoff[1], bi_expect);
    let tri_expect = expectation::get_trigrams();
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
