use std::fmt::Debug;
use std::{fs, io::BufReader};

use approx::assert_abs_diff_eq;

use crate::mapping::Mappings::HashVecMap;
use crate::mapping::{BidirectionalMapping, HashMapVecMap, Mappings};
use crate::reader::arpa::{
    ArpaFileSections, ArpaModel, FstIndexer, IntVocabProcessor, NGramIndexer, State,
    StringProcessor,
};
use crate::reader::{NGramRep, ToByteVec};
use crate::{
    headers::{Counts, NGramCardinality},
    reader::arpa::read_arpa,
};

use super::{ArpaReadError, ArpaReader, NGram, ProbBackoff, ProbBackoffNgram, ProbNgram};

fn compare_expectation(thing: ProbBackoff, expectation: ProbBackoff) {
    assert_abs_diff_eq!(thing.backoff, expectation.backoff);
    assert_abs_diff_eq!(thing.log_prob, expectation.log_prob);
}

fn check_probbackoff_for_order<T>(
    thing: &[ProbBackoffNgram<T>],
    expectation: Vec<ProbBackoffNgram<T>>,
) where
    T: NGramRep + std::fmt::Debug + Clone + PartialEq + Eq,
{
    thing
        .iter()
        .cloned()
        .zip(expectation.into_iter())
        .for_each(|(a, b)| {
            compare_expectation(a.prob_backoff, b.prob_backoff);
            assert_eq!(a.ngram, b.ngram);
        })
}

fn check_prob_for_order<T>(thing: &[ProbNgram<T>], expectation: Vec<ProbNgram<T>>)
where
    T: NGramRep + std::fmt::Debug + Clone + PartialEq + Eq,
{
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
    let fd = fs::File::open("test_data/arpa/lm_small.arpa").unwrap();
    let br = BufReader::new(fd);

    let ArpaModel {
        vocab,
        sections:
            ArpaFileSections {
                counts: _,
                backoffs,
                no_backoff,
            },
    } = read_arpa(br, StringProcessor).unwrap();
    assert_eq!(backoffs.len(), 2);

    let uni_expect = get_unigrams();
    check_probbackoff_for_order(&backoffs[0], uni_expect);
    let bi_expect = get_bigrams();
    check_probbackoff_for_order(&backoffs[1], bi_expect);
    let tri_expect = get_trigrams();
    check_prob_for_order(&no_backoff, tri_expect);
}

#[test]
fn test_unigram() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    let fst_indexer = FstIndexer::new(sections);

    let klm = crate::model::Model::new("test_data/arpa/lm.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let mut running = 0f32;
    for word in &vec![String::from("have")] {
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;
        assert_abs_diff_eq!(dbg!(state.score), running);
        std::mem::swap(&mut state_a, &mut state_b);
    }
    assert_abs_diff_eq!(dbg!(state.score), running);
}

#[test]
fn test_unigram_unk() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    let fst_indexer = FstIndexer::new(sections);

    let klm = crate::model::Model::new("test_data/arpa/lm.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let mut running = 0f32;
    for word in &vec![String::from("greaaa")] {
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;
        assert_abs_diff_eq!(dbg!(state.score), running);
        std::mem::swap(&mut state_a, &mut state_b);
    }
    assert_abs_diff_eq!(dbg!(state.score), running);
}

#[test]
fn test_bigram_unk() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    let fst_indexer = FstIndexer::new(sections);

    let klm = crate::model::Model::new("test_data/arpa/lm.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let mut running = 0f32;
    for word in &vec![String::from("greaaa"), String::from("have")] {
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;
        assert_abs_diff_eq!(dbg!(state.score), running);
        std::mem::swap(&mut state_a, &mut state_b);
    }
    assert_abs_diff_eq!(dbg!(state.score), running);
}

#[test]
fn test_bigram() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    let fst_indexer = FstIndexer::new(sections);

    let klm = crate::model::Model::new("test_data/arpa/lm.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let mut running = 0f32;
    for word in &vec![String::from("i"), String::from("have")] {
        let idx = *vocab.get_index(dbg!(&word)).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;
        std::mem::swap(dbg!(&mut state_a), dbg!(&mut state_b));
        assert_abs_diff_eq!(dbg!(state.score), running);
    }
    assert_abs_diff_eq!(dbg!(state.score), running);
}

#[test]
fn test_trigram() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    let fst_indexer = FstIndexer::new(sections);

    let klm = crate::model::Model::new("test_data/arpa/lm.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let mut running = 0f32;
    for word in &vec![String::from("i"), String::from("have"), String::from("a")] {
        let idx = *vocab.get_index(dbg!(&word)).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;
        std::mem::swap(dbg!(&mut state_a), dbg!(&mut state_b));
        assert_abs_diff_eq!(dbg!(state.score), running);
    }
    assert_abs_diff_eq!(dbg!(state.score), running);
}

#[test]
fn test_trigram_unk() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    let fst_indexer = FstIndexer::new(sections);

    let klm = crate::model::Model::new("test_data/arpa/lm.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let mut running = 0f32;
    for word in &vec![String::from("i"), String::from("game"), String::from("a")] {
        let idx = *vocab.get_index(dbg!(&word)).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;
        std::mem::swap(dbg!(&mut state_a), dbg!(&mut state_b));
        assert_abs_diff_eq!(dbg!(state.score), running);
    }
    assert_abs_diff_eq!(dbg!(state.score), running);
}

#[test]
fn test_quadgram_unk() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    let fst_indexer = FstIndexer::new(sections);

    let klm = crate::model::Model::new("test_data/arpa/lm.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let mut running = 0f32;
    for word in &vec![
        String::from("i"),
        String::from("have"),
        String::from("a"),
        String::from("troll"),
    ] {
        let idx = *vocab.get_index(dbg!(&word)).unwrap_or(&0);
        eprintln!("Into");
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(dbg!(&mut state_a), &mut state_b, word);
        running += klm_score;
        eprintln!("Swapping");
        std::mem::swap(&mut state_a, dbg!(&mut state_b));
        assert_abs_diff_eq!(dbg!(state.score), running);
    }
    assert_abs_diff_eq!(dbg!(state.score), running);
}

#[test]
fn test_with_trigram_model() {
    let fd = fs::File::open("test_data/arpa/lm_small.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    let fst_indexer = FstIndexer::new(sections);

    let klm = crate::model::Model::new("test_data/arpa/lm_small.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let mut running = 0f32;
    for word in &vec![
        String::from("have"),
        String::from("have"),
        String::from("a"),
    ] {
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state((&mut state_a), &mut state_b, word);
        running += klm_score;
        eprintln!("Swapping");
        if (state.score - running).abs() > 0.01 {
            dbg!(&state_a, &state_b);
        }
        std::mem::swap(&mut state_a, (&mut state_b));

        assert_abs_diff_eq!(dbg!(state.score), running);
    }
    assert_abs_diff_eq!(state.score, -2.6895976);
}

#[test]
fn test_b() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    dbg!(&sections.counts);
    let fst_indexer = FstIndexer::new(sections);

    let mut state = State::default();
    for word in &vec![
        String::from("have"),
        String::from("have"),
        String::from("a"),
    ] {
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        // eprintln!("{state:?}")
    }
    // assert_abs_diff_eq!(state.score, -7.539001);

    let klm = crate::model::Model::new("test_data/arpa/lm.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let mut running = 0f32;
    for word in &vec![
        String::from("have"),
        String::from("a"),
        String::from("game"),
    ] {
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;
        assert_abs_diff_eq!(dbg!(state.score), running);

        std::mem::swap(&mut state_a, &mut state_b);
    }
    eprintln!("{state:?}");
    assert_abs_diff_eq!(state.score, -7.352627);

    let mut state = State::default();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();
    let mut running = 0f32;

    for word in &vec![
        String::from("i"),
        String::from("have"),
        String::from("a"),
        String::from("troll"),
        String::from("at"),
        String::from("home"),
    ] {
        eprintln!("Word: {word}");
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;

        eprintln!("Running: {running:?} vs {:?}", state.score);
        eprintln!("{state_a:?}");
        eprintln!("{idx}, {word} \n\n{state:?}\n\n{klm_score:?} :: {state_b:?}\n\n");
        assert_abs_diff_eq!(dbg!(state.score), running);
        std::mem::swap(&mut state_a, &mut state_b);
    }
    assert_abs_diff_eq!(state.score, -13.404704);

    let mut state = State::default();
    let sent = "you remember i a i".to_string();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();
    let mut running = 0f32;

    for word in sent.split_ascii_whitespace() {
        let idx = *vocab.get_index(&word.to_string()).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);

        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;

        eprintln!("Running: {running:?} vs {:?}", state.score);
        eprintln!("{state_a:?}");
        eprintln!("{idx}, {word} \n\n{state:?}\n\n{klm_score:?} :: {state_b:?}\n\n");
        assert_abs_diff_eq!(dbg!(state.score), running);
        std::mem::swap(&mut state_a, &mut state_b);
    }
    eprintln!("{state_a:?}");
    eprintln!("{state:?}");
    assert_abs_diff_eq!(state.score, -12.566512);

    let mut state = State::default();
    let sent = "i have a good deal of will you remember i a".to_string();
    for word in sent.split_ascii_whitespace() {
        let idx = *vocab.get_index(&word.to_string()).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
    }
    eprintln!("{state:?}");
    assert_abs_diff_eq!(state.score, -23.532686);
    let idx = *vocab.get_index(&"i".to_string()).unwrap_or(&0);
    (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
    eprintln!("{state:?}");
    assert_abs_diff_eq!(state.score, -25.979502);
}

#[test]
fn test_a() {
    let fd = fs::File::open("test_data/arpa/lm_small.arpa").unwrap();
    let br = BufReader::new(fd);
    let hm = Mappings::default();

    let ArpaModel { vocab, sections } = read_arpa(br, IntVocabProcessor(hm)).unwrap();
    let fst_indexer = FstIndexer::new(sections);

    let mut state = State::default();
    for word in &vec![
        String::from("have"),
        String::from("have"),
        String::from("a"),
    ] {
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
    }
    assert_abs_diff_eq!(state.score, -2.6895976);

    let klm = crate::model::Model::new("test_data/arpa/lm_small.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();
    let mut running = 0f32;
    let mut state = State::default();
    for word in &vec![
        String::from("have"),
        String::from("a"),
        String::from("game"),
    ] {
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;
        eprintln!("Running: {running:?} vs {:?}", state.score);
        eprintln!("{state_a:?}");
        eprintln!("{idx}, {word} \n\n{state:?}\n\n{klm_score:?} :: {state_b:?}\n\n");
        assert_abs_diff_eq!(dbg!(state.score), running);
        std::mem::swap(&mut state_a, &mut state_b);
    }
    dbg!(&state);
    assert_abs_diff_eq!(state.score, -3.2568955);

    let mut state = State::default();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();
    let mut running = 0f32;
    for word in &vec![
        String::from("i"),
        String::from("have"),
        String::from("a"),
        String::from("troll"),
        String::from("at"),
        String::from("home"),
    ] {
        let idx = *vocab.get_index(&word).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        running += klm_score;
        eprintln!("Running: {running:?} vs {:?}", state.score);
        eprintln!("{state_a:?}");
        eprintln!("{idx}, {word} \n\n{state:?}\n\n{klm_score:?} :: {state_b:?}\n\n");
        assert_abs_diff_eq!(dbg!(state.score), running);
        std::mem::swap(&mut state_a, &mut state_b);
    }
    assert_abs_diff_eq!(state.score, -6.178563);
    let klm = crate::model::Model::new("test_data/arpa/lm_small.arpa", true).unwrap();
    let mut state_a = klm.new_state();
    let mut state_b = klm.new_state();

    let mut state = State::default();
    let sent = "you remember i a i".to_string();
    for word in sent.split_ascii_whitespace() {
        let idx = *vocab.get_index(&word.to_string()).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
        let klm_score = klm.score_word_given_state(&mut state_a, &mut state_b, word);
        eprintln!(
            "{idx}, {word} klm_score {klm_score:?} {} {:?}",
            state_b.len(),
            state_b.words()
        );

        std::mem::swap(&mut state_a, &mut state_b);
    }
    eprintln!("{state_a:?}");
    eprintln!("{state:?}");
    assert_abs_diff_eq!(state.score, -3.1411633);

    let mut state = State::default();
    let sent = "i have a good deal of will you remember i a".to_string();
    for word in sent.split_ascii_whitespace() {
        let idx = *vocab.get_index(&word.to_string()).unwrap_or(&0);
        (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
    }
    eprintln!("{state:?}");
    assert_abs_diff_eq!(state.score, -2.6173768);
    let idx = *vocab.get_index(&"i".to_string()).unwrap_or(&0);
    (state) = <FstIndexer as NGramIndexer<Vec<u32>>>::find_with_state(&fst_indexer, state, idx);
    eprintln!("{state:?}");
    assert_abs_diff_eq!(state.score, -4.131735);
}

#[test]
fn test_no_data_header() {
    let fd = fs::File::open("test_data/arpa/arpa_no_data_header.arpa").unwrap();
    let buf_read = BufReader::new(fd);
    let err = ArpaReader::new(buf_read, StringProcessor);
    match err {
        Ok(_) => panic!("returned Ok when it should have been `Err(DataHeaderMissing)`"),
        Err(err) => assert!(matches!(err, ArpaReadError::DataHeaderMissing)),
    }
}

#[test]
fn test_no_ngram_counts() {
    let fd = fs::File::open("test_data/arpa/arpa_no_counts.arpa").unwrap();
    let buf_read = BufReader::new(fd);
    let err = ArpaReader::new(buf_read, StringProcessor);
    match err {
        Ok(_) => panic!("returned Ok when it should have been `Err(NgramCountsMissing)`"),
        Err(err) => assert!(matches!(err, ArpaReadError::NgramCountsMissing)),
    }
}

#[test]
fn test_header() {
    let fd = fs::File::open("test_data/arpa/lm.arpa").unwrap();
    let buf_read = BufReader::new(fd);
    let err = ArpaReader::new(buf_read, StringProcessor).unwrap();
    assert_eq!(
        Counts::from_count_vec(vec![
            NGramCardinality::try_from_order_and_cardinality(1, 4415).unwrap(),
            NGramCardinality::try_from_order_and_cardinality(2, 18349).unwrap(),
        ])
        .unwrap(),
        err.counts
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
pub fn get_trigrams() -> Vec<ProbNgram<String>> {
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
pub fn get_bigrams() -> Vec<ProbBackoffNgram<String>> {
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
pub fn get_unigrams() -> Vec<ProbBackoffNgram<String>> {
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
