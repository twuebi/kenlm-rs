pub mod arpa;

#[derive(Debug, Clone)]
pub struct ProbBackoff {
    pub log_prob: f32,
    pub backoff: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NGram {
    ngram: String, // TODO: this sensible?
}

#[derive(Debug, Clone)]
pub struct ProbBackoffNgram {
    pub ngram: NGram,
    pub prob_backoff: ProbBackoff,
}

#[derive(Debug, Clone)]
pub struct ProbNgram {
    pub ngram: NGram,
    pub prob: f32,
}
