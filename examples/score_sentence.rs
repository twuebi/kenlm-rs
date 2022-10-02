use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[clap(long)]
    model_path: PathBuf,
    #[clap(default_value = "This is a test sentence.")]
    sentence: String,
    #[clap(action, short = 'b', default_value = "false")]
    score_bos: bool,
    #[clap(action, short = 'e', default_value = "false")]
    score_eos: bool,
}

fn main() -> anyhow::Result<(), anyhow::Error> {
    let Args {
        model_path,
        sentence,
        score_bos,
        score_eos,
    } = Args::parse();

    let model = kenlm_rs::Model::new(
        model_path
            .to_str()
            .ok_or(anyhow::anyhow!("Path could not be converted into &str"))?,
        true,
    )?;

    // We constructed with_vocab: true
    let vocab_ref = model.get_vocab().unwrap();
    eprintln!(
        "The vocab has {} elements. The first element is: {:?}, the tenth: {:?} and the last: {:?}",
        vocab_ref.len(),
        vocab_ref.get(0),
        vocab_ref.get(10),
        vocab_ref[vocab_ref.len() - 1]
    );
    let inputs = sentence.split_ascii_whitespace().collect::<Vec<&str>>();
    let score = model.score_sentence(&inputs, score_bos, score_eos);
    eprintln!(
        "Total score of the sentence \"{}\", calculated from rust: {:?}",
        inputs.join(" "),
        score
    );

    Ok(())
}
