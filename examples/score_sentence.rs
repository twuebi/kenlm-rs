use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[clap(long, default_value = "test_data/carol.bin")]
    model_path: PathBuf,
    #[clap(
        default_value = "the register of his burial was signed by the clergyman the clerk the undertaker and the chief mourner"
    )]
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

    let inputs = sentence.split_ascii_whitespace().collect::<Vec<&str>>();
    let score = model.score_sentence(&inputs, score_bos, score_eos);
    eprintln!(
        "Total score of the sentence \"{}\" is: {:?}",
        inputs.join(" "),
        score
    );

    Ok(())
}
