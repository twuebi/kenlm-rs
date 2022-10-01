pub const DEFAULT_ORDER: &str = "3";

fn main() -> anyhow::Result<()> {
    let max_order = std::env::var("KENLM_MAX_ORDER").unwrap_or_else(|_| {
        eprintln!("cargo:warning=No max-order provided, defaulting to {DEFAULT_ORDER}");
        DEFAULT_ORDER.into()
    });
    println!(
        "cargo:warning=Using max-order: KENLM_MAX_ORDER={max_order}.\
This is a compile-time setting that will limit your runtime.\
Using a higher order model will cause runtime crashes.\n
cargo:warning=Set `KENLM_MAX_ORDER=5` in your env to change it."
    );
    let max_order_flag = format!("-DKENLM_MAX_ORDER={max_order}");

    let mut b = autocxx_build::Builder::new("src/bridge.rs", &[&"src/"])
        .extra_clang_args(&[&max_order_flag])
        .build()?;
    b.flag_if_supported("-std=c++14")
        .extra_warnings(false)
        .warnings(false)
        .flag(&max_order_flag)
        .files(&[
            "src/util/bit_packing.cc",
            "src/util/ersatz_progress.cc",
            "src/util/exception.cc",
            "src/util/file.cc",
            "src/util/file_piece.cc",
            "src/util/float_to_string.cc",
            "src/util/integer_to_string.cc",
            "src/util/mmap.cc",
            "src/util/murmur_hash.cc",
            "src/util/parallel_read.cc",
            "src/util/pool.cc",
            "src/util/read_compressed.cc",
            "src/util/scoped.cc",
            "src/util/spaces.cc",
            "src/util/string_piece.cc",
            "src/util/usage.cc",
            "src/lm/bhiksha.cc",
            "src/lm/binary_format.cc",
            "src/lm/config.cc",
            "src/lm/lm_exception.cc",
            "src/lm/model.cc",
            "src/lm/quantize.cc",
            "src/lm/read_arpa.cc",
            "src/lm/search_hashed.cc",
            "src/lm/search_trie.cc",
            "src/lm/sizes.cc",
            "src/lm/trie.cc",
            "src/lm/trie_sort.cc",
            "src/lm/value_build.cc",
            "src/lm/virtual_interface.cc",
            "src/lm/vocab.cc",
            "src/util/double-conversion/bignum-dtoa.cc",
            "src/util/double-conversion/bignum.cc",
            "src/util/double-conversion/cached-powers.cc",
            "src/util/double-conversion/double-to-string.cc",
            "src/util/double-conversion/fast-dtoa.cc",
            "src/util/double-conversion/fixed-dtoa.cc",
            "src/util/double-conversion/string-to-double.cc",
            "src/util/double-conversion/strtod.cc",
        ])
        .compile("autocxx-kenlm");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
