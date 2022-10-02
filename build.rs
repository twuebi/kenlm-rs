pub const DEFAULT_ORDER: &str = "3";

fn main() -> anyhow::Result<()> {
    println!("cargo:rerun-if-changed=src/cxx/bridge.rs");
    println!("cargo:rerun-if-changed=build.rs");
    // println!("cargo:rerun-if-env-changed=KENLM_MAX_ORDER");
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

    let mut b = autocxx_build::Builder::new("src/cxx/bridge.rs", &[&"src/cxx/"])
        .extra_clang_args(&[&max_order_flag])
        .build()?;
    b.flag_if_supported("-std=c++14")
        .extra_warnings(false)
        .warnings(false)
        .flag(&max_order_flag)
        .files(&[
            "src/cxx/util/bit_packing.cc",
            "src/cxx/util/ersatz_progress.cc",
            "src/cxx/util/exception.cc",
            "src/cxx/util/file.cc",
            "src/cxx/util/file_piece.cc",
            "src/cxx/util/float_to_string.cc",
            "src/cxx/util/integer_to_string.cc",
            "src/cxx/util/mmap.cc",
            "src/cxx/util/murmur_hash.cc",
            "src/cxx/util/parallel_read.cc",
            "src/cxx/util/pool.cc",
            "src/cxx/util/read_compressed.cc",
            "src/cxx/util/scoped.cc",
            "src/cxx/util/spaces.cc",
            "src/cxx/util/string_piece.cc",
            "src/cxx/util/usage.cc",
            "src/cxx/lm/bhiksha.cc",
            "src/cxx/lm/binary_format.cc",
            "src/cxx/lm/config.cc",
            "src/cxx/lm/lm_exception.cc",
            "src/cxx/lm/model.cc",
            "src/cxx/lm/quantize.cc",
            "src/cxx/lm/read_arpa.cc",
            "src/cxx/lm/search_hashed.cc",
            "src/cxx/lm/search_trie.cc",
            "src/cxx/lm/sizes.cc",
            "src/cxx/lm/trie.cc",
            "src/cxx/lm/trie_sort.cc",
            "src/cxx/lm/value_build.cc",
            "src/cxx/lm/virtual_interface.cc",
            "src/cxx/lm/vocab.cc",
            "src/cxx/util/double-conversion/bignum-dtoa.cc",
            "src/cxx/util/double-conversion/bignum.cc",
            "src/cxx/util/double-conversion/cached-powers.cc",
            "src/cxx/util/double-conversion/double-to-string.cc",
            "src/cxx/util/double-conversion/fast-dtoa.cc",
            "src/cxx/util/double-conversion/fixed-dtoa.cc",
            "src/cxx/util/double-conversion/string-to-double.cc",
            "src/cxx/util/double-conversion/strtod.cc",
        ])
        .compile("autocxx-kenlm");
    Ok(())
}
