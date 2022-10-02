# KENLM-RS

This crate is an experimental Rust wrapper around the [kenlm C++ library](https://github.com/kpu/kenlm/). It uses [autocxx](https://google.github.io/autocxx/) to generate the glue-code. It uses a modified version of kenlm located in `src/lm` and `src/util`. Most of the modifications add constructor functions or expose config fields.

Loading ARPA files will likely not work as file-type recognition has not been implemented and constructing a model will try to deserialize the headers of the binary format to perform validation before calling to C++ to do the actual loading. This decision was taken to avoid ugly SIGABRT that would occur if the C++ throws exceptions which are currently [unsupported](https://google.github.io/autocxx/other_features.html?highlight=exception#exceptions) in `autocxx` 

## Maxorder

Kenlm's build-flag `-DKENLM_MAX_ORDER` governs the maximal ngram order you'll be able to load with this library. Loading a model with larger order than the library was built with will cause a runtime exception originating in C++. `-DKENLM_MAX_ORDER` also governs the size of state, you may set it via the env var `KENLM_MAX_ORDER` or by changing the default value in [src/build.rs](src/build.rs). The current default is `3`. Increasing it comes at the cost of increased state-sizes.

The a state is a plain-old-data struct with:

```rust
[c_uint, KENLM_MAX_ORDER-1]
[f32, KENLM_MAX_ORDER-1]
u8
```

## Usage

### Score a sample sentence using a model

```
cargo run --example score_sentence -- --model-path your_model.bin "Some test sentence"
```

### Library

See [examples/score_sentence.rs](examples/score_sentence.rs) and `Model` in [src/model/mod.rs](src/model/mod.rs).

-----------------------------

## Modifications to kenlm

Besides formatting, a few functions were added, mostly to provide easier access from Rust or to help autocxx. An incomplete list can be found below:

### config.cc

In [src/lm/config.cc](src/lm/config.cc), there are 3 added functions. 

```c++
namespace lm
{
  namespace ngram
  {
    std::unique_ptr<Config> Config_Create();
    void Config_set_load_method(Config &config, util::LoadMethod load_method);
    void Config_set_enumerate_callback(Config &config, EnumerateVocab &enumerateCallback);
  }
}
```
- `Config_Create` is a constructor
- `Config_set_load_method` sets the load_method 
- `Config_set_enumerate_callback` sets the enumerate callback that gets executed for each vocab entry, see `VocabCallback` in [src/bridge.rs](src/bridge.rs) for an example callback.


### virtual_interface.cc

In [src/lm/virtual_interface.cc](src/lm/virtual_interface.cc) there is a single added function `LoadVirtualPtr`, it is essentially `LoadVirtual` but returns a unique pointer.
