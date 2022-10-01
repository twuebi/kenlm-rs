#include "config.hh"

#include <iostream>

namespace lm
{
  namespace ngram
  {

    Config::Config() : show_progress(true),
                       messages(&std::cerr),
                       enumerate_vocab(NULL),
                       unknown_missing(COMPLAIN),
                       sentence_marker_missing(THROW_UP),
                       positive_log_probability(THROW_UP),
                       unknown_missing_logprob(-100.0),
                       probing_multiplier(1.5),
                       building_memory(1073741824ULL), // 1 GB
                       temporary_directory_prefix(""),
                       arpa_complain(ALL),
                       write_mmap(NULL),
                       write_method(WRITE_AFTER),
                       include_vocab(true),
                       rest_function(REST_MAX),
                       prob_bits(8),
                       backoff_bits(8),
                       pointer_bhiksha_bits(22),
                       load_method(util::POPULATE_OR_READ)
    {
    }

  } // namespace ngram
} // namespace lm

namespace lm
{
  namespace ngram
  {
    std::unique_ptr<Config> Config_Create()
    {
      return std::make_unique<Config>();
    }
    void Config_set_load_method(Config &config, util::LoadMethod load_method)
    {
      config.load_method = load_method;
    }
    void Config_set_enumerate_callback(Config &config, EnumerateVocab &enumerateCallback)
    {
      EnumerateVocab* enumerateCallbackPtr = &enumerateCallback;
      config.enumerate_vocab = enumerateCallbackPtr;
    }
  }
}
