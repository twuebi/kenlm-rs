#include "virtual_interface.hh"

#include "lm_exception.hh"
#include <memory>
#include "model.hh"
#include "model_type.hh"
#include "config.hh"
#include <iostream>

namespace lm
{
  namespace base
  {

    Vocabulary::~Vocabulary() {}

    void Vocabulary::SetSpecial(WordIndex begin_sentence, WordIndex end_sentence, WordIndex not_found)
    {
      begin_sentence_ = begin_sentence;
      end_sentence_ = end_sentence;
      not_found_ = not_found;
    }

    Model::~Model() {}

    ::std::unique_ptr<base::Model> LoadVirtualPtr(const ::std::string &file_name, const ::lm::ngram::Config &config)
    {
      lm::ngram::ModelType model_type = lm::ngram::ModelType::PROBING;
      lm::ngram::RecognizeBinary(file_name.c_str(), model_type);
      switch (model_type)
      {
      case lm::ngram::PROBING:
        return ::std::make_unique<::lm::ngram::ProbingModel>(file_name.c_str(), config);
      case lm::ngram::REST_PROBING:
        return ::std::make_unique<::lm::ngram::RestProbingModel>(file_name.c_str(), config);
      case lm::ngram::TRIE:
        return ::std::make_unique<::lm::ngram::TrieModel>(file_name.c_str(), config);
      case lm::ngram::QUANT_TRIE:
        return ::std::make_unique<::lm::ngram::QuantTrieModel>(file_name.c_str(), config);
      case lm::ngram::ARRAY_TRIE:
        return ::std::make_unique<::lm::ngram::ArrayTrieModel>(file_name.c_str(), config);
      case lm::ngram::QUANT_ARRAY_TRIE:
        return ::std::make_unique<::lm::ngram::QuantArrayTrieModel>(file_name.c_str(), config);
      default:
        UTIL_THROW(FormatLoadException, "Confused by model type " << model_type);
      }
    }
    std::unique_ptr<Config> Config_Create()
    {
      return std::make_unique<Config>();
    }
  } // namespace base
} // namespace lm
