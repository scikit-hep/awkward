// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/type/RecordType.cpp", line)

#include <string>
#include <sstream>
#include <algorithm>

#include "awkward/array/RecordArray.h"
#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"
#include "awkward/util.h"

#include "awkward/type/RecordType.h"

namespace awkward {
  RecordType::RecordType(const util::Parameters& parameters,
                         const std::string& typestr,
                         const std::vector<TypePtr>& types,
                         const util::RecordLookupPtr& recordlookup)
      : Type(parameters, typestr)
      , types_(types)
      , recordlookup_(recordlookup) {
    if (recordlookup_.get() != nullptr  &&
        recordlookup_.get()->size() != types_.size()) {
      throw std::runtime_error(
        std::string("recordlookup and types must have the same length")
        + FILENAME(__LINE__));
    }
  }

  RecordType::RecordType(const util::Parameters& parameters,
                         const std::string& typestr,
                         const std::vector<TypePtr>& types)
      : Type(parameters, typestr)
      , types_(types)
      , recordlookup_(nullptr) { }

  const std::vector<TypePtr>
  RecordType::types() const {
    return types_;
  };

  const util::RecordLookupPtr
  RecordType::recordlookup() const {
    return recordlookup_;
  }

  bool
  RecordType::istuple() const {
    return recordlookup_.get() == nullptr;
  }

  const std::vector<std::string> datashape_keywords({
      "var", "option",
      "bool", "int8", "int16", "int32", "int64", "int128",
      "uint8", "uint16", "uint32", "uint64", "uint128",
      "float16", "float32", "float64", "float128",
      "decimal32", "decimal64", "decimal128",
      "bignum", "int", "real", "complex", "intptr", "uintptr",
      "string", "char", "bytes", "date", "json",
      "void", "datetime", "categorical", "pointer"});

  std::string
  RecordType::tostring_part(const std::string& indent,
                            const std::string& pre,
                            const std::string& post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return wrap_categorical(typestr);
    }

    std::stringstream out;
    if (parameters_.size() == 1  &&  parameter_isname("__record__")) {
      std::string name = parameter_asstring("__record__");
      auto item = std::find(datashape_keywords.begin(),
                            datashape_keywords.end(),
                            name);
      if (item == datashape_keywords.end()) {
        out << name << "[";
        for (size_t j = 0;  j < types_.size();  j++) {
          if (j != 0) {
            out << ", ";
          }
          if (recordlookup_.get() != nullptr) {
            out << util::quote(recordlookup_.get()->at(j)) << ": ";
          }
          out << types_[j].get()->tostring_part("", "", "");
        }
        out << "]";
        return wrap_categorical(out.str());
      }
    }

    if (parameters_empty()) {
      if (recordlookup_.get() != nullptr) {
        out << "{";
        for (size_t j = 0;  j < types_.size();  j++) {
          if (j != 0) {
            out << ", ";
          }
          out << util::quote(recordlookup_.get()->at(j)) << ": ";
          out << types_[j].get()->tostring_part("", "", "");
        }
        out << "}";
      }
      else {
        out << "(";
        for (size_t j = 0;  j < types_.size();  j++) {
          if (j != 0) {
            out << ", ";
          }
          out << types_[j].get()->tostring_part("", "", "");
        }
        out << ")";
      }
    }
    else {
      if (recordlookup_.get() != nullptr) {
        out << "struct[[";
        for (size_t j = 0;  j < types_.size();  j++) {
          if (j != 0) {
            out << ", ";
          }
          out << util::quote(recordlookup_.get()->at(j));
        }
        out << "], [";
        for (size_t j = 0;  j < types_.size();  j++) {
          if (j != 0) {
            out << ", ";
          }
          out << types_[j].get()->tostring_part("", "", "");
        }
      }
      else {
        out << "tuple[[";
        for (size_t j = 0;  j < types_.size();  j++) {
          if (j != 0) {
            out << ", ";
          }
          out << types_[j].get()->tostring_part("", "", "");
        }
      }
      out << "], " << string_parameters() << "]";
    }
    return wrap_categorical(out.str());
  }

  const TypePtr
  RecordType::shallow_copy() const {
    return std::make_shared<RecordType>(parameters_,
                                        typestr_,
                                        types_,
                                        recordlookup_);
  }

  bool
  RecordType::equal(const TypePtr& other, bool check_parameters) const {
    if (RecordType* t = dynamic_cast<RecordType*>(other.get())) {
      if (check_parameters  &&  !parameters_equal(other.get()->parameters())) {
        return false;
      }
      if (numfields() != t->numfields()) {
        return false;
      }
      if (recordlookup_.get() != nullptr) {
        if (t->istuple()) {
          return false;
        }
        for (auto key : keys()) {
          if (!t->haskey(key)) {
            return false;
          }
          if (!field(key).get()->equal(t->field(key), check_parameters)) {
            return false;
          }
        }
        return true;
      }
      else {
        if (!t->istuple()) {
          return false;
        }
        for (int64_t j = 0;  j < numfields();  j++) {
          if (!field(j).get()->equal(t->field(j), check_parameters)) {
            return false;
          }
        }
        return true;
      }
    }
    else {
      return false;
    }
  }

  int64_t
  RecordType::numfields() const {
    return (int64_t)types_.size();
  }

  int64_t
  RecordType::fieldindex(const std::string& key) const {
    return util::fieldindex(recordlookup_, key, numfields());
  }

  const std::string
  RecordType::key(int64_t fieldindex) const {
    return util::key(recordlookup_, fieldindex, numfields());
  }

  bool
  RecordType::haskey(const std::string& key) const {
    return util::haskey(recordlookup_, key, numfields());
  }

  const std::vector<std::string>
  RecordType::keys() const {
    return util::keys(recordlookup_, numfields());
  }

  const ContentPtr
  RecordType::empty() const {
    ContentPtrVec contents;
    for (auto type : types_) {
      contents.push_back(type.get()->empty());
    }
    return std::make_shared<RecordArray>(Identities::none(),
                                         parameters_,
                                         contents,
                                         recordlookup_);
  }

  const TypePtr
  RecordType::field(int64_t fieldindex) const {
    if (fieldindex >= numfields()) {
      throw std::invalid_argument(
        std::string("fieldindex ") + std::to_string(fieldindex)
        + std::string(" for record with only ") + std::to_string(numfields())
        + std::string(" fields") + FILENAME(__LINE__));
    }
    return types_[(size_t)fieldindex];
  }

  const TypePtr
  RecordType::field(const std::string& key) const {
    return types_[(size_t)fieldindex(key)];
  }

  const std::vector<TypePtr>
  RecordType::fields() const {
    return std::vector<TypePtr>(types_);
  }

  const std::vector<std::pair<std::string, TypePtr>>
  RecordType::fielditems() const {
    std::vector<std::pair<std::string, TypePtr>> out;
    if (recordlookup_.get() != nullptr) {
      size_t cols = types_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(
          std::pair<std::string, TypePtr>(recordlookup_.get()->at(j),
                                          types_[j]));
      }
    }
    else {
      size_t cols = types_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(
          std::pair<std::string, TypePtr>(std::to_string(j),
                                          types_[j]));
      }
    }
    return out;
  }

  const TypePtr
  RecordType::astuple() const {
    return std::make_shared<RecordType>(parameters_,
                                        typestr_,
                                        types_,
                                        util::RecordLookupPtr(nullptr));
  }

}
