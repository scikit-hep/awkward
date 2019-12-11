// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"
#include "awkward/util.h"

#include "awkward/type/RecordType.h"

namespace awkward {
  std::string RecordType::tostring_part(std::string indent, std::string pre, std::string post) const {
    std::stringstream out;
    if (reverselookup_.get() == nullptr) {
      out << "(";
      for (size_t j = 0;  j < types_.size();  j++) {
        if (j != 0) {
          out << ", ";
        }
        out << types_[j].get()->tostring_part("", "", "");
      }
      out << ")";
    }
    else {
      out << "{";
      for (size_t j = 0;  j < types_.size();  j++) {
        if (j != 0) {
          out << ", ";
        }
        out << util::quote(reverselookup_.get()->at(j), false) << ": ";
        out << types_[j].get()->tostring_part("", "", "");
      }
      out << "}";
    }
    return out.str();
  }

  const std::shared_ptr<Type> RecordType::shallow_copy() const {
    return std::shared_ptr<Type>(new RecordType(parameters_FIXME_, types_, lookup_, reverselookup_));
  }

  bool RecordType::equal(const std::shared_ptr<Type> other, bool check_parameters) const {
    if (RecordType* t = dynamic_cast<RecordType*>(other.get())) {
      if (check_parameters  &&  !equal_parameters(other.get()->parameters())) {
        return false;
      }
      if (numfields() != t->numfields()) {
        return false;
      }
      if (reverselookup_.get() == nullptr) {
        if (t->reverselookup().get() != nullptr) {
          return false;
        }
        for (int64_t j = 0;  j < numfields();  j++) {
          if (!field(j).get()->equal(t->field(j), check_parameters)) {
            return false;
          }
        }
        return true;
      }
      else {
        if (t->reverselookup().get() == nullptr) {
          return false;
        }
        if (lookup_.get()->size() != t->lookup().get()->size()) {
          return false;
        }
        for (auto pair : *lookup_.get()) {
          int64_t otherindex;
          try {
            otherindex = (int64_t)t->lookup().get()->at(pair.first);
          }
          catch (std::out_of_range err) {
            return false;
          }
          if (!field((int64_t)pair.second).get()->equal(t->field(otherindex), check_parameters)) {
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

  std::shared_ptr<Type> RecordType::level() const {
    return shallow_copy();
  }

  std::shared_ptr<Type> RecordType::inner() const {
    throw std::invalid_argument("RecordType has no inner type without a key");
  }

  std::shared_ptr<Type> RecordType::inner(const std::string& key) const {
    return field(key);
  }

  int64_t RecordType::numfields() const {
    return (int64_t)types_.size();
  }

  int64_t RecordType::fieldindex(const std::string& key) const {
    int64_t out = -1;
    if (lookup_.get() != nullptr) {
      try {
        out = (int64_t)lookup_.get()->at(key);
      }
      catch (std::out_of_range err) { }
      if (out != -1  &&  out >= numfields()) {
        throw std::invalid_argument(std::string("key \"") + key + std::string("\" points to fieldindex ") + std::to_string(out) + std::string(" for RecordType with only " + std::to_string(numfields()) + std::string(" fields")));
      }
    }
    if (out == -1) {
      try {
        out = (int64_t)std::stoi(key);
      }
      catch (std::invalid_argument err) {
        throw std::invalid_argument(std::string("key \"") + key + std::string("\" is not in RecordType"));
      }
      if (out >= numfields()) {
        throw std::invalid_argument(std::string("key interpreted as fieldindex ") + key + std::string(" for RecordType with only " + std::to_string(numfields()) + std::string(" fields")));
      }
    }
    return out;
  }

  const std::string RecordType::key(int64_t fieldindex) const {
    if (fieldindex >= numfields()) {
      throw std::invalid_argument(std::string("fieldindex ") + std::to_string(fieldindex) + std::string(" for RecordType with only " + std::to_string(numfields()) + std::string(" fields")));
    }
    if (reverselookup_.get() != nullptr) {
      return reverselookup_.get()->at((size_t)fieldindex);
    }
    else {
      return std::to_string(fieldindex);
    }
  }

  bool RecordType::haskey(const std::string& key) const {
    try {
      fieldindex(key);
    }
    catch (std::invalid_argument err) {
      return false;
    }
    return true;
  }

  const std::vector<std::string> RecordType::keyaliases(int64_t fieldindex) const {
    std::vector<std::string> out;
    std::string _default = std::to_string(fieldindex);
    bool has_default = false;
    if (lookup_.get() != nullptr) {
      for (auto pair : *lookup_.get()) {
        if (pair.second == fieldindex) {
          out.push_back(pair.first);
          if (pair.first == _default) {
            has_default = true;
          }
        }
      }
    }
    if (!has_default) {
      out.push_back(_default);
    }
    return out;
  }

  const std::vector<std::string> RecordType::keyaliases(const std::string& key) const {
    return keyaliases(fieldindex(key));
  }

  const std::vector<std::string> RecordType::keys() const {
    std::vector<std::string> out;
    if (reverselookup_.get() == nullptr) {
      int64_t cols = numfields();
      for (int64_t j = 0;  j < cols;  j++) {
        out.push_back(std::to_string(j));
      }
    }
    else {
      out.insert(out.end(), reverselookup_.get()->begin(), reverselookup_.get()->end());
    }
    return out;
  }

  const std::shared_ptr<Type> RecordType::field(int64_t fieldindex) const {
    if (fieldindex >= numfields()) {
      throw std::invalid_argument(std::string("fieldindex ") + std::to_string(fieldindex) + std::string(" for RecordType with only " + std::to_string(numfields()) + std::string(" fields")));
    }
    return types_[(size_t)fieldindex];
  }

  const std::shared_ptr<Type> RecordType::field(const std::string& key) const {
    return types_[(size_t)fieldindex(key)];
  }

  const std::vector<std::shared_ptr<Type>> RecordType::fields() const {
    return std::vector<std::shared_ptr<Type>>(types_);
  }

  const std::vector<std::pair<std::string, std::shared_ptr<Type>>> RecordType::fielditems() const {
    std::vector<std::pair<std::string, std::shared_ptr<Type>>> out;
    if (reverselookup_.get() == nullptr) {
      size_t cols = types_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, std::shared_ptr<Type>>(std::to_string(j), types_[j]));
      }
    }
    else {
      size_t cols = types_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, std::shared_ptr<Type>>(reverselookup_.get()->at(j), types_[j]));
      }
    }
    return out;
  }
}
