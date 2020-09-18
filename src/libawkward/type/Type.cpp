// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "rapidjson/document.h"

#include "awkward/util.h"

#include "awkward/type/Type.h"

namespace rj = rapidjson;

namespace awkward {
  Type::Type(const util::Parameters& parameters, const std::string& typestr)
      : parameters_(parameters)
      , typestr_(typestr) { }

  Type::~Type() = default;

  const util::Parameters
  Type::parameters() const {
    return parameters_;
  }

  const std::string
  Type::typestr() const {
    return typestr_;
  }

  void
  Type::setparameters(const util::Parameters& parameters) {
    parameters_ = parameters;
  }

  const std::string
  Type::parameter(const std::string& key) const {
    auto item = parameters_.find(key);
    if (item == parameters_.end()) {
      return "null";
    }
    return item->second;
  }

  void
  Type::setparameter(const std::string& key, const std::string& value) {
    if (value == std::string("null")) {
      parameters_.erase(key);
    }
    else {
      parameters_[key] = value;
    }
  }

  bool
  Type::parameter_equals(const std::string& key,
                         const std::string& value) const {
    return util::parameter_equals(parameters_, key, value);
  }

  bool
  Type::parameters_equal(const util::Parameters& other) const {
    return util::parameters_equal(parameters_, other);
  }

  bool
  Type::parameter_isstring(const std::string& key) const {
    return util::parameter_isstring(parameters_, key);
  }

  bool
  Type::parameter_isname(const std::string& key) const {
    return util::parameter_isname(parameters_, key);
  }

  const std::string
  Type::parameter_asstring(const std::string& key) const {
    return util::parameter_asstring(parameters_, key);
  }

  const std::string
  Type::tostring() const {
    return tostring_part("", "", "");
  };

  const std::string
  Type::compare(TypePtr supertype) {
    // FIXME: better side-by-side comparison
    return tostring() + std::string(" versus ") + supertype.get()->tostring();
  }

  bool
  Type::get_typestr(std::string& output) const {
    if (typestr_.empty()) {
      return false;
    }
    else {
      output.assign(typestr_);
      return true;
    }
  }

  bool
  Type::parameters_empty() const {
    if (parameters_.empty()) {
      return true;
    }
    else if (parameters_.size() == 1  &&  parameter_equals("__categorical__", "true")) {
      return true;
    }
    else {
      return false;
    }
  }

  std::string
  Type::wrap_categorical(const std::string& output) const {
    if (parameter_equals("__categorical__", "true")) {
      return std::string("categorical[type=") + output + std::string("]");
    }
    else {
      return output;
    }
  }

  const std::string
  Type::string_parameters() const {
    std::stringstream out;
    out << "parameters={";
    bool first = true;
    for (auto pair : parameters_) {
      if (pair.first != std::string("__categorical__")) {
        if (!first) {
          out << ", ";
        }
        out << util::quote(pair.first) << ": " << pair.second;
        first = false;
      }
    }
    out << "}";
    return out.str();
  }
}
