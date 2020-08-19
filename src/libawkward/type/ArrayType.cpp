// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/type/ArrayType.cpp", line)

#include <string>

#include "awkward/type/ArrayType.h"

namespace awkward {
  ArrayType::ArrayType(const util::Parameters& parameters,
                       const std::string& typestr,
                       const TypePtr& type,
                       int64_t length)
      : Type(parameters, typestr)
      , type_(type)
      , length_(length) { }

  std::string
  ArrayType::tostring_part(const std::string& indent,
                           const std::string& pre,
                           const std::string& post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return typestr;
    }
    return (indent + pre + std::to_string(length_) + " * "
            + type_.get()->tostring_part(indent, "", "") + post);
  }

  const TypePtr
  ArrayType::shallow_copy() const {
    return std::make_shared<ArrayType>(parameters_, typestr_, type_, length_);
  }

  bool
  ArrayType::equal(const TypePtr& other, bool check_parameters) const {
    if (ArrayType* t = dynamic_cast<ArrayType*>(other.get())) {
      if (check_parameters  &&  !parameters_equal(other.get()->parameters())) {
        return false;
      }
      return (length_ == t->length_  &&
              type_.get()->equal(t->type_, check_parameters));
    }
    else {
      return false;
    }
  }

  int64_t
  ArrayType::numfields() const {
    return type_.get()->numfields();
  }

  int64_t
  ArrayType::fieldindex(const std::string& key) const {
    return type_.get()->fieldindex(key);
  }

  const std::string
  ArrayType::key(int64_t fieldindex) const {
    return type_.get()->key(fieldindex);
  }

  bool
  ArrayType::haskey(const std::string& key) const {
    return type_.get()->haskey(key);
  }

  const std::vector<std::string>
  ArrayType::keys() const {
    return type_.get()->keys();
  }

  const ContentPtr
  ArrayType::empty() const {
    if (length_ != 0) {
      throw std::invalid_argument(
        std::string("ArrayType with length ") + std::to_string(length_)
        + std::string(" does not describe an empty array") + FILENAME(__LINE__));
    }
    return type_.get()->empty();
  }

  int64_t
  ArrayType::length() const {
    return length_;
  }

  const TypePtr
  ArrayType::type() const {
    return type_;
  }
}
