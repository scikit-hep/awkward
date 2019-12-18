// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/ArrayType.h"

namespace awkward {
  std::string ArrayType::tostring_part(std::string indent, std::string pre, std::string post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return typestr;
    }

    return indent + pre + std::to_string(length_) + " * " + type_.get()->tostring_part(indent, "", "") + post;
  }

  const std::shared_ptr<Type> ArrayType::shallow_copy() const {
    return std::make_shared<ArrayType>(parameters_, type_, length_);
  }

  bool ArrayType::equal(const std::shared_ptr<Type> other, bool check_parameters) const {
    if (ArrayType* t = dynamic_cast<ArrayType*>(other.get())) {
      if (check_parameters  &&  !equal_parameters(other.get()->parameters())) {
        return false;
      }
      return length_ == t->length_  &&  type_.get()->equal(t->type_, check_parameters);
    }
    else {
      return false;
    }
  }

  int64_t ArrayType::numfields() const {
    return type_.get()->numfields();
  }

  int64_t ArrayType::fieldindex(const std::string& key) const {
    return type_.get()->fieldindex(key);
  }

  const std::string ArrayType::key(int64_t fieldindex) const {
    return type_.get()->key(fieldindex);
  }

  bool ArrayType::haskey(const std::string& key) const {
    return type_.get()->haskey(key);
  }

  const std::vector<std::string> ArrayType::keyaliases(int64_t fieldindex) const {
    return type_.get()->keyaliases(fieldindex);
  }

  const std::vector<std::string> ArrayType::keyaliases(const std::string& key) const {
    return type_.get()->keyaliases(key);
  }

  const std::vector<std::string> ArrayType::keys() const {
    return type_.get()->keys();
  }

  int64_t ArrayType::length() const {
    return length_;
  }

  const std::shared_ptr<Type> ArrayType::type() const {
    return type_;
  }
}
