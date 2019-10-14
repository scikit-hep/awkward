// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/ArrayType.h"

namespace awkward {
  std::string ArrayType::tostring_part(std::string indent, std::string pre, std::string post) const {
    return indent + pre + std::to_string(length_) + " * " + type_.get()->tostring_part(indent, "", "") + post;
  }

  bool ArrayType::equal(std::shared_ptr<Type> other) const {
    if (ArrayType* t = dynamic_cast<ArrayType*>(other.get())) {
      return type_.get()->equal(t->type_);   // lengths DO NOT need to be equal (unlike RegularType)
    }
    else {
      return false;
    }
  }

  int64_t ArrayType::length() const {
    return length_;
  }

  std::shared_ptr<Type> ArrayType::type() const {
    return type_;
  }
}
