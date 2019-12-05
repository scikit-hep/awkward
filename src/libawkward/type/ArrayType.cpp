// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/ArrayType.h"

namespace awkward {
  std::string ArrayType::tostring_part(std::string indent, std::string pre, std::string post) const {
    return indent + pre + std::to_string(length_) + " * " + type_.get()->tostring_part(indent, "", "") + post;
  }

  const std::shared_ptr<Type> ArrayType::shallow_copy() const {
    return std::shared_ptr<Type>(new ArrayType(type_, length_));
  }

  bool ArrayType::shallow_equal(const std::shared_ptr<Type> other) const {
    return (dynamic_cast<ArrayType*>(other.get()) != nullptr);
  }

  bool ArrayType::equal(const std::shared_ptr<Type> other) const {
    if (ArrayType* t = dynamic_cast<ArrayType*>(other.get())) {
      return length_ == t->length_  &&  type_.get()->equal(t->type_);
    }
    else {
      return false;
    }
  }

  std::shared_ptr<Type> ArrayType::level() const {
    return shallow_copy();
  }

  std::shared_ptr<Type> ArrayType::inner() const {
    return type_;
  }

  std::shared_ptr<Type> ArrayType::inner(const std::string& key) const {
    throw std::runtime_error("FIXME: ArrayType::inner(key)");
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
