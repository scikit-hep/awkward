// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>

#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/PrimitiveType.h"

namespace awkward {
  std::string PrimitiveType::tostring_part(std::string indent, std::string pre, std::string post) const {
    std::string s;
    switch (dtype_) {
      case boolean: s = "bool"; break;
      case int8:    s = "int8"; break;
      case int16:   s = "int16"; break;
      case int32:   s = "int32"; break;
      case int64:   s = "int64"; break;
      case uint8:   s = "uint8"; break;
      case uint16:  s = "uint16"; break;
      case uint32:  s = "uint32"; break;
      case uint64:  s = "uint64"; break;
      case float32: s = "float32"; break;
      case float64: s = "float64"; break;
      default:      assert(dtype_ < numtypes);
    }
    return indent + pre + s + post;
  }

  const std::shared_ptr<Type> PrimitiveType::shallow_copy() const {
    return std::shared_ptr<Type>(new PrimitiveType(parameters_FIXME_, dtype_));
  }

  bool PrimitiveType::shallow_equal(const std::shared_ptr<Type> other, bool check_parameters) const {
    return equal(other, check_parameters);
  }

  bool PrimitiveType::equal(const std::shared_ptr<Type> other, bool check_parameters) const {
    if (PrimitiveType* t = dynamic_cast<PrimitiveType*>(other.get())) {
      if (check_parameters  &&  !equal_parameters(other.get()->parameters())) {
        return false;
      }
      return dtype_ == t->dtype_;
    }
    else {
      return false;
    }
  }

  std::shared_ptr<Type> PrimitiveType::level() const {
    return shallow_copy();
  }

  std::shared_ptr<Type> PrimitiveType::inner() const {
    throw std::invalid_argument("PrimitiveType has no inner type");
  }

  std::shared_ptr<Type> PrimitiveType::inner(const std::string& key) const {
    throw std::invalid_argument("PrimitiveType has no inner type");
  }

  int64_t PrimitiveType::numfields() const {
    return -1;
  }

  int64_t PrimitiveType::fieldindex(const std::string& key) const {
    throw std::invalid_argument("type contains no Records");
  }

  const std::string PrimitiveType::key(int64_t fieldindex) const {
    throw std::invalid_argument("type contains no Records");
  }

  bool PrimitiveType::haskey(const std::string& key) const {
    throw std::invalid_argument("type contains no Records");
  }

  const std::vector<std::string> PrimitiveType::keyaliases(int64_t fieldindex) const {
    throw std::invalid_argument("type contains no Records");
  }

  const std::vector<std::string> PrimitiveType::keyaliases(const std::string& key) const {
    throw std::invalid_argument("type contains no Records");
  }

  const std::vector<std::string> PrimitiveType::keys() const {
    throw std::invalid_argument("type contains no Records");
  }

  const PrimitiveType::DType PrimitiveType::dtype() const {
    return dtype_;
  }
}
