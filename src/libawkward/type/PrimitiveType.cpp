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
    return std::shared_ptr<Type>(new PrimitiveType(dtype_));
  }

  bool PrimitiveType::equal(std::shared_ptr<Type> other) const {
    if (PrimitiveType* t = dynamic_cast<PrimitiveType*>(other.get())) {
      return dtype_ == t->dtype_;
    }
    else {
      return false;
    }
  }

  const PrimitiveType::DType PrimitiveType::dtype() const {
    return dtype_;
  }
}
