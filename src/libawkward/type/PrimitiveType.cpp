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

  bool PrimitiveType::compatible(std::shared_ptr<Type> other) const {
    if (UnknownType* t = dynamic_cast<UnknownType*>(other.get())) {
      return true;
    }
    else if (OptionType* t = dynamic_cast<OptionType*>(other.get())) {
      return compatible(t->type());
    }
    else if (PrimitiveType* t = dynamic_cast<PrimitiveType*>(other.get())) {
      switch (dtype_) {
        case boolean:
        switch (t->dtype_) {
          case boolean:
          return true;
          default:
          return false;
        }
        case int8:
        case int16:
        case int32:
        case int64:
        case uint8:
        case uint16:
        case uint32:
        case uint64:
        switch (t->dtype_) {
          case int8:
          case int16:
          case int32:
          case int64:
          case uint8:
          case uint16:
          case uint32:
          case uint64:
          return true;
          default:
          return false;
        }
        case float32:
        case float64:
        switch (t->dtype_) {
          case float32:
          case float64:
          return true;
          default:
          return false;
        }
        default:
        return false;
      }
    }
    else {
      return false;
    }
  }
}
