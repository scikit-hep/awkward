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

  bool PrimitiveType::compatible(std::shared_ptr<Type> other, bool bool_is_int, bool int_is_float, bool ignore_null, bool unknown_is_anything) const {
    if (unknown_is_anything  &&  dynamic_cast<UnknownType*>(other.get())) {
      return true;
    }
    else if (ignore_null  &&  dynamic_cast<OptionType*>(other.get())) {
      return compatible(dynamic_cast<OptionType*>(other.get())->type(), bool_is_int, int_is_float, ignore_null, unknown_is_anything);
    }
    else if (PrimitiveType* t = dynamic_cast<PrimitiveType*>(other.get())) {
      DType me = dtype_;
      DType you = t->dtype();
      if (bool_is_int) {
        if (me == boolean) {
          me = int8;
        }
        if (you == boolean) {
          you = int8;
        }
      }
      if (int_is_float) {
        if (me == int8  ||  me == int16  ||  me == int32  ||  me == int64  ||  me == uint8  ||  me == uint16  ||  me == uint32  ||  me == uint64) {
          me = float64;
        }
        if (you == int8  ||  you == int16  ||  you == int32  ||  you == int64  ||  you == uint8  ||  you == uint16  ||  you == uint32  ||  you == uint64) {
          you = float64;
        }
      }
      switch (me) {
        case boolean:
        switch (you) {
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
        switch (you) {
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
        switch (you) {
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

  const PrimitiveType::DType PrimitiveType::dtype() const {
    return dtype_;
  }
}
