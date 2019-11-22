// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/UnknownType.h"

namespace awkward {
  std::string UnknownType::tostring_part(std::string indent, std::string pre, std::string post) const {
    return indent + pre + "unknown" + post;
  }

  const std::shared_ptr<Type> UnknownType::shallow_copy() const {
    return std::shared_ptr<Type>(new UnknownType());
  }

  bool UnknownType::equal(std::shared_ptr<Type> other) const {
    if (UnknownType* t = dynamic_cast<UnknownType*>(other.get())) {
      return true;
    }
    else {
      return false;
    }
  }

  bool UnknownType::compatible(std::shared_ptr<Type> other, bool bool_is_int, bool int_is_float, bool ignore_null, bool unknown_is_anything) const {
    if (UnknownType* t = dynamic_cast<UnknownType*>(other.get())) {
      return true;
    }
    else {
      return unknown_is_anything;
    }
  }
}
