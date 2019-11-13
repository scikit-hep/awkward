// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/ListType.h"

namespace awkward {
  std::string ListType::tostring_part(std::string indent, std::string pre, std::string post) const {
    return indent + pre + "var * " + type().get()->tostring_part(indent, "", "") + post;
  }

  const std::shared_ptr<Type> ListType::shallow_copy() const {
    return std::shared_ptr<Type>(new ListType(type_));
  }

  bool ListType::equal(std::shared_ptr<Type> other) const {
    if (ListType* t = dynamic_cast<ListType*>(other.get())) {
      return type().get()->equal(t->type());
    }
    else {
      return false;
    }
  }

  bool ListType::compatible(std::shared_ptr<Type> other, bool bool_is_int, bool int_is_float, bool ignore_null, bool unknown_is_anything) const {
    if (unknown_is_anything  &&  dynamic_cast<UnknownType*>(other.get())) {
      return true;
    }
    else if (ignore_null  &&  dynamic_cast<OptionType*>(other.get())) {
      return compatible(dynamic_cast<OptionType*>(other.get())->type(), bool_is_int, int_is_float, ignore_null, unknown_is_anything);
    }
    else if (ListType* t = dynamic_cast<ListType*>(other.get())) {
      return type_.get()->compatible(t->type(), bool_is_int, int_is_float, ignore_null, unknown_is_anything);
    }
    else {
      return false;
    }
  }

  const std::shared_ptr<Type> ListType::type() const {
    return type_;
  }
}
