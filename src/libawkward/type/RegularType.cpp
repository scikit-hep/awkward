// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/RegularType.h"

namespace awkward {
  std::string RegularType::tostring_part(std::string indent, std::string pre, std::string post) const {
    return indent + pre + std::to_string(size_) + std::string(" * ") + type_.get()->tostring_part(indent, "", "") + post;
  }

  const std::shared_ptr<Type> RegularType::shallow_copy() const {
    return std::shared_ptr<Type>(new RegularType(type_, size_));
  }

  bool RegularType::equal(std::shared_ptr<Type> other) const {
    if (RegularType* t = dynamic_cast<RegularType*>(other.get())) {
      return size() == t->size()  &&  type().get()->equal(t->type());
    }
    else {
      return false;
    }
  }

  bool RegularType::compatible(std::shared_ptr<Type> other, bool bool_is_int, bool int_is_float, bool ignore_null, bool unknown_is_anything) const {
    if (unknown_is_anything  &&  dynamic_cast<UnknownType*>(other.get())) {
      return true;
    }
    else if (ignore_null  &&  dynamic_cast<OptionType*>(other.get())) {
      return compatible(dynamic_cast<OptionType*>(other.get())->type(), bool_is_int, int_is_float, ignore_null, unknown_is_anything);
    }
    else if (RegularType* t = dynamic_cast<RegularType*>(other.get())) {
      return size() == t->size()  &&  type_.get()->compatible(t->type(), bool_is_int, int_is_float, ignore_null, unknown_is_anything);
    }
    else {
      return false;
    }
  }

  const std::shared_ptr<Type> RegularType::type() const {
    return type_;
  }

  int64_t RegularType::size() const {
    return size_;
  }
}
