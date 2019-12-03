// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/UnknownType.h"
#include "awkward/type/ListType.h"
#include "awkward/type/RegularType.h"

#include "awkward/type/OptionType.h"

namespace awkward {
  std::string OptionType::tostring_part(std::string indent, std::string pre, std::string post) const {
    if (dynamic_cast<ListType*>(type_.get()) != nullptr  ||
        dynamic_cast<RegularType*>(type_.get()) != nullptr) {
      return indent + pre + "option[" + type().get()->tostring_part(indent, "", "") + "]" + post;
    }
    else {
      return indent + pre + "?" + type_.get()->tostring_part("", "", "") + post;
    }
  }

  const std::shared_ptr<Type> OptionType::shallow_copy() const {
    return std::shared_ptr<Type>(new OptionType(type_));
  }

  bool OptionType::equal(std::shared_ptr<Type> other) const {
    if (OptionType* t = dynamic_cast<OptionType*>(other.get())) {
      return type().get()->equal(t->type());
    }
    else {
      return false;
    }
  }

  const std::shared_ptr<Type> OptionType::type() const {
    std::shared_ptr<Type> out = type_;
    while (OptionType* t = dynamic_cast<OptionType*>(out.get())) {
      out = t->type_;
    }
    return out;
  }
}
