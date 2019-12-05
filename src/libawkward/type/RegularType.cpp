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

  std::shared_ptr<Type> RegularType::inner() const {
    return type_;
  }

  std::shared_ptr<Type> RegularType::inner(std::string key) const {
    throw std::runtime_error("FIXME: RegularType::inner(key)");
  }

  const std::shared_ptr<Type> RegularType::type() const {
    return type_;
  }

  int64_t RegularType::size() const {
    return size_;
  }
}
