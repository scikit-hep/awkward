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
    return std::shared_ptr<Type>(new ListType(parameters_FIXME_, type_));
  }

  bool ListType::shallow_equal(const std::shared_ptr<Type> other, bool check_parameters) const {
    if (dynamic_cast<ListType*>(other.get()) != nullptr) {
      if (check_parameters  &&  !equal_parameters(other.get()->parameters())) {
        return false;
      }
      return true;
    }
    else {
      return false;
    }
  }

  bool ListType::equal(const std::shared_ptr<Type> other, bool check_parameters) const {
    if (ListType* t = dynamic_cast<ListType*>(other.get())) {
      if (check_parameters  &&  !equal_parameters(other.get()->parameters())) {
        return false;
      }
      return type().get()->equal(t->type(), check_parameters);
    }
    else {
      return false;
    }
  }

  std::shared_ptr<Type> ListType::level() const {
    return shallow_copy();
  }

  std::shared_ptr<Type> ListType::inner() const {
    return type_;
  }

  std::shared_ptr<Type> ListType::inner(const std::string& key) const {
    throw std::runtime_error("FIXME: ListType::inner(key)");
  }

  int64_t ListType::numfields() const {
    return type_.get()->numfields();
  }

  int64_t ListType::fieldindex(const std::string& key) const {
    return type_.get()->fieldindex(key);
  }

  const std::string ListType::key(int64_t fieldindex) const {
    return type_.get()->key(fieldindex);
  }

  bool ListType::haskey(const std::string& key) const {
    return type_.get()->haskey(key);
  }

  const std::vector<std::string> ListType::keyaliases(int64_t fieldindex) const {
    return type_.get()->keyaliases(fieldindex);
  }

  const std::vector<std::string> ListType::keyaliases(const std::string& key) const {
    return type_.get()->keyaliases(key);
  }

  const std::vector<std::string> ListType::keys() const {
    return type_.get()->keys();
  }

  const std::shared_ptr<Type> ListType::type() const {
    return type_;
  }
}
