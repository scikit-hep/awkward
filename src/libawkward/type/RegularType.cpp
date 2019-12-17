// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/RegularType.h"

namespace awkward {
  std::string RegularType::tostring_part(std::string indent, std::string pre, std::string post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return typestr;
    }

    std::stringstream out;
    if (parameters_.size() == 0) {
      out << indent << pre << size_ << " * " << type_.get()->tostring_part(indent, "", "") << post;
    }
    else {
      out << indent << pre << "[" << size_ << " * " << type_.get()->tostring_part(indent, "", "") << ", " << string_parameters() << "]" << post;
    }
    return out.str();
  }

  const std::shared_ptr<Type> RegularType::shallow_copy() const {
    return std::shared_ptr<Type>(new RegularType(parameters_, type_, size_));
  }

  bool RegularType::equal(const std::shared_ptr<Type> other, bool check_parameters) const {
    if (RegularType* t = dynamic_cast<RegularType*>(other.get())) {
      if (check_parameters  &&  !equal_parameters(other.get()->parameters())) {
        return false;
      }
      return size() == t->size()  &&  type().get()->equal(t->type(), check_parameters);
    }
    else {
      return false;
    }
  }

  std::shared_ptr<Type> RegularType::level() const {
    return shallow_copy();
  }

  int64_t RegularType::numfields() const {
    return type_.get()->numfields();
  }

  int64_t RegularType::fieldindex(const std::string& key) const {
    return type_.get()->fieldindex(key);
  }

  const std::string RegularType::key(int64_t fieldindex) const {
    return type_.get()->key(fieldindex);
  }

  bool RegularType::haskey(const std::string& key) const {
    return type_.get()->haskey(key);
  }

  const std::vector<std::string> RegularType::keyaliases(int64_t fieldindex) const {
    return type_.get()->keyaliases(fieldindex);
  }

  const std::vector<std::string> RegularType::keyaliases(const std::string& key) const {
    return type_.get()->keyaliases(key);
  }

  const std::vector<std::string> RegularType::keys() const {
    return type_.get()->keys();
  }

  const std::shared_ptr<Type> RegularType::type() const {
    return type_;
  }

  int64_t RegularType::size() const {
    return size_;
  }
}
