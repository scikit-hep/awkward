// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "awkward/array/RegularArray.h"
#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/RegularType.h"

namespace awkward {
  RegularType::RegularType(const util::Parameters& parameters, const std::string& typestr, const std::shared_ptr<Type>& type, int64_t size)
      : Type(parameters, typestr)
      , type_(type)
      , size_(size) { }

  std::string RegularType::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return typestr;
    }

    std::stringstream out;
    if (parameters_.empty()) {
      out << indent << pre << size_ << " * " << type_.get()->tostring_part(indent, "", "") << post;
    }
    else {
      out << indent << pre << "[" << size_ << " * " << type_.get()->tostring_part(indent, "", "") << ", " << string_parameters() << "]" << post;
    }
    return out.str();
  }

  const std::shared_ptr<Type> RegularType::shallow_copy() const {
    return std::make_shared<RegularType>(parameters_, typestr_, type_, size_);
  }

  bool RegularType::equal(const std::shared_ptr<Type>& other, bool check_parameters) const {
    if (RegularType* t = dynamic_cast<RegularType*>(other.get())) {
      if (check_parameters  &&  !parameters_equal(other.get()->parameters())) {
        return false;
      }
      return size() == t->size()  &&  type().get()->equal(t->type(), check_parameters);
    }
    else {
      return false;
    }
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

  const std::vector<std::string> RegularType::keys() const {
    return type_.get()->keys();
  }

  ContentPtr RegularType::empty() const {
    std::shared_ptr<Content> content = type_.get()->empty();
    return std::make_shared<RegularArray>(Identities::none(), parameters_, content, size_);
  }

  const std::shared_ptr<Type> RegularType::type() const {
    return type_;
  }

  int64_t RegularType::size() const {
    return size_;
  }
}
