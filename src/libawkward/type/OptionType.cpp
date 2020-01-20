// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "awkward/type/UnknownType.h"
#include "awkward/type/ListType.h"
#include "awkward/type/RegularType.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/Index.h"

#include "awkward/type/OptionType.h"

namespace awkward {
  OptionType::OptionType(const util::Parameters& parameters, const std::shared_ptr<Type>& type)
      : Type(parameters)
      , type_(type) { }

  std::string OptionType::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return typestr;
    }

    std::stringstream out;
    if (parameters_.empty()) {
      if (dynamic_cast<ListType*>(type_.get()) != nullptr  ||
          dynamic_cast<RegularType*>(type_.get()) != nullptr) {
        out << indent << pre << "option[" << type_.get()->tostring_part(indent, "", "") << "]" << post;
      }
      else {
        out << indent << pre << "?" << type_.get()->tostring_part("", "", "") << post;
      }
    }
    else {
      out << indent << pre << "option[" << type_.get()->tostring_part(indent, "", "") << ", " << string_parameters() << "]" << post;
    }
    return out.str();
  }

  const std::shared_ptr<Type> OptionType::shallow_copy() const {
    return std::make_shared<OptionType>(parameters_, type_);
  }

  bool OptionType::equal(const std::shared_ptr<Type>& other, bool check_parameters) const {
    if (OptionType* t = dynamic_cast<OptionType*>(other.get())) {
      if (check_parameters  &&  !parameters_equal(other.get()->parameters())) {
        return false;
      }
      return type().get()->equal(t->type(), check_parameters);
    }
    else {
      return false;
    }
  }

  int64_t OptionType::numfields() const {
    return type_.get()->numfields();
  }

  int64_t OptionType::fieldindex(const std::string& key) const {
    return type_.get()->fieldindex(key);
  }

  const std::string OptionType::key(int64_t fieldindex) const {
    return type_.get()->key(fieldindex);
  }

  bool OptionType::haskey(const std::string& key) const {
    return type_.get()->haskey(key);
  }

  const std::vector<std::string> OptionType::keys() const {
    return type_.get()->keys();
  }

  const std::shared_ptr<Content> OptionType::empty() const {
    std::shared_ptr<Content> content = type_.get()->empty();
    Index64 index(0);
    return std::make_shared<IndexedOptionArray64>(Identities::none(), parameters_, index, content);
  }

  const std::shared_ptr<Type> OptionType::type() const {
    std::shared_ptr<Type> out = type_;
    while (OptionType* t = dynamic_cast<OptionType*>(out.get())) {
      out = t->type_;
    }
    return out;
  }
}
