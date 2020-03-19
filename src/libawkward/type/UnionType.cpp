// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"
#include "awkward/array/UnionArray.h"

#include "awkward/type/UnionType.h"

namespace awkward {
  UnionType::UnionType(const util::Parameters& parameters, const std::string& typestr, const std::vector<std::shared_ptr<Type>>& types)
      : Type(parameters, typestr)
      , types_(types) { }

  std::string UnionType::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return typestr;
    }

    std::stringstream out;
    out << indent << pre << "union[";
    for (int64_t i = 0;  i < numtypes();  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << type(i).get()->tostring_part(indent, "", "");
    }
    if (!parameters_.empty()) {
      out << ", " << string_parameters();
    }
    out << "]" << post;
    return out.str();
  }

  const std::shared_ptr<Type> UnionType::shallow_copy() const {
    return std::make_shared<UnionType>(parameters_, typestr_, types_);
  }

  bool UnionType::equal(const std::shared_ptr<Type>& other, bool check_parameters) const {
    if (UnionType* t = dynamic_cast<UnionType*>(other.get())) {
      if (check_parameters  &&  !parameters_equal(other.get()->parameters())) {
        return false;
      }
      if (types_.size() != t->types_.size()) {
        return false;
      }
      for (size_t i = 0;  i < types_.size();  i++) {
        if (!types_[i].get()->equal(t->types_[i], check_parameters)) {
          return false;
        }
      }
      return true;
    }
    else {
      return false;
    }
  }

  int64_t UnionType::numtypes() const {
    return (int64_t)types_.size();
  }

  int64_t UnionType::numfields() const {
    throw std::runtime_error("FIXME: UnionType::numfields");
  }

  int64_t UnionType::fieldindex(const std::string& key) const {
    throw std::runtime_error("FIXME: UnionType::fieldindex(key)");
  }

  const std::string UnionType::key(int64_t fieldindex) const {
    throw std::runtime_error("FIXME: UnionType::key(fieldindex)");
  }

  bool UnionType::haskey(const std::string& key) const {
    throw std::runtime_error("FIXME: UnionType::haskey(key)");
  }

  const std::vector<std::string> UnionType::keys() const {
    throw std::runtime_error("FIXME: UnionType::keys");
  }

  const std::vector<std::shared_ptr<Type>> UnionType::types() const {
    return types_;
  }

  ContentPtr UnionType::empty() const {
    std::vector<std::shared_ptr<Content>> contents;
    for (auto type : types_) {
      contents.push_back(type.get()->empty());
    }
    Index8 tags(0);
    Index64 index(0);
    return std::make_shared<UnionArray8_64>(Identities::none(), parameters_, tags, index, contents);
  }

  const std::shared_ptr<Type> UnionType::type(int64_t index) const {
    return types_[(size_t)index];
  }
}
