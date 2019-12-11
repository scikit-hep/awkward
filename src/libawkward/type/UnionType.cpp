// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/UnionType.h"

namespace awkward {
  std::string UnionType::tostring_part(std::string indent, std::string pre, std::string post) const {
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
    if (parameters_FIXME_.size() != 0) {
      out << ", " << string_parameters();
    }
    out << "]" << post;
    return out.str();
  }

  const std::shared_ptr<Type> UnionType::shallow_copy() const {
    return std::shared_ptr<Type>(new UnionType(parameters_FIXME_, types_));
  }

  bool UnionType::equal(const std::shared_ptr<Type> other, bool check_parameters) const {
    if (UnionType* t = dynamic_cast<UnionType*>(other.get())) {
      if (check_parameters  &&  !equal_parameters(other.get()->parameters())) {
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

  std::shared_ptr<Type> UnionType::level() const {
    std::vector<std::shared_ptr<Type>> types;
    for (auto t : types_) {
      types.push_back(t.get()->level());
    }
    return std::shared_ptr<Type>(new UnionType(Type::Parameters(), types));
  }

  std::shared_ptr<Type> UnionType::inner() const {
    throw std::runtime_error("FIXME: UnionType::inner()");
  }

  std::shared_ptr<Type> UnionType::inner(const std::string& key) const {
    throw std::runtime_error("FIXME: UnionType::inner(key)");
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

  const std::vector<std::string> UnionType::keyaliases(int64_t fieldindex) const {
    throw std::runtime_error("FIXME: UnionType::keyaliases(fieldindex)");
  }

  const std::vector<std::string> UnionType::keyaliases(const std::string& key) const {
    throw std::runtime_error("FIXME: UnionType::keyaliases(key)");
  }

  const std::vector<std::string> UnionType::keys() const {
    throw std::runtime_error("FIXME: UnionType::keys");
  }

  const std::vector<std::shared_ptr<Type>> UnionType::types() const {
    return types_;
  }

  const std::shared_ptr<Type> UnionType::type(int64_t index) const {
    return types_[(size_t)index];
  }
}
