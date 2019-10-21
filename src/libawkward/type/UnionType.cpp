// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "awkward/type/UnionType.h"

namespace awkward {
  std::string UnionType::tostring_part(std::string indent, std::string pre, std::string post) const {
    std::stringstream out;
    out << indent << pre << "union[";
    for (size_t i = 0;  i < numtypes();  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << type(i).get()->tostring_part(indent, "", "");
    }
    out << "]" << post;
    return out.str();
  }

  const std::shared_ptr<Type> UnionType::shallow_copy() const {
    return std::shared_ptr<Type>(new UnionType(types_));
  }

  bool UnionType::equal(std::shared_ptr<Type> other) const {
    if (UnionType* t = dynamic_cast<UnionType*>(other.get())) {
      if (numtypes() != t->numtypes()) {
        return false;
      }
      for (size_t i = 0;  i < numtypes();  i++) {
        if (!type(i).get()->equal(t->type(i))) {
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

  const std::vector<std::shared_ptr<Type>> UnionType::types() const {
    return types_;
  }

  const std::shared_ptr<Type> UnionType::type(int64_t i) const {
    return types_[(size_t)i];
  }
}
