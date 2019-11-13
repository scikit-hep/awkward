// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/UnionType.h"

namespace awkward {
  std::string UnionType::tostring_part(std::string indent, std::string pre, std::string post) const {
    std::stringstream out;
    out << indent << pre << "union[";
    for (int64_t i = 0;  i < numtypes();  i++) {
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

  bool UnionType::compatible(std::shared_ptr<Type> other) const {
    if (UnknownType* t = dynamic_cast<UnknownType*>(other.get())) {
      return true;
    }
    else if (OptionType* t = dynamic_cast<OptionType*>(other.get())) {
      return compatible(t->type());
    }
    else if (UnionType* t = dynamic_cast<UnionType*>(other.get())) {
      for (auto me : types_) {
        bool any = false;
        for (auto you : t->types_) {
          if (me.get()->compatible(you)) {
            any = true;
            break;
          }
        }
        if (!any) {
          return false;
        }
      }
      for (auto you : t->types_) {
        bool any = false;
        for (auto me : types_) {
          if (you.get()->compatible(me)) {
            any = true;
            break;
          }
        }
        if (!any) {
          return false;
        }
      }
      return true;
    }
    else {
      for (auto me : types_) {
        if (!me.get()->compatible(other)) {
          return false;
        }
      }
      return true;
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
