// BSD 3-Clause License; see
// https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <string>

#include "awkward/type/UnknownType.h"

namespace awkward {
std::string UnknownType::tostring_part(std::string indent, std::string pre,
                                       std::string post) const {
  std::string typestr;
  if (get_typestr(typestr)) {
    return typestr;
  }

  std::stringstream out;
  if (parameters_.size() == 0) {
    out << indent << pre << "unknown" << post;
  } else {
    out << indent << pre << "unknown[" << string_parameters() << "]" << post;
  }
  return out.str();
}

const std::shared_ptr<Type> UnknownType::shallow_copy() const {
  return std::shared_ptr<Type>(new UnknownType(parameters_));
}

bool UnknownType::equal(const std::shared_ptr<Type> other,
                        bool check_parameters) const {
  if (UnknownType *t = dynamic_cast<UnknownType *>(other.get())) {
    if (check_parameters && !equal_parameters(other.get()->parameters())) {
      return false;
    }
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<Type> UnknownType::level() const { return shallow_copy(); }

std::shared_ptr<Type> UnknownType::inner() const { return shallow_copy(); }

std::shared_ptr<Type> UnknownType::inner(const std::string &key) const {
  return shallow_copy();
}

int64_t UnknownType::numfields() const { return -1; }

int64_t UnknownType::fieldindex(const std::string &key) const {
  throw std::invalid_argument("type contains no Records");
}

const std::string UnknownType::key(int64_t fieldindex) const {
  throw std::invalid_argument("type contains no Records");
}

bool UnknownType::haskey(const std::string &key) const {
  throw std::invalid_argument("type contains no Records");
}

const std::vector<std::string>
UnknownType::keyaliases(int64_t fieldindex) const {
  throw std::invalid_argument("type contains no Records");
}

const std::vector<std::string>
UnknownType::keyaliases(const std::string &key) const {
  throw std::invalid_argument("type contains no Records");
}

const std::vector<std::string> UnknownType::keys() const {
  throw std::invalid_argument("type contains no Records");
}
} // namespace awkward
