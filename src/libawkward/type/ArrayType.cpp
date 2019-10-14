// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/ArrayType.h"

namespace awkward {
  std::string ArrayType::tostring_part(std::string indent, std::string pre, std::string post) const {
    return indent + pre + std::to_string(length_) + " * " + type_.get()->tostring_part(indent, "", "") + post;
  }
}
