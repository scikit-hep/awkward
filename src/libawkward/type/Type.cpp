// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/Type.h"

namespace awkward {
  const std::string Type::compare(std::shared_ptr<Type> supertype) {
    return tostring() + std::string(" vs ") + supertype.get()->tostring();
  }
}
