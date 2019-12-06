// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/type/Type.h"

namespace awkward {
  std::shared_ptr<Type> Type::nolength() const {
    return shallow_copy();
  }

  const std::string Type::compare(std::shared_ptr<Type> supertype) {
    // FIXME: better side-by-side comparison
    return tostring() + std::string(" versus ") + supertype.get()->tostring();
  }
}
