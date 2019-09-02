// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"
#include "awkward/Content.h"

using namespace awkward;

const std::string Content::repr() const {
  return repr("", "", "");
}
