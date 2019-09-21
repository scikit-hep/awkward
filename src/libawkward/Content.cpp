// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Content.h"

using namespace awkward;

const std::string Content::tostring() const {
  return tostring_part("", "", "");
}
