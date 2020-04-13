// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/builder/Builder.h"

namespace awkward {
  void
  Builder::setthat(const BuilderPtr& that) {
    that_ = that;
  }
}
