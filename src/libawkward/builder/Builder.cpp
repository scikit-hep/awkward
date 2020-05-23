// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/builder/Builder.h"

namespace awkward {
  Builder::~Builder() = default;

  void
  Builder::setthat(const BuilderPtr& that) {
    that_ = that;
  }
}
