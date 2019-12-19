// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/fillable/FillableOptions.h"

namespace awkward {
  FillableOptions::FillableOptions(int64_t initial, double resize)
      : initial_(initial)
      , resize_(resize) { }

  int64_t FillableOptions::initial() const {
    return initial_;
  }

  double FillableOptions::resize() const {
    return resize_;
  }
}
