// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/builder/ArrayBuilderOptions.h"

namespace awkward {
  ArrayBuilderOptions::ArrayBuilderOptions(int64_t initial, double resize,
    bool convert_nan_and_inf, bool replace_nan_and_inf)
      : initial_(initial)
      , resize_(resize)
      , convert_nan_and_inf_(convert_nan_and_inf)
      , replace_nan_and_inf_(replace_nan_and_inf) { }

  int64_t
  ArrayBuilderOptions::initial() const {
    return initial_;
  }

  double
  ArrayBuilderOptions::resize() const {
    return resize_;
  }
}
