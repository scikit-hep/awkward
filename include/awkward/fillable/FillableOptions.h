// BSD 3-Clause License; see
// https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLEOPTIONS_H_
#define AWKWARD_FILLABLEOPTIONS_H_

#include <cmath>
#include <cstring>

#include "awkward/cpu-kernels/util.h"

namespace awkward {
class FillableOptions {
public:
  FillableOptions(int64_t initial, double resize)
      : initial_(initial), resize_(resize) {}

  int64_t initial() const { return initial_; }
  double resize() const { return resize_; }

private:
  int64_t initial_;
  double resize_;
};
} // namespace awkward

#endif // AWKWARD_FILLABLEOPTIONS_H_
