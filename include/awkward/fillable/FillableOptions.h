// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLEOPTIONS_H_
#define AWKWARD_FILLABLEOPTIONS_H_

#include <cmath>
#include <cstring>

#include "awkward/cpu-kernels/util.h"

namespace awkward {
  class EXPORT_SYMBOL FillableOptions {
  public:
    FillableOptions(int64_t initial, double resize);
    int64_t initial() const;
    double resize() const;

  private:
    int64_t initial_;
    double resize_;
  };
}

#endif // AWKWARD_FILLABLEOPTIONS_H_
