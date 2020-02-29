// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IO_ROOT_H_
#define AWKWARD_IO_ROOT_H_

#include <cstdio>
#include <string>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"
#include "awkward/Index.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/Content.h"
#include "awkward/array/NumpyArray.h"

namespace awkward {
  EXPORT_SYMBOL const std::shared_ptr<Content> FromROOT_nestedvector(const Index64& byteoffsets, const NumpyArray& rawdata, int64_t depth, int64_t itemsize, std::string format, const FillableOptions& options);
}

#endif // AWKWARD_IO_ROOT_H_
