// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_DLPACK_UTIL_H_
#define AWKWARD_DLPACK_UTIL_H_

#include "dlpack/dlpack.h"
#include "awkward/util.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/util.h"

namespace ak = awkward;

DLDataType data_type_dispatch(ak::util::dtype dt) {
  switch (dt) {
    case ak::util::dtype::int8:
    return {kDLInt, 8, 1};
  case ak::util::dtype::int16:
    return {kDLInt, 16, 1};
  case ak::util::dtype::int32:
    return {kDLInt, 32, 1};
  case ak::util::dtype::int64:
    return {kDLInt, 64, 1};
  case ak::util::dtype::uint8:
    return {kDLUInt, 8, 1};
  case ak::util::dtype::uint16:
    return {kDLUInt, 16, 1};
  case ak::util::dtype::uint32:
    return {kDLUInt, 32, 1};
  case ak::util::dtype::uint64:
    return {kDLUInt, 64, 1};
  case ak::util::dtype::float16:
    return {kDLFloat, 16, 1};
  case ak::util::dtype::float32:
    return {kDLFloat, 32, 1};
  case ak::util::dtype::float64:
    return {kDLFloat, 64, 1};
  case ak::util::dtype::float128:
    return {kDLFloat, 128, 1};
  // case ak::util::dtype::datetime64:
  //   return 8;
  // case ak::util::dtype::timedelta64:
  //   return 8;
  }
}
#endif
