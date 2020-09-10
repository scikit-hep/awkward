// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IO_UPROOT_H_
#define AWKWARD_IO_UPROOT_H_

#include "awkward/Index.h"
#include "awkward/Content.h"
#include "awkward/array/NumpyArray.h"

namespace awkward {
  LIBAWKWARD_EXPORT_SYMBOL const ContentPtr
    uproot_issue_90(const NumpyArray& data,
                    const Index32& byte_offsets,
                    const Form& form);
}

#endif // AWKWARD_IO_UPROOT_H_
