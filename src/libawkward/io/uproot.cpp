// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/io/root.cpp", line)

#include <cstring>

#include "awkward/Index.h"
#include "awkward/Content.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/builder/GrowableBuffer.h"

#include "awkward/io/uproot.h"

namespace awkward {
  const ContentPtr
    uproot_issue_90(const NumpyArray& data,
                    const Index32& byte_offsets,
                    const Form& form) {
    return std::make_shared<EmptyArray>(Identities::none(), util::Parameters());
  }
}
