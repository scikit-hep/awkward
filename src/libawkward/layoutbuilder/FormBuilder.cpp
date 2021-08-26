// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/FormBuilder.cpp", line)

#include "awkward/layoutbuilder/FormBuilder.h"

namespace awkward {

  template <typename T, typename I>
  FormBuilder<T, I>::~FormBuilder() = default;

  template class EXPORT_TEMPLATE_INST FormBuilder<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST FormBuilder<int64_t, int32_t>;

}
