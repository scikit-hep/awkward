// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_prepare_utf8_to_utf32_padded.cpp", line)

#include "awkward/kernels.h"
#include "awkward/unicode.h"


ERROR awkward_NumpyArray_prepare_utf8_to_utf32_padded(
  const uint8_t *fromptr,
  const int64_t *fromoffsets,
  int64_t offsetslength,
  int64_t *outmaxcodepoints) {

  *outmaxcodepoints = 0;
  int64_t i_code_unit = fromoffsets[0];
  int64_t code_point_width;

  // For each sublist of code units
  for (auto k_sublist = 0; k_sublist < offsetslength - 1; k_sublist++) {
    auto n_code_units = fromoffsets[k_sublist + 1] - fromoffsets[k_sublist];
    auto n_code_point_sublist = 0;

    // Repeat until we exhaust the code units within this sublist
    for (auto j_code_unit_last = i_code_unit + n_code_units; i_code_unit < j_code_unit_last;) {
      code_point_width = utf8_codepoint_size(fromptr[i_code_unit]);

      // Shift the code-unit start index
      i_code_unit += code_point_width;

      // Increment the code-point counter for this sublist
      n_code_point_sublist += 1;
    }

    // Set largest substring length (in code points)
    *outmaxcodepoints = ( *outmaxcodepoints < n_code_point_sublist) ? n_code_point_sublist : *outmaxcodepoints;
  }

  return success();
}
