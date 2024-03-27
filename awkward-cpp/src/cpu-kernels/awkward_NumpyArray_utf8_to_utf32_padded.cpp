// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_utf8_to_utf32_padded.cpp", line)

#include "awkward/kernels.h"
#include "awkward/unicode.h"


ERROR awkward_NumpyArray_utf8_to_utf32_padded(
  const uint8_t *fromptr,
  const int64_t *fromoffsets,
  int64_t offsetslength,
  int64_t maxcodepoints,
  uint32_t *toptr) {

  int64_t i_code_unit = fromoffsets[0];
  int64_t code_point_width;
  int64_t n_code_point = 0;

  // For each sublist of code units
  for (auto k_sublist = 0; k_sublist < offsetslength - 1; k_sublist++) {
    auto n_code_units = fromoffsets[k_sublist + 1] - fromoffsets[k_sublist];
    int64_t n_code_point_sublist = 0;

    // Repeat until we exhaust the code units within this sublist
    for (auto j_code_unit_last = i_code_unit + n_code_units; i_code_unit < j_code_unit_last;) {
      // Parse a single codepoint
      code_point_width = utf8_codepoint_size(fromptr[i_code_unit]);
      switch (code_point_width) {
      case 1:
        toptr[n_code_point] = ((uint32_t) fromptr[i_code_unit] & ~UTF8_ONE_BYTE_MASK);
        break;
      case 2:
        toptr[n_code_point] =
          ((uint32_t) fromptr[i_code_unit] & ~UTF8_TWO_BYTES_MASK) << 6 |
          ((uint32_t) fromptr[i_code_unit + 1] & ~UTF8_CONTINUATION_MASK);
        break;
      case 3:
        toptr[n_code_point] =
          ((uint32_t) fromptr[i_code_unit] & ~UTF8_THREE_BYTES_MASK) << 12 |
          ((uint32_t) fromptr[i_code_unit + 1] & ~UTF8_CONTINUATION_MASK) << 6 |
          ((uint32_t) fromptr[i_code_unit + 2] & ~UTF8_CONTINUATION_MASK);

        break;
      case 4:
        toptr[n_code_point] =
          ((uint32_t) fromptr[i_code_unit] & ~UTF8_FOUR_BYTES_MASK) << 18 |
          ((uint32_t) fromptr[i_code_unit + 1] & ~UTF8_CONTINUATION_MASK) << 12 |
          ((uint32_t) fromptr[i_code_unit + 2] & ~UTF8_CONTINUATION_MASK) << 6 |
          ((uint32_t) fromptr[i_code_unit + 3] & ~UTF8_CONTINUATION_MASK);
        break;
      default:
        return failure("could not convert UTF8 code point to UTF32: invalid byte in UTF8 string", kSliceNone, fromptr[i_code_unit], FILENAME(__LINE__));
      }
      // Increment the code-point counter
      n_code_point++;

      // Shift the code-unit start index
      i_code_unit += code_point_width;

      // Increment the code-point counter for this sublist
      n_code_point_sublist += 1;
    }

    // Zero pad the remaining characters
    int64_t n_pad_code_points = maxcodepoints - n_code_point_sublist;
    for (auto j = 0; j < n_pad_code_points; j++) {
      toptr[n_code_point++] = 0;
    }
  }

  return success();
}
