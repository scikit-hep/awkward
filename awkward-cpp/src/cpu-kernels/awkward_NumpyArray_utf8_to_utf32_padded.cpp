// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_utf8_to_utf32_padded.cpp", line)

#include "awkward/kernels.h"
#include "awkward/unicode.h"


template <typename T>
ERROR awkward_NumpyArray_utf8_to_utf32_padded(
  const uint8_t* __restrict__ fromptr,
  const T* __restrict__ fromoffsets,
  int64_t offsetslength,
  int64_t maxcodepoints,
  uint32_t* __restrict__ toptr) {

  int64_t n_code_point = 0;

  // For each sublist of code units
  for (int64_t k_sublist = 0;  k_sublist < offsetslength - 1;  k_sublist++) {
    // Anchor each sublist at its own offset, so that a malformed sublist
    // cannot shift the ones that follow it
    int64_t i_code_unit = (int64_t)fromoffsets[k_sublist];
    int64_t j_code_unit_last = (int64_t)fromoffsets[k_sublist + 1];
    int64_t n_code_point_sublist = 0;

    // Repeat until we exhaust the code units within this sublist
    while (i_code_unit < j_code_unit_last) {
      // Parse a single codepoint
      int64_t code_point_width = (int64_t)utf8_codepoint_size(fromptr[i_code_unit]);

      // A sequence that runs past the end of its sublist would read into the
      // next string, or past the end of the buffer entirely. Checked before
      // the decode below, which reads up to `code_point_width` bytes.
      if (code_point_width != 0  &&  i_code_unit + code_point_width > j_code_unit_last) {
        return failure("could not convert UTF8 code point to UTF32: truncated UTF8 sequence", kSliceNone, fromptr[i_code_unit], FILENAME(__LINE__));
      }
      // More code points than the buffer was sized for
      if (n_code_point_sublist >= maxcodepoints) {
        return failure("could not convert UTF8 code point to UTF32: string is longer than maxcodepoints", kSliceNone, n_code_point_sublist, FILENAME(__LINE__));
      }

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
    for (int64_t j = 0;  j < n_pad_code_points;  j++) {
      toptr[n_code_point++] = 0;
    }
  }

  return success();
}

#define WRAPPER(FUNC, T) \
  ERROR FUNC(const uint8_t *fromptr, const T *fromoffsets, int64_t offsetslength, int64_t maxcodepoints, uint32_t *toptr) { \
    return awkward_NumpyArray_utf8_to_utf32_padded<T>(fromptr, fromoffsets, offsetslength, maxcodepoints, toptr); \
  }

WRAPPER(awkward_NumpyArray_utf8_to_utf32_padded_int32, int32_t)
WRAPPER(awkward_NumpyArray_utf8_to_utf32_padded_uint32, uint32_t)
WRAPPER(awkward_NumpyArray_utf8_to_utf32_padded_int64, int64_t)
