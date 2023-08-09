// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_prepare_utf8_to_utf32_padded.cpp", line)

#include "awkward/kernels.h"
#include "awkward/unicode.h"


ERROR awkward_NumpyArray_prepare_utf8_to_utf32_padded(
    const uint8_t* fromptr,
    const int64_t* fromoffsets,
    int64_t offsetslength,
    int64_t* outmaxcodepoints) {

    *outmaxcodepoints = 0;
	int64_t i = fromoffsets[0];
    int64_t cp_size;

    // For each sublist of code units
    for (auto k = 0; k < offsetslength-1; k++) {
        auto n_code_units = fromoffsets[k+1] - fromoffsets[k];
        auto n_sublist_code_points = 0;

        // Parse one code point at a time, until we exhaust the code units
        for (auto i_last=i+n_code_units; i<i_last;) {
            cp_size = utf8_codepoint_size(fromptr[i]);
            i += cp_size;
            n_sublist_code_points += 1;
        }

        // Set largest substring length (in code points)
        *outmaxcodepoints = (*outmaxcodepoints < n_sublist_code_points) ? n_sublist_code_points : *outmaxcodepoints;
    }

    return success();
}
