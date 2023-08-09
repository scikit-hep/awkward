// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_utf8_to_utf32_padded.cpp", line)

#include "awkward/kernels.h"
#include "awkward/unicode.h"


ERROR awkward_NumpyArray_utf8_to_utf32_padded(
    const uint8_t* fromptr,
    const int64_t* fromoffsets,
    int64_t offsetslength,
    int64_t maxcodepoints,
    uint32_t* toptr) {

	int64_t i = fromoffsets[0];
	int64_t cp_size;
    int64_t n = 0;

    // For each sublist of code units
	for (auto k = 0; k < offsetslength-1; k++) {
	    auto n_code_units = fromoffsets[k+1] - fromoffsets[k];
	    int64_t n_sublist_code_points = 0;

        // Parse one code point at a time, until we exhaust the code units
        for (auto i_last=i+n_code_units; i<i_last;) {
            cp_size = utf8_codepoint_size(fromptr[i]);

            switch (cp_size) {
            case 1:
                toptr[n] = ((uint32_t) fromptr[i] & ~UTF8_ONE_BYTE_MASK);
                break;
            case 2:
                toptr[n] =
                    ((uint32_t) fromptr[i] & ~UTF8_TWO_BYTES_MASK) << 6 |
                    ((uint32_t) fromptr[i + 1] & ~UTF8_CONTINUATION_MASK)
                ;
                break;
            case 3:
                toptr[n] =
                    ((uint32_t) fromptr[i] & ~UTF8_THREE_BYTES_MASK) << 12 |
                    ((uint32_t) fromptr[i + 1] & ~UTF8_CONTINUATION_MASK) << 6 |
                    ((uint32_t) fromptr[i + 2] & ~UTF8_CONTINUATION_MASK)
                ;

                break;
            case 4:
                toptr[n] =
                    ((uint32_t) fromptr[i] & ~UTF8_FOUR_BYTES_MASK) << 18 |
                    ((uint32_t) fromptr[i + 1] & ~UTF8_CONTINUATION_MASK) << 12 |
                    ((uint32_t) fromptr[i + 2] & ~UTF8_CONTINUATION_MASK) << 6 |
                    ((uint32_t) fromptr[i + 3] & ~UTF8_CONTINUATION_MASK)
                ;
                break;
            default:
                return failure( "utf8_to_utf32: invalid byte in UTF8 string", kSliceNone, fromptr[i], FILENAME(__LINE__));
            }

            n++;
            i += cp_size;
            n_sublist_code_points += 1;
        }

        // Zero pad the remaining characters
        int64_t n_pad_code_points = maxcodepoints-n_sublist_code_points;
        for (auto j=0; j<n_pad_code_points; j++){
            toptr[n++] = 0;
        }
	}

    return success();
}
