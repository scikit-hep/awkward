// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/unicode.cpp", line)

#include "awkward/unicode.h"


size_t utf8_codepoint_size(const uint8_t byte) {
	if ((byte & UTF8_ONE_BYTE_MASK) == UTF8_ONE_BYTE_BITS) {
		return 1;
	}

	if ((byte & UTF8_TWO_BYTES_MASK) == UTF8_TWO_BYTES_BITS) {
		return 2;
	}

	if ((byte & UTF8_THREE_BYTES_MASK) == UTF8_THREE_BYTES_BITS) {
		return 3;
	}

	if ((byte & UTF8_FOUR_BYTES_MASK) == UTF8_FOUR_BYTES_BITS) {
		return 4;
	}

	return 0;
}
