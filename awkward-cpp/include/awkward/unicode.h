// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#include <cstddef>
#include <cstdint>

#ifndef AWKWARD_UNICODE_H_
#define AWKWARD_UNICODE_H_


#define UTF8_ONE_BYTE_MASK 0x80
#define UTF8_ONE_BYTE_BITS 0
#define UTF8_TWO_BYTES_MASK 0xE0
#define UTF8_TWO_BYTES_BITS 0xC0
#define UTF8_THREE_BYTES_MASK 0xF0
#define UTF8_THREE_BYTES_BITS 0xE0
#define UTF8_FOUR_BYTES_MASK 0xF8
#define UTF8_FOUR_BYTES_BITS 0xF0
#define UTF8_CONTINUATION_MASK 0xC0
#define UTF8_CONTINUATION_BITS 0x80


size_t utf8_codepoint_size(const uint8_t byte);

#endif // AWKWARD_UNICODE_H_
