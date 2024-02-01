// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthInputBuffer.cpp", line)

#include "awkward/forth/ForthInputBuffer.h"

namespace awkward {
  ForthInputBuffer::ForthInputBuffer(const std::shared_ptr<void> ptr,
                                     int64_t offset,
                                     int64_t length)
    : ptr_(ptr)
    , offset_(offset)
    , length_(length)
    , pos_(0) { }

  uint8_t
  ForthInputBuffer::peek_byte(int64_t after, util::ForthError& err) noexcept {
    if (pos_ + after + 1 > length_) {
      err = util::ForthError::read_beyond;
      return 0;
    }
    return *reinterpret_cast<uint8_t*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_ + (size_t)pos_ + (size_t)after
    );
  }

  void*
  ForthInputBuffer::read(int64_t num_bytes, util::ForthError& err) noexcept {
    int64_t next = pos_ + num_bytes;
    if (next > length_) {
      err = util::ForthError::read_beyond;
      return nullptr;
    }
    int64_t tmp = pos_;
    pos_ = next;
    return reinterpret_cast<void*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_ + (size_t)tmp
    );
  }

  uint8_t
  ForthInputBuffer::read_byte(util::ForthError& err) noexcept {
    if (pos_ + 1 > length_) {
      err = util::ForthError::read_beyond;
      return 0;
    }
    int64_t tmp = pos_;
    pos_++;
    return *reinterpret_cast<uint8_t*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_ + (size_t)tmp
    );
  }

  int64_t
  ForthInputBuffer::read_enum(const std::vector<std::string>& strings, int64_t start, int64_t stop) noexcept {
    // Ideally, we'd use a string trie instead of repeatedly looping over the same set of strings.
    // However, C++ doesn't seem to have one in its standard library, and anyway, we don't
    // care about insertion/removal, so it can be compact (small array indexes, not pointers).

    if (pos_ >= length_) {
      return -1;
    }

    const char* ptr = reinterpret_cast<char*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_ + (size_t)pos_
    );

    int64_t i = 0;
    int64_t howmany = stop - start;
    int64_t len;

    for (auto it = strings.begin() + start;  i < howmany;  i++) {
      len = (int64_t)it->length();
      if (pos_ + len <= length_) {
        if (strncmp(it->data(), ptr, (size_t)len) == 0) {
          pos_ += len;
          return i;
        }
      }
      ++it;
    }

    return -1;
  }

  uint64_t
  ForthInputBuffer::read_varint(util::ForthError& err) noexcept {
    const uint8_t* ptr = reinterpret_cast<uint8_t*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_
    );

    int64_t shift = 0;
    uint64_t result = 0;
    uint8_t byte;
    do {
      if (pos_ >= length_) {
        err = util::ForthError::read_beyond;
        return 0;
      }
      byte = ptr[pos_];
      pos_++;

      if (shift == 7 * 9) {
        err = util::ForthError::varint_too_big;
        return 0;
      }

      result |= (uint64_t)(byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);

    return result;
  }

  int64_t
  ForthInputBuffer::read_zigzag(util::ForthError& err) noexcept {
    const uint8_t* ptr = reinterpret_cast<uint8_t*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_
    );

    int64_t shift = 0;
    int64_t result = 0;
    uint8_t byte;
    do {
      if (pos_ >= length_) {
        err = util::ForthError::read_beyond;
        return 0;
      }
      byte = ptr[pos_];
      pos_++;

      if (shift == 7 * 9) {
        err = util::ForthError::varint_too_big;
        return 0;
      }

      result |= (int64_t)(byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);

    // This is the difference between VARINT and ZIGZAG: conversion to signed.
    return (result >> 1) ^ (-(result & 1));
  }

  int64_t
  ForthInputBuffer::read_textint(util::ForthError& err) noexcept {
    if (pos_ >= length_) {
      err = util::ForthError::read_beyond;
      return 0;
    }

    const uint8_t* ptr = reinterpret_cast<uint8_t*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_
    );

    bool negative = false;
    if (ptr[pos_] == '-') {
      negative = true;
      pos_++;
      if (pos_ == length_) {
        err = util::ForthError::text_number_missing;
        return 0;
      }
    }

    if (ptr[pos_] < '0' || ptr[pos_] > '9') {
      err = util::ForthError::text_number_missing;
      return 0;
    }

    int64_t digits = 0;
    int64_t result = 0;
    do {
      digits++;
      result *= 10;
      result += ptr[pos_] - '0';

      pos_++;
      if (pos_ == length_) {
        break;
      }

      if (digits == 19) {
        err = util::ForthError::varint_too_big;
        return 0;
      }
    } while (ptr[pos_] >= '0' && ptr[pos_] <= '9');

    if (negative) {
      result = -result;
    }
    return result;
  }

  uint64_t bits_infinity = 0x7ff0000000000000;
  double positive_infinity = *(double*)&bits_infinity;
  double negative_infinity = -positive_infinity;

  double exponents[616] = {
    1e-307, 1e-306, 1e-305, 1e-304, 1e-303, 1e-302, 1e-301, 1e-300,

    1e-299, 1e-298, 1e-297, 1e-296, 1e-295, 1e-294, 1e-293, 1e-292, 1e-291, 1e-290,
    1e-289, 1e-288, 1e-287, 1e-286, 1e-285, 1e-284, 1e-283, 1e-282, 1e-281, 1e-280,
    1e-279, 1e-278, 1e-277, 1e-276, 1e-275, 1e-274, 1e-273, 1e-272, 1e-271, 1e-270,
    1e-269, 1e-268, 1e-267, 1e-266, 1e-265, 1e-264, 1e-263, 1e-262, 1e-261, 1e-260,
    1e-259, 1e-258, 1e-257, 1e-256, 1e-255, 1e-254, 1e-253, 1e-252, 1e-251, 1e-250,
    1e-249, 1e-248, 1e-247, 1e-246, 1e-245, 1e-244, 1e-243, 1e-242, 1e-241, 1e-240,
    1e-239, 1e-238, 1e-237, 1e-236, 1e-235, 1e-234, 1e-233, 1e-232, 1e-231, 1e-230,
    1e-229, 1e-228, 1e-227, 1e-226, 1e-225, 1e-224, 1e-223, 1e-222, 1e-221, 1e-220,
    1e-219, 1e-218, 1e-217, 1e-216, 1e-215, 1e-214, 1e-213, 1e-212, 1e-211, 1e-210,
    1e-209, 1e-208, 1e-207, 1e-206, 1e-205, 1e-204, 1e-203, 1e-202, 1e-201, 1e-200,

    1e-199, 1e-198, 1e-197, 1e-196, 1e-195, 1e-194, 1e-193, 1e-192, 1e-191, 1e-190,
    1e-189, 1e-188, 1e-187, 1e-186, 1e-185, 1e-184, 1e-183, 1e-182, 1e-181, 1e-180,
    1e-179, 1e-178, 1e-177, 1e-176, 1e-175, 1e-174, 1e-173, 1e-172, 1e-171, 1e-170,
    1e-169, 1e-168, 1e-167, 1e-166, 1e-165, 1e-164, 1e-163, 1e-162, 1e-161, 1e-160,
    1e-159, 1e-158, 1e-157, 1e-156, 1e-155, 1e-154, 1e-153, 1e-152, 1e-151, 1e-150,
    1e-149, 1e-148, 1e-147, 1e-146, 1e-145, 1e-144, 1e-143, 1e-142, 1e-141, 1e-140,
    1e-139, 1e-138, 1e-137, 1e-136, 1e-135, 1e-134, 1e-133, 1e-132, 1e-131, 1e-130,
    1e-129, 1e-128, 1e-127, 1e-126, 1e-125, 1e-124, 1e-123, 1e-122, 1e-121, 1e-120,
    1e-119, 1e-118, 1e-117, 1e-116, 1e-115, 1e-114, 1e-113, 1e-112, 1e-111, 1e-110,
    1e-109, 1e-108, 1e-107, 1e-106, 1e-105, 1e-104, 1e-103, 1e-102, 1e-101, 1e-100,

    1e-099, 1e-098, 1e-097, 1e-096, 1e-095, 1e-094, 1e-093, 1e-092, 1e-091, 1e-090,
    1e-089, 1e-088, 1e-087, 1e-086, 1e-085, 1e-084, 1e-083, 1e-082, 1e-081, 1e-080,
    1e-079, 1e-078, 1e-077, 1e-076, 1e-075, 1e-074, 1e-073, 1e-072, 1e-071, 1e-070,
    1e-069, 1e-068, 1e-067, 1e-066, 1e-065, 1e-064, 1e-063, 1e-062, 1e-061, 1e-060,
    1e-059, 1e-058, 1e-057, 1e-056, 1e-055, 1e-054, 1e-053, 1e-052, 1e-051, 1e-050,
    1e-049, 1e-048, 1e-047, 1e-046, 1e-045, 1e-044, 1e-043, 1e-042, 1e-041, 1e-040,
    1e-039, 1e-038, 1e-037, 1e-036, 1e-035, 1e-034, 1e-033, 1e-032, 1e-031, 1e-030,
    1e-029, 1e-028, 1e-027, 1e-026, 1e-025, 1e-024, 1e-023, 1e-022, 1e-021, 1e-020,
    1e-019, 1e-018, 1e-017, 1e-016, 1e-015, 1e-014, 1e-013, 1e-012, 1e-011, 1e-010,
    1e-009, 1e-008, 1e-007, 1e-006, 1e-005, 1e-004, 1e-003, 1e-002, 1e-001,

    1e+000, 1e+001, 1e+002, 1e+003, 1e+004, 1e+005, 1e+006, 1e+007, 1e+008, 1e+009,
    1e+010, 1e+011, 1e+012, 1e+013, 1e+014, 1e+015, 1e+016, 1e+017, 1e+018, 1e+019,
    1e+020, 1e+021, 1e+022, 1e+023, 1e+024, 1e+025, 1e+026, 1e+027, 1e+028, 1e+029,
    1e+030, 1e+031, 1e+032, 1e+033, 1e+034, 1e+035, 1e+036, 1e+037, 1e+038, 1e+039,
    1e+040, 1e+041, 1e+042, 1e+043, 1e+044, 1e+045, 1e+046, 1e+047, 1e+048, 1e+049,
    1e+050, 1e+051, 1e+052, 1e+053, 1e+054, 1e+055, 1e+056, 1e+057, 1e+058, 1e+059,
    1e+060, 1e+061, 1e+062, 1e+063, 1e+064, 1e+065, 1e+066, 1e+067, 1e+068, 1e+069,
    1e+070, 1e+071, 1e+072, 1e+073, 1e+074, 1e+075, 1e+076, 1e+077, 1e+078, 1e+079,
    1e+080, 1e+081, 1e+082, 1e+083, 1e+084, 1e+085, 1e+086, 1e+087, 1e+088, 1e+089,
    1e+090, 1e+091, 1e+092, 1e+093, 1e+094, 1e+095, 1e+096, 1e+097, 1e+098, 1e+099,

    1e+100, 1e+101, 1e+102, 1e+103, 1e+104, 1e+105, 1e+106, 1e+107, 1e+108, 1e+109,
    1e+110, 1e+111, 1e+112, 1e+113, 1e+114, 1e+115, 1e+116, 1e+117, 1e+118, 1e+119,
    1e+120, 1e+121, 1e+122, 1e+123, 1e+124, 1e+125, 1e+126, 1e+127, 1e+128, 1e+129,
    1e+130, 1e+131, 1e+132, 1e+133, 1e+134, 1e+135, 1e+136, 1e+137, 1e+138, 1e+139,
    1e+140, 1e+141, 1e+142, 1e+143, 1e+144, 1e+145, 1e+146, 1e+147, 1e+148, 1e+149,
    1e+150, 1e+151, 1e+152, 1e+153, 1e+154, 1e+155, 1e+156, 1e+157, 1e+158, 1e+159,
    1e+160, 1e+161, 1e+162, 1e+163, 1e+164, 1e+165, 1e+166, 1e+167, 1e+168, 1e+169,
    1e+170, 1e+171, 1e+172, 1e+173, 1e+174, 1e+175, 1e+176, 1e+177, 1e+178, 1e+179,
    1e+180, 1e+181, 1e+182, 1e+183, 1e+184, 1e+185, 1e+186, 1e+187, 1e+188, 1e+189,
    1e+190, 1e+191, 1e+192, 1e+193, 1e+194, 1e+195, 1e+196, 1e+197, 1e+198, 1e+199,

    1e+200, 1e+201, 1e+202, 1e+203, 1e+204, 1e+205, 1e+206, 1e+207, 1e+208, 1e+209,
    1e+210, 1e+211, 1e+212, 1e+213, 1e+214, 1e+215, 1e+216, 1e+217, 1e+218, 1e+219,
    1e+220, 1e+221, 1e+222, 1e+223, 1e+224, 1e+225, 1e+226, 1e+227, 1e+228, 1e+229,
    1e+230, 1e+231, 1e+232, 1e+233, 1e+234, 1e+235, 1e+236, 1e+237, 1e+238, 1e+239,
    1e+240, 1e+241, 1e+242, 1e+243, 1e+244, 1e+245, 1e+246, 1e+247, 1e+248, 1e+249,
    1e+250, 1e+251, 1e+252, 1e+253, 1e+254, 1e+255, 1e+256, 1e+257, 1e+258, 1e+259,
    1e+260, 1e+261, 1e+262, 1e+263, 1e+264, 1e+265, 1e+266, 1e+267, 1e+268, 1e+269,
    1e+270, 1e+271, 1e+272, 1e+273, 1e+274, 1e+275, 1e+276, 1e+277, 1e+278, 1e+279,
    1e+280, 1e+281, 1e+282, 1e+283, 1e+284, 1e+285, 1e+286, 1e+287, 1e+288, 1e+289,
    1e+290, 1e+291, 1e+292, 1e+293, 1e+294, 1e+295, 1e+296, 1e+297, 1e+298, 1e+299,

    1e+300, 1e+301, 1e+302, 1e+303, 1e+304, 1e+305, 1e+306, 1e+307, 1e+308
  };

  double
  ForthInputBuffer::read_textfloat(util::ForthError& err) noexcept {
    if (pos_ >= length_) {
      err = util::ForthError::read_beyond;
      return 0.0;
    }

    const uint8_t* ptr = reinterpret_cast<uint8_t*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_
    );

    bool negative = false;
    if (ptr[pos_] == '-') {
      negative = true;
      pos_++;
      if (pos_ == length_) {
        err = util::ForthError::text_number_missing;
        return 0.0;
      }
    }

    if (ptr[pos_] < '0' || ptr[pos_] > '9') {
      err = util::ForthError::text_number_missing;
      return 0.0;
    }

    int64_t integral = 0;
    do {
      integral *= 10;
      integral += ptr[pos_] - '0';
      pos_++;
    } while (pos_ != length_ && ptr[pos_] >= '0' && ptr[pos_] <= '9');

    double result = (double)integral;

    if (pos_ != length_ && ptr[pos_] == '.') {
      pos_++;

      if (pos_ == length_ || ptr[pos_] < '0' || ptr[pos_] > '9') {
        err = util::ForthError::text_number_missing;
        return 0.0;
      }

      int64_t power = 1;
      int64_t fractional = 0;
      do {
        power *= 10;
        fractional *= 10;
        fractional += ptr[pos_] - '0';
        pos_++;
      } while (pos_ != length_ && ptr[pos_] >= '0' && ptr[pos_] <= '9');

      result += (double)fractional / (double)power;
    }

    if (pos_ != length_ && (ptr[pos_] == 'e' || ptr[pos_] == 'E')) {
      pos_++;

      if (pos_ == length_) {
        err = util::ForthError::text_number_missing;
        return 0.0;
      }

      bool negative_exponent = false;
      if (ptr[pos_] == '+') {
        pos_++;
        if (pos_ == length_) {
          err = util::ForthError::text_number_missing;
          return 0.0;
        }
      }
      else if (ptr[pos_] == '-') {
        negative_exponent = true;
        pos_++;
        if (pos_ == length_) {
          err = util::ForthError::text_number_missing;
          return 0.0;
        }
      }

      if (ptr[pos_] < '0' || ptr[pos_] > '9') {
        err = util::ForthError::text_number_missing;
        return 0.0;
      }

      int64_t exponent = 0;
      do {
        exponent *= 10;
        exponent += ptr[pos_] - '0';
        pos_++;
      } while (pos_ != length_ && ptr[pos_] >= '0' && ptr[pos_] <= '9');

      if (negative_exponent) {
        exponent = -exponent;
      }

      exponent += 307;

      if (exponent < 0) {
        result = negative_infinity;
      }
      else if (exponent >= 616) {
        result = positive_infinity;
      }
      else {
        result *= exponents[exponent];
      }
    }

    if (negative) {
      result = -result;
    }
    return result;
  }

  void
  ForthInputBuffer::read_quotedstr(char* string_buffer, int64_t max_string_size, int64_t& length,
                                   util::ForthError& err) noexcept {
    if (pos_ >= length_) {
      err = util::ForthError::read_beyond;
      return;
    }

    const uint8_t* ptr = reinterpret_cast<uint8_t*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_
    );

    if (ptr[pos_] == '\"') {
      pos_++;
      if (pos_ == length_) {
        err = util::ForthError::quoted_string_missing;
        return;
      }
    }
    else {
      err = util::ForthError::quoted_string_missing;
      return;
    }

    length = 0;
    uint64_t code_point;
    int64_t i;
    while (ptr[pos_] != '\"') {
      // this while loop puts one character in the output buffer
      if (length == max_string_size) {
        // "doesn't fit in string buffer" means "it's not a string" for our purposes
        err = util::ForthError::quoted_string_missing;
        return;
      }

      // if escaped character
      if (ptr[pos_] == '\\') {
        pos_++;
        if (pos_ == length_) {
          err = util::ForthError::quoted_string_missing;
          return;
        }

        // which escape sequence?
        switch (ptr[pos_]) {
          case 'n':
            string_buffer[length] = '\n';
            break;
          case 'r':
            string_buffer[length] = '\r';
            break;
          case 't':
            string_buffer[length] = '\t';
            break;
          case 'b':
            string_buffer[length] = '\b';
            break;
          case 'f':
            string_buffer[length] = '\f';
            break;
          case '\"':
          case '/':
          case '\\':
            string_buffer[length] = (char)ptr[pos_];
            break;
          case 'u':
            // check all the positions that will be used *in the following loop*
            if (pos_ + 4 >= length_) {
              err = util::ForthError::quoted_string_missing;
              return;
            }
            code_point = 0;
            for (i = 0; i < 4; i++) {
              pos_++;  // first get past the 'u', then end on the last of the 4 hex digits
              code_point *= 16;
              if (ptr[pos_] >= '0' && ptr[pos_] <= '9') {
                code_point += ptr[pos_] - '0';
              }
              else if (ptr[pos_] >= 'a' && ptr[pos_] <= 'f') {
                code_point += ptr[pos_] - 'a' + 10;
              }
              else if (ptr[pos_] >= 'A' && ptr[pos_] <= 'F') {
                code_point += ptr[pos_] - 'A' + 10;
              }
              else {
                err = util::ForthError::quoted_string_missing;
                return;
              }
            }
            // https://stackoverflow.com/a/4609989/1623645
            if (code_point < 0x80) {
              string_buffer[length] = (char)code_point;
            }
            else if (code_point < 0x800) {
              if (length + 1 >= max_string_size) {
                err = util::ForthError::quoted_string_missing;
                return;
              }
              string_buffer[length] = (char)(192 + code_point / 64);
              length++;
              string_buffer[length] = (char)(128 + code_point % 64);
            }
            else if (code_point - 0xd800u < 0x800) {
              err = util::ForthError::quoted_string_missing;
              return;
            }
            else if (code_point < 0x10000) {
              if (length + 2 >= max_string_size) {
                err = util::ForthError::quoted_string_missing;
                return;
              }
              string_buffer[length] = (char)(224 + code_point / 4096);
              length++;
              string_buffer[length] = (char)(128 + code_point / 64 % 64);
              length++;
              string_buffer[length] = (char)(128 + code_point % 64);
            }
            else if (code_point < 0x110000) {
              // this one can't be reached by 4 hex-digits in JSON, but for completeness...
              if (length + 3 >= max_string_size) {
                err = util::ForthError::quoted_string_missing;
                return;
              }
              string_buffer[length] = (char)(240 + code_point / 262144);
              length++;
              string_buffer[length] = (char)(128 + code_point / 4096 % 64);
              length++;
              string_buffer[length] = (char)(128 + code_point / 64 % 64);
              length++;
              string_buffer[length] = (char)(128 + code_point % 64);
            }
            else {
              err = util::ForthError::quoted_string_missing;
              return;
            }
            break;
          default:
            err = util::ForthError::quoted_string_missing;
            return;
        }
      }
      // else unescaped character
      else {
        string_buffer[length] = (char)ptr[pos_];
      }

      // whether the input was an escaped sequence or not, pos_ ended on an interpreted byte
      pos_++;

      // the input buffer should not end without closing the string (this while loop)
      if (pos_ == length_) {
        err = util::ForthError::quoted_string_missing;
        return;
      }
      // and the output has increased by only one character
      length++;
    }

    // we are now at the final quotation mark, so step one past it
    pos_++;
  }

  void
  ForthInputBuffer::seek(int64_t to, util::ForthError& err) noexcept {
    if (to < 0  ||  to > length_) {
      err = util::ForthError::seek_beyond;
    }
    else {
      pos_ = to;
    }
  }

  void
  ForthInputBuffer::skip(int64_t num_bytes, util::ForthError& err) noexcept {
    int64_t next = pos_ + num_bytes;
    if (next < 0  ||  next > length_) {
      err = util::ForthError::skip_beyond;
    }
    else {
      pos_ = next;
    }
  }

  void
  ForthInputBuffer::skipws() noexcept {
    uint8_t* byte;
    while (pos_ < length_) {
      byte = reinterpret_cast<uint8_t*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_ + (size_t)pos_
      );
      if (*byte == ' ' || *byte == '\n' || *byte == '\r' || *byte == '\t') {
        pos_++;
      }
      else {
        break;
      }
    }
  }

  bool
  ForthInputBuffer::end() const noexcept {
    return pos_ == length_;
  }

  int64_t
  ForthInputBuffer::pos() const noexcept {
    return pos_;
  }

  int64_t
  ForthInputBuffer::len() const noexcept {
    return length_;
  }

  std::shared_ptr<void>
  ForthInputBuffer::ptr() noexcept {
    return ptr_;
  }

}
