#include <iostream>
#include <stdint.h>

int main(int argc, char** argv) {
  const size_t byte_count = 126;

  uint8_t raw_bytes[byte_count * 2]{
       252, 255, 191, 255, 251, 255, 255, 239, 251, 255, 191, 255, 255,
       251, 255, 251, 247, 255, 191, 255, 239, 255, 251, 255, 255, 251,
       251, 255, 239, 255, 255, 251, 255, 247, 255, 251, 254, 255, 191,
       255, 255, 254, 255, 191, 255, 251, 255, 239, 255, 239, 254, 255,
       254, 255, 223, 255, 255, 239, 251, 255, 255, 251, 251, 255, 255,
       239, 239, 255, 255, 239, 255, 191, 255, 247, 255, 254, 191, 255,
       255, 239, 255, 253, 251, 255, 191, 255, 191, 255, 239, 255, 255,
       191, 255, 251, 255, 254, 255, 239, 253, 255, 254, 255, 255, 239,
       255, 191, 255, 254, 254, 255, 255, 254, 239, 255, 251, 255, 255,
       127, 251, 255, 251, 255, 254, 255, 251, 255, 127, 254, 255, 255,
       239, 255, 191, 254, 255, 254, 255, 255, 251, 255, 254, 239, 255,
       255, 127, 255, 254, 255, 254, 251, 255, 239, 255, 239, 255, 254,
       255, 255, 191, 207, 255, 239, 255, 255, 239, 253, 255, 251, 255,
       255, 251, 191, 255, 255, 254, 251, 255, 254, 255, 254, 255, 127,
       255, 255, 254, 239, 255, 251, 255, 254, 255, 191, 255, 223, 255,
       255, 254, 239, 255, 255, 254, 191, 255, 254, 255, 255, 254, 255,
       251, 127, 255, 255, 251, 191, 255, 255, 254, 255, 191, 255, 239,
       255, 191, 191, 255, 251, 255, 255, 239, 255, 223, 191, 255, 251,
       255, 255, 251, 191, 255, 255, 191, 255, 191, 255, 127, 251, 255,
       251, 255, 251, 255, 254};

  uint8_t output[100];
  for (int64_t i = 0;  i < 100;  i++) {
    output[i] = 123;
  }
  size_t output_i = 0;

  size_t bits_remaining = byte_count * 8;
  size_t bit_width = 2;
  uint8_t mask = 3;

  size_t current_byte = 0;
  uint8_t data = raw_bytes[current_byte];
  uint8_t bits_wnd_l = 8;
  uint8_t bits_wnd_r = 0;

  while (bits_remaining >= bit_width) {
    // std::cout << "total " << bits_remaining << " current_byte " << current_byte << " bits_wnd_l " << (int64_t)bits_wnd_l << " bits_wnd_r " << (int64_t)bits_wnd_r << " data " << (int64_t)data << std::endl;

    if (bits_wnd_r >= 8) {
      // std::cout << "bits_wnd_r >= 8" << std::endl;
      bits_wnd_r -= 8;
      bits_wnd_l -= 8;
      data >>= 8;
    }
    else if (bits_wnd_l - bits_wnd_r >= bit_width) {
        // std::cout << "bits_wnd_l - bits_wnd_r >= width" << std::endl;
        if (output_i < 100) {
          output[output_i] = (data >> bits_wnd_r) & mask;
        }
        output_i++;
        // std::cout << "WROTE BYTE " << ((data >> bits_wnd_r) & mask) << std::endl;
        bits_remaining -= bit_width;
        bits_wnd_r += bit_width;
    }
    else if (current_byte + 1 < byte_count) {
        // std::cout << "current_byte + 1 < byte_count" << std::endl;
        current_byte++;
        data |= (raw_bytes[current_byte] << bits_wnd_l);
        bits_wnd_l += 8;
    }
    else {
      // std::cout << "can this happen?" << std::endl;
    }
  }

  for (output_i = 0;  output_i < 100;  output_i++) {
    if (output[output_i] == 0) {
      std::cout << " ZERO";
    }
    else if (output[output_i] == 1) {
      std::cout << " ONE!";
    }
    else if (output[output_i] == 2) {
      std::cout << "  two";
    }
    else if (output[output_i] == 3) {
      std::cout << "    3";
    }
    else {
      std::cout << "   DIE DIE DIE   ";
    }
    if ((output_i + 1) % 22 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}
