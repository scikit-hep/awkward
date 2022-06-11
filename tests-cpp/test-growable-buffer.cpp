#include "../src/awkward/cpp-headers/GrowableBuffer.h"

#include <complex>
#include <cassert>

void test_full() {
  int data_size = 100;
  size_t initial = 25;
  auto buffer = awkward::GrowableBuffer<int16_t>::full(initial, -2, data_size);
  for (int64_t at = 0; at < buffer.length(); at++) {
    assert(buffer.getitem_at_nowrap(at) == int16_t(-2));
  }
}

void test_arange() {
  int data_size = 25;
  size_t initial = 50;
  auto buffer = awkward::GrowableBuffer<int64_t>::arange(initial, data_size);
  for (int64_t at = 0; at < buffer.length(); at++) {
    assert(buffer.getitem_at_nowrap(at) == at);
  }
}

void test_zeros() {
  int data_size = 100;
  size_t initial = 100;
  auto buffer = awkward::GrowableBuffer<uint32_t>::zeros(initial, data_size);
  for (int64_t at = 0; at < buffer.length(); at++) {
    assert(buffer.getitem_at_nowrap(at) == 0);
  }
}

void test_float() {
  int data_size = 18;
  float data[18] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9};
  size_t initial = 4;
  auto buffer = awkward::GrowableBuffer<float>::empty(initial);
  for (int64_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }
  float* ptr = new float[data_size];
  buffer.concatenate(ptr);
  for (int64_t at = 0; at < buffer.length(); at++) {
    assert(ptr[at] == data[at]);
  }
}

void test_int64() {
  int data_size = 10;
  float data[10] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
  size_t initial = 8;
  auto buffer = awkward::GrowableBuffer<int64_t>::empty(initial);
  for (int i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }
  int64_t* ptr = new int64_t[data_size];
  buffer.concatenate(ptr);
  for (int at = 0; at < buffer.length(); at++) {
    assert(ptr[at] == data[at]);
  }
}

void test_bool() {
  int data_size = 12;
  bool data[12] = {false, true, false, false, true, false, true,
                   true, false, true, false, false};
  size_t initial = 5;
  auto buffer = awkward::GrowableBuffer<bool>(initial);
  for (int64_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }
  bool* ptr = new bool[data_size];
  buffer.concatenate(ptr);
  for (int64_t at = 0; at < buffer.length(); at++) {
    assert(ptr[at] == data[at]);
  }
}

void test_double() {
  int data_size = 9;
  double data[9] = {1.01, 2.02, 3.03, 4.04, 5.05, 6.06, 7.07, 8.08, 9.09};
  size_t initial = 6;
  auto buffer = awkward::GrowableBuffer<double>::empty(initial);
  for (int64_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }
  double* ptr = new double[data_size];
  buffer.concatenate(ptr);
  for (int64_t at = 0; at < buffer.length(); at++) {
    assert(ptr[at] == data[at]);
  }
}

void test_complex() {
  int data_size = 10;
  std::complex<double> data[10] = {{0, 0}, {1.1, 0.1}, {2.2, 0.2}, {3.3, 0.3}, {4.4, 0.4},
                                   {5.5, 0.5}, {6.6, 0.6}, {7.7, 0.7}, {8.8, 0.8}, {9.9, 0.9}};
  size_t initial = 3;
  auto buffer = awkward::GrowableBuffer<std::complex<double>>::empty(initial);
  for (int64_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }
  std::complex<double>* ptr = new std::complex<double>[data_size];
  buffer.concatenate(ptr);
  for (int64_t at = 0; at < buffer.length(); at++) {
    assert(ptr[at] == data[at]);
  }
}

int main(int argc, const char ** argv) {
    test_full();
    test_arange();
    test_zeros();
    test_float();
    test_int64();
    test_bool();
    test_double();
    test_complex();
    return 0;
}
