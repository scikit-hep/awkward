// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "../src/awkward/_v2/cpp-headers/GrowableBuffer.h"

#include <complex>
#include <cassert>

void test_full() {
  size_t data_size = 100;
  size_t initial = 25;

  auto buffer = awkward::GrowableBuffer<int16_t>::full(initial, -2, data_size);

  int16_t* ptr = new int16_t[buffer.length()];
  buffer.concatenate(ptr);

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr[i] == int16_t(-2));
  }
}

void test_arange() {
  size_t data_size = 25;
  size_t initial = 50;

  auto buffer = awkward::GrowableBuffer<int64_t>::arange(initial, data_size);

  int64_t* ptr = new int64_t[buffer.length()];
  buffer.concatenate(ptr);

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr[i] == (int64_t)i);
  }
}

void test_zeros() {
  size_t data_size = 100;
  size_t initial = 100;

  auto buffer = awkward::GrowableBuffer<uint32_t>::zeros(initial, data_size);

  uint32_t* ptr = new uint32_t[buffer.length()];
  buffer.concatenate(ptr);

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr[i] == (uint32_t)0);
  }
}

void test_float() {
  size_t data_size = 18;
  size_t initial = 4;
  float data[18] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9};

  auto buffer = awkward::GrowableBuffer<float>::empty(initial);

  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }

  float* ptr = new float[buffer.length()];
  buffer.concatenate(ptr);

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr[i] == data[i]);
  }
}

void test_int64() {
  size_t data_size = 10;
  size_t initial = 8;
  float data[10] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

  auto buffer = awkward::GrowableBuffer<int64_t>::empty(initial);

  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }

  int64_t* ptr = new int64_t[buffer.length()];
  buffer.concatenate(ptr);

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr[i] == data[i]);
  }
}

void test_bool() {
  size_t data_size = 12;
  size_t initial = 5;
  bool data[12] = {false, true, false, false, true, false, true,
                   true, false, true, false, false};

  auto buffer = awkward::GrowableBuffer<bool>(initial);

  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }

  bool* ptr = new bool[buffer.length()];
  buffer.concatenate(ptr);

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr[i] == data[i]);
  }
}

void test_double() {
  size_t data_size = 9;
  size_t initial = 6;
  double data[9] = {1.01, 2.02, 3.03, 4.04, 5.05, 6.06, 7.07, 8.08, 9.09};

  auto buffer = awkward::GrowableBuffer<double>::empty(initial);

  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }

  double* ptr = new double[buffer.length()];
  buffer.concatenate(ptr);

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr[i] == data[i]);
  }
}

void test_complex() {
  size_t data_size = 10;
  size_t initial = 3;
  std::complex<double> data[10] = {{0, 0}, {1.1, 0.1}, {2.2, 0.2}, {3.3, 0.3}, {4.4, 0.4},
                                   {5.5, 0.5}, {6.6, 0.6}, {7.7, 0.7}, {8.8, 0.8}, {9.9, 0.9}};

  auto buffer = awkward::GrowableBuffer<std::complex<double>>::empty(initial);
  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }

  std::complex<double>* ptr = new std::complex<double>[buffer.length()];
  buffer.concatenate(ptr);

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr[i] == data[i]);
  }
}

void test_extend() {
  size_t data_size = 15;
  size_t initial = 5;
  double data[15] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                     2.1, 2.2, 2.3, 2.4, 2.5, 2.6};

  auto buffer = awkward::GrowableBuffer<double>::empty(initial);

  buffer.extend(data, data_size);

  double* ptr = new double[buffer.length()];
  buffer.concatenate(ptr);

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr[i] == data[i]);
  }
}

int main(int /* argc */, const char ** /* argv */) {
  test_full();
  test_arange();
  test_zeros();
  test_float();
  test_int64();
  test_bool();
  test_double();
  test_complex();
  test_extend();

  return 0;
}
