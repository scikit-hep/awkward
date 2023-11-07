// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#include "awkward/BuilderOptions.h"
#include "awkward/GrowableBuffer.h"

#include <complex>
#include <cassert>

void test_full() {
  constexpr size_t data_size = 100;
  awkward::BuilderOptions options { 25, 1 };

  auto buffer = awkward::GrowableBuffer<int16_t>::full(options, -2, data_size);
  assert(buffer.length() == data_size);

  std::unique_ptr<int16_t[]> ptr(new int16_t[buffer.length()]);
  buffer.move_to(ptr.get());
  assert(buffer.length() == 0);
  buffer.clear();
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr.get()[i] == int16_t(-2));
  }
}

void test_arange() {
  constexpr size_t data_size = 25;
  awkward::BuilderOptions options { 50, 1 };

  auto buffer = awkward::GrowableBuffer<int64_t>::arange(options, data_size);
  assert(buffer.length() == data_size);

  std::unique_ptr<int64_t[]> ptr(new int64_t[buffer.length()]);
  buffer.concatenate(ptr.get());
  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr.get()[i] == (int64_t)i);
  }
}

void test_zeros() {
  constexpr size_t data_size = 100;
  awkward::BuilderOptions options { 100, 1 };

  auto buffer = awkward::GrowableBuffer<uint32_t>::zeros(options, data_size);
  assert(buffer.length() == data_size);

  std::unique_ptr<uint32_t[]> ptr(new uint32_t[buffer.length()]);
  buffer.concatenate(ptr.get());
  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr.get()[i] == (uint32_t)0);
  }
}

void test_float() {
  constexpr size_t data_size = 18;
  awkward::BuilderOptions options { 4, 1 };

  float data[data_size] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                           2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9};

  auto buffer = awkward::GrowableBuffer<float>::empty(options);
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }
  assert(buffer.length() == data_size);

  std::unique_ptr<float[]> ptr(new float[buffer.length()]);
  buffer.concatenate(ptr.get());
  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr.get()[i] == data[i]);
  }
}

void test_int64() {
  constexpr size_t data_size = 10;
  awkward::BuilderOptions options { 8, 1 };

  int64_t data[data_size] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

  auto buffer = awkward::GrowableBuffer<int64_t>::empty(options);

  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }
  assert(buffer.length() == data_size);

  std::unique_ptr<int64_t[]> ptr(new int64_t[buffer.length()]);
  buffer.concatenate(ptr.get());
  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr.get()[i] == data[i]);
  }
}

void test_bool() {
  constexpr size_t data_size = 12;
  awkward::BuilderOptions options {5, 1};

  bool data[data_size] = {false, true, false, false, true, false, true,
                          true, false, true, false, false};

  auto buffer = awkward::GrowableBuffer<bool>(options);

  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }
  assert(buffer.length() == data_size);

  std::unique_ptr<bool[]> ptr(new bool[buffer.length()]);
  buffer.concatenate(ptr.get());
  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr.get()[i] == data[i]);
  }
}

void test_double() {
  constexpr size_t data_size = 9;
  awkward::BuilderOptions options { 6, 1 };

  double data[data_size] = {1.01, 2.02, 3.03, 4.04, 5.05, 6.06, 7.07, 8.08, 9.09};

  auto buffer = awkward::GrowableBuffer<double>::empty(options);

  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }

  std::unique_ptr<double[]> ptr(new double[buffer.length()]);
  buffer.concatenate(ptr.get());
  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr.get()[i] == data[i]);
  }
}

void test_complex() {
  constexpr size_t data_size = 10;
  awkward::BuilderOptions options { 3, 1 };

  std::complex<double> data[data_size] = {{0, 0}, {1.1, 0.1}, {2.2, 0.2}, {3.3, 0.3}, {4.4, 0.4},
                                          {5.5, 0.5}, {6.6, 0.6}, {7.7, 0.7}, {8.8, 0.8}, {9.9, 0.9}};

  auto buffer = awkward::GrowableBuffer<std::complex<double>>::empty(options);
  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }

  std::complex<double>* ptr = new std::complex<double>[buffer.length()];
  buffer.concatenate(ptr);
  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr[i] == data[i]);
  }
}

void test_extend() {
  constexpr size_t data_size = 15;
  awkward::BuilderOptions options { 5, 1 };

  double data[data_size] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                            2.1, 2.2, 2.3, 2.4, 2.5, 2.6};

  auto buffer = awkward::GrowableBuffer<double>::empty(options);

  buffer.extend(data, data_size);

  std::unique_ptr<double[]> ptr(new double[buffer.length()]);
  buffer.concatenate(ptr.get());
  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr.get()[i] == data[i]);
  }
}

void test_append_and_get_ref() {
  constexpr size_t data_size = 15;
  awkward::BuilderOptions options { 5, 1 };

  double data[data_size] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                            2.1, 2.2, 2.3, 2.4, 2.5, 2.6};

  double val;
  double& ref = val;

  auto buffer = awkward::GrowableBuffer<double>::empty(options);
  for (size_t i = 0; i < data_size; i++) {
    ref = buffer.append_and_get_ref(data[i]);
    assert(ref == data[i]);
    assert(val == data[i]);
  }

  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);
}

template<typename FROM, typename TO>
void test_copy_complex_as_complex() {
  constexpr size_t data_size = 10;
  awkward::BuilderOptions options { 5, 1 };

  std::complex<FROM> data[data_size] = {{0, 0}, {1.1, 0.1}, {2.2, 0.2}, {3.3, 0.3}, {4.4, 0.4},
                                        {5.5, 0.5}, {6.6, 0.6}, {7.7, 0.7}, {8.8, 0.8}, {9.9, 0.9}};

  auto buffer = awkward::GrowableBuffer<std::complex<FROM>>::empty(options);
  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }

  std::unique_ptr<std::complex<FROM>[]> ptr(new std::complex<FROM>[buffer.length()]);
  buffer.concatenate(ptr.get());

  for (size_t i = 0; i < buffer.length(); i++) {
    assert(ptr.get()[i] == data[i]);
  }

  auto to_buffer = awkward::GrowableBuffer<std::complex<FROM>>::template copy_as<std::complex<TO>>(buffer);
  std::complex<TO>* ptr2 = new std::complex<TO>[to_buffer.length()];
  to_buffer.concatenate(ptr2);

  for (size_t i = 0; i < to_buffer.length(); i++) {
    assert(ptr2[i].real() == (TO)data[i].real());
    assert(ptr2[i].imag() == (TO)data[i].imag());
  }

  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  assert(to_buffer.length() == data_size);
  to_buffer.clear();
  assert(to_buffer.length() == 0);
}

template<typename FROM, typename TO>
void test_copy_complex_as() {
  constexpr size_t data_size = 10;
  awkward::BuilderOptions options { 5, 1 };

  std::complex<FROM> data[data_size] = {{0, 0}, {1.1, 0.1}, {2.2, 0.2}, {3.3, 0.3}, {4.4, 0.4},
                                        {5.5, 0.5}, {6.6, 0.6}, {7.7, 0.7}, {8.8, 0.8}, {9.9, 0.9}};

  auto buffer = awkward::GrowableBuffer<std::complex<FROM>>::empty(options);
  for (size_t i = 0; i < data_size; i++) {
    buffer.append(data[i]);
  }

  std::unique_ptr<std::complex<FROM>[]> ptr(new std::complex<FROM>[buffer.length()]);
  buffer.concatenate(ptr.get());

  for (size_t i = 0; i < data_size; i++) {
    assert(ptr.get()[i] == data[i]);
  }

  auto to_buffer = awkward::GrowableBuffer<std::complex<FROM>>::template copy_as<TO>(buffer);
  std::unique_ptr<TO[]> ptr2(new TO[to_buffer.length()]);
  to_buffer.concatenate(ptr2.get());

  for (size_t i = 0, j = 0; i < to_buffer.length() * 0.5; i++, j+=2) {
    assert(ptr2.get()[j] == (TO)data[i].real());
    assert(ptr2.get()[j+1] == (TO)data[i].imag());
  }

  assert(buffer.length() == data_size);
  buffer.clear();
  assert(buffer.length() == 0);

  assert(to_buffer.length() == data_size << 1);
  to_buffer.clear();
  assert(to_buffer.length() == 0);
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
  test_append_and_get_ref();
  test_copy_complex_as_complex<double, double>();
  test_copy_complex_as_complex<double, long double>();
  test_copy_complex_as_complex<double, float>();
  test_copy_complex_as_complex<float, float>();
  test_copy_complex_as_complex<float, double>();
  test_copy_complex_as_complex<float, long double>();
  test_copy_complex_as_complex<long double, float>();
  test_copy_complex_as_complex<long double, double>();
  test_copy_complex_as_complex<long double, long double>();
  test_copy_complex_as<long double, long double>();
  test_copy_complex_as<long double, double>();
  test_copy_complex_as<long double, float>();
  test_copy_complex_as<long double, int64_t>();
  test_copy_complex_as<long double, uint8_t>();
  test_copy_complex_as<double, long double>();
  test_copy_complex_as<double, double>();
  test_copy_complex_as<double, float>();
  test_copy_complex_as<double, int64_t>();
  test_copy_complex_as<double, uint8_t>();
  test_copy_complex_as<float, long double>();
  test_copy_complex_as<float, double>();
  test_copy_complex_as<float, float>();
  test_copy_complex_as<float, int64_t>();
  test_copy_complex_as<float, uint8_t>();

  return 0;
}
