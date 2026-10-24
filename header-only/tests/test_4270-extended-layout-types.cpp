// BSD 3-Clause License; see
// https://github.com/scikit-hep/awkward/blob/main/LICENSE
#include "awkward/LayoutBuilder.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>

// Emit real builder buffers so Python can validate their NumPy interpretation.
template <typename Builder> void emit(const Builder &builder) {
  std::map<std::string, size_t> sizes;
  builder.buffer_nbytes(sizes);
  std::map<std::string, std::vector<std::max_align_t>> storage;
  std::map<std::string, void *> pointers;
  for (const auto &[name, size] : sizes) {
    storage[name].resize((size + sizeof(std::max_align_t) - 1) /
                         sizeof(std::max_align_t));
    pointers[name] = storage[name].data();
  }
  builder.to_buffers(pointers);
  std::cout << "{\"form\":" << builder.form()
            << ",\"length\":" << builder.length() << ",\"buffers\":{";
  bool first = true;
  for (const auto &[name, size] : sizes) {
    if (!first)
      std::cout << ',';
    first = false;
    std::cout << '"' << name << "\":\"";
    auto bytes = static_cast<const unsigned char *>(pointers[name]);
    for (size_t i = 0; i < size; ++i) {
      std::cout << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned>(bytes[i]);
    }
    std::cout << std::dec << '"';
  }
  std::cout << "}}\n";
}

template <typename T> void check(const std::vector<T> &values) {
  awkward::LayoutBuilder::Numpy<T> flat(awkward::BuilderOptions(2, 2));
  const auto primitive = awkward::type_to_name<T>();
  assert(flat.form().find(primitive) != std::string::npos);
  assert((awkward::type_to_form<T, int64_t>(0).find(primitive) !=
          std::string::npos));
  for (const auto value : values)
    flat.append(value);
  std::map<std::string, size_t> sizes;
  flat.buffer_nbytes(sizes);
  assert(sizes.at("node0-data") == values.size() * sizeof(T));
  std::vector<T> copy(values.size());
  std::map<std::string, void *> buffers{{"node0-data", copy.data()}};
  flat.to_buffers(buffers);
  for (size_t i = 0; i < values.size(); ++i) {
    if constexpr (awkward::is_specialization<T, std::complex>::value) {
      assert(copy[i] == values[i]);
      assert(std::signbit(copy[i].real()) == std::signbit(values[i].real()));
    } else {
      // IEEE binary16 has no padding bytes.
      assert(std::memcmp(&copy[i], &values[i], sizeof(T)) == 0);
    }
  }
  emit(flat);

  awkward::LayoutBuilder::ListOffset<int64_t, awkward::LayoutBuilder::Numpy<T>>
      nested;
  auto &content = nested.begin_list();
  for (const auto value : values)
    content.append(value);
  nested.end_list();
  nested.begin_list();
  nested.end_list();
  emit(nested);
  flat.clear();
  assert(flat.length() == 0);
}

int main() {
#ifdef AWKWARD_HAVE_FLOAT16
  assert(awkward::type_to_name<awkward::float16>() == "float16");
  std::vector<awkward::float16> half_values;
  for (float value :
       {-0.0f, std::ldexp(1.0f, -24), 1.5f, 65504.0f, INFINITY, NAN}) {
    half_values.push_back(static_cast<awkward::float16>(value));
  }
  check(half_values);
#endif
  using Complex = std::complex<long double>;
  const auto name = awkward::type_to_name<Complex>();
  assert(name == (sizeof(Complex) == 32 ? "complex256" : "complex128"));
  check<Complex>(
      {Complex(1.0L + std::ldexp(1.0L, -60), -2.5L), Complex(-0.0L, 3.0L)});
  assert(awkward::type_to_name<long double>() ==
         (sizeof(long double) == 16 ? "float128" : "float64"));
  return 0;
}
