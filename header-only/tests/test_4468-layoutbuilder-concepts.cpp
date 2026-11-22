// BSD 3-Clause License; see
// https://github.com/scikit-hep/awkward/blob/main/LICENSE
#include "awkward/LayoutBuilder.h"
#include <cassert>

namespace lb = awkward::LayoutBuilder;

template <typename T>
concept HasNumpy = requires {
  typename lb::Numpy<T>;
};
template <typename T>
concept HasList = requires {
  typename lb::ListOffset<T, lb::Numpy<double>>;
};
template <typename T>
concept HasIndexed = requires {
  typename lb::Indexed<T, lb::Numpy<double>>;
};
template <typename T>
concept HasOption = requires {
  typename lb::IndexedOption<T, lb::Numpy<double>>;
};
template <typename T>
concept HasString = requires {
  typename lb::String<T>;
};
template <typename T, typename I>
concept HasUnion = requires {
  typename lb::Union<T, I, lb::Numpy<double>, lb::Numpy<bool>>;
};

static_assert(HasNumpy<bool> && HasNumpy<char> && HasNumpy<int8_t>);
static_assert(HasNumpy<uint8_t> && HasNumpy<int16_t> && HasNumpy<uint16_t>);
static_assert(HasNumpy<int32_t> && HasNumpy<uint32_t>);
static_assert(HasNumpy<int64_t> && HasNumpy<uint64_t>);
static_assert(HasNumpy<float> && HasNumpy<double>);
static_assert(HasNumpy<std::complex<float>> && HasNumpy<std::complex<double>>);
static_assert(HasNumpy<long double> ==
              (sizeof(long double) == 8 || sizeof(long double) == 16));
static_assert(HasNumpy<std::complex<long double>> ==
              (sizeof(std::complex<long double>) == 16 ||
               sizeof(std::complex<long double>) == 32));
#ifdef AWKWARD_HAVE_FLOAT16
static_assert(HasNumpy<awkward::float16>);
#endif
static_assert(!HasNumpy<void> && !HasNumpy<std::string>);
static_assert(!HasNumpy<double *> && !HasNumpy<double &>);
static_assert(!HasNumpy<const double> && !HasNumpy<volatile double>);
static_assert(!HasNumpy<std::complex<int>>);

template <typename T>
constexpr bool accepts_index =
    HasList<T> &&HasIndexed<T> &&HasString<T> &&HasUnion<int8_t, T>;
static_assert(accepts_index<int32_t> && accepts_index<uint32_t> &&
              accepts_index<int64_t>);
static_assert(!HasList<bool> && !HasList<int8_t> && !HasList<int16_t>);
static_assert(!HasIndexed<uint64_t> && !HasIndexed<float>);
static_assert(!HasString<const int64_t> && !HasString<int64_t &>);
static_assert(!HasUnion<int8_t, double> && !HasUnion<int8_t, uint64_t>);
static_assert(HasOption<int32_t> && HasOption<int64_t>);
static_assert(!HasOption<uint32_t> && !HasOption<uint64_t> && !HasOption<bool>);
static_assert(!HasOption<double> && !HasOption<const int32_t>);
static_assert(!HasUnion<uint8_t, int32_t> && !HasUnion<int16_t, int32_t>);
static_assert(!HasUnion<const int8_t, int32_t>);

int main() {
  lb::ListOffset<int64_t, lb::Numpy<double>> lists;
  lists.begin_list().append(1.5);
  lists.end_list();
  std::string error;
  assert(lists.is_valid(error));
  assert(lists.length() == 1);
  lb::IndexedOption<int32_t, lb::Numpy<double>> options;
  options.append_valid().append(2.5);
  options.append_invalid();
  assert(options.is_valid(error));
  assert(options.length() == 2);
}
