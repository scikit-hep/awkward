// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_subrange_equal.cpp", line)

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "awkward/kernels.h"

namespace awkward {
  // Checks if the first range [first1, last1) is lexicographically less than
  // the second range [first2, last2).
  // Complexity: At most 2·min(N1, N2)
  template<class InputIt1, class InputIt2>
  bool lexicographical_compare(InputIt1 first1, InputIt1 last1,
                               InputIt2 first2, InputIt2 last2)
  {
      for ( ; (first1 != last1) && (first2 != last2); ++first1, (void) ++first2 ) {
          if (*first1 < *first2) return true;
          if (*first2 < *first1) return false;
      }
      return (first1 == last1) && (first2 != last2);
  }

  // A function to do lexicographical comparisons
  template <typename Container>
  bool compare(const Container& left, const Container& right) {
    return awkward::lexicographical_compare(left.begin(), left.end(),
                                            right.begin(), right.end());
  }

  // Use that comparison function to sort a range:
  template <typename ContainerIterator>
  void sort_by_lexicographical_comapre(ContainerIterator from,
                                       ContainerIterator to)
  {
    std::sort(from, to, awkward::compare<typename ContainerIterator::value_type>);
  }
}

template <typename T>
ERROR awkward_NumpyArray_subrange_equal(
    const T* fromptr,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t length,
    bool* toequal) {

  // FIXME: so far empty ranges are ignored
  std::vector<std::vector<T>> ranges;
  for (int64_t i = 0; i < length; i++) {
    std::vector<T> range;
    for(int64_t j = fromstarts[i]; j < fromstops[i]; j++) {
      range.push_back(fromptr[j]);
    }
    if (!range.empty()) {
      ranges.push_back(range);
    }
    std::sort(ranges.back().begin(), ranges.back().end());
  }
  awkward::sort_by_lexicographical_comapre(ranges.begin(), ranges.end());

  bool differ = true;
  for (int64_t i = 0;  i < ranges.size() - 1;  i++) {
    if ((ranges[i].size() == ranges[i + 1].size())  &&
        (ranges[i] == ranges[i + 1])) {
      differ = false;
      break;
    }
  }

  *toequal = !differ;

  return success();
}

ERROR awkward_NumpyArray_subrange_equal_bool(
  const bool* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<bool>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int8(
  const int8_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int8_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint8(
  const uint8_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint8_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int16(
  const int16_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int16_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint16(
  const uint16_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint16_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int32(
  const int32_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int32_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint32(
  const uint32_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint32_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int64(
  const int64_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int64_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint64(
  const uint64_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint64_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_float32(
  const float* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<float>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_float64(
  const double* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<double>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
