// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <type_traits>

#include "awkward/Index.h"

using namespace awkward;

template <typename T>
const std::string IndexOf<T>::tostring() const {
  return tostring_part("", "", "");
}

template <typename T>
const std::string IndexOf<T>::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  std::string name = "Unrecognized Index";
  if (std::is_same<T, uint8_t>::value) {
    name = "Index8";
  }
  else if (std::is_same<T, int32_t>::value) {
    name = "Index32";
  }
  else if (std::is_same<T, int64_t>::value) {
    name = "Index64";
  }
  out << indent << pre << "<" << name << " i=\"[";
  if (length_ <= 10) {
    for (int64_t i = 0;  i < length_;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << (int64_t)get(i);
    }
  }
  else {
    for (int64_t i = 0;  i < 5;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << (int64_t)get(i);
    }
    out << " ... ";
    for (int64_t i = length_ - 5;  i < length_;  i++) {
      if (i != length_ - 5) {
        out << " ";
      }
      out << (int64_t)get(i);
    }
  }
  out << "]\" at=\"0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
  return out.str();
}

template <typename T>
T IndexOf<T>::get(int64_t at) const {
  return ptr_.get()[(size_t)(offset_ + at)];
}

template <typename T>
IndexOf<T> IndexOf<T>::slice(int64_t start, int64_t stop) const {
  return IndexOf<T>(ptr_, offset_ + start*(start != stop), stop - start);
}

template <typename T>
const std::shared_ptr<Index> IndexOf<T>::shallow_copy() const {
  return std::shared_ptr<Index>(new IndexOf<T>(ptr_, offset_, length_));
}

namespace awkward {
  template class IndexOf<uint8_t>;
  template class IndexOf<int32_t>;
  template class IndexOf<int64_t>;
}
