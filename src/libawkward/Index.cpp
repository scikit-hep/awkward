// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <type_traits>

#include "awkward/Slice.h"

#include "awkward/Index.h"

namespace awkward {
  template <typename T>
  const std::string IndexOf<T>::classname() const {
    if (std::is_same<T, int8_t>::value) {
      return "Index8";
    }
    else if (std::is_same<T, uint8_t>::value) {
      return "IndexU8";
    }
    else if (std::is_same<T, int32_t>::value) {
      return "Index32";
    }
    else if (std::is_same<T, uint32_t>::value) {
      return "IndexU32";
    }
    else if (std::is_same<T, int64_t>::value) {
      return "Index64";
    }
    else {
      return "UnrecognizedIndex";
    }
  }

  template <typename T>
  const std::string IndexOf<T>::tostring() const {
    return tostring_part("", "", "");
  }

  template <typename T>
  const std::string IndexOf<T>::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " i=\"[";
    if (length_ <= 10) {
      for (int64_t i = 0;  i < length_;  i++) {
        if (i != 0) {
          out << " ";
        }
        out << (int64_t)getitem_at_nowrap(i);
      }
    }
    else {
      for (int64_t i = 0;  i < 5;  i++) {
        if (i != 0) {
          out << " ";
        }
        out << (int64_t)getitem_at_nowrap(i);
      }
      out << " ... ";
      for (int64_t i = length_ - 5;  i < length_;  i++) {
        if (i != length_ - 5) {
          out << " ";
        }
        out << (int64_t)getitem_at_nowrap(i);
      }
    }
    out << "]\" offset=\"" << offset_ << "\" at=\"0x";
    out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
    return out.str();
  }

  template <typename T>
  T IndexOf<T>::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length_;
    }
    if (!(0 <= regular_at  &&  regular_at < length_)) {
      util::handle_error(failure("index out of range", kSliceNone, at), classname(), nullptr);
    }
    return getitem_at_nowrap(regular_at);
  }

  template <typename T>
  T IndexOf<T>::getitem_at_nowrap(int64_t at) const {
    assert(0 <= at  &&  at < length_);
    return ptr_.get()[(size_t)(offset_ + at)];
  }

  template <typename T>
  void IndexOf<T>::setitem_at_nowrap(int64_t at, T value) const {
    assert(0 <= at  &&  at < length_);
    ptr_.get()[(size_t)(offset_ + at)] = value;
  }

  template <typename T>
  IndexOf<T> IndexOf<T>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), length_);
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template <typename T>
  IndexOf<T> IndexOf<T>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    assert(0 <= start  &&  start < length_  &&  start <= stop  &&  stop <= length_);
    return IndexOf<T>(ptr_, offset_ + start*(start != stop), stop - start);
  }

  template <typename T>
  const std::shared_ptr<Index> IndexOf<T>::shallow_copy() const {
    return std::shared_ptr<Index>(new IndexOf<T>(ptr_, offset_, length_));
  }

  template class IndexOf<int8_t>;
  template class IndexOf<uint8_t>;
  template class IndexOf<int32_t>;
  template class IndexOf<uint32_t>;
  template class IndexOf<int64_t>;
}
