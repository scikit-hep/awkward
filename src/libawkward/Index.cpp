// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>
#include <iomanip>
#include <sstream>
#include <type_traits>

#include "awkward/Slice.h"

#include "awkward/Index.h"

namespace awkward {
  template <typename T>
  IndexOf<T>::IndexOf(int64_t length)
      : ptr_(std::shared_ptr<T>(length == 0 ? nullptr : new T[(size_t)length], util::array_deleter<T>()))
      , offset_(0)
      , length_(length) { }

  template <typename T>
  IndexOf<T>::IndexOf(const std::shared_ptr<T>& ptr, int64_t offset, int64_t length)
      : ptr_(ptr)
      , offset_(offset)
      , length_(length) { }

  template <typename T>
  const std::shared_ptr<T> IndexOf<T>::ptr() const {
    return ptr_;
  }

  template <typename T>
  int64_t IndexOf<T>::offset() const {
    return offset_;
  }

  template <typename T>
  int64_t IndexOf<T>::length() const {
    return length_;
  }

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
  const std::string IndexOf<T>::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
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
    out << "]\" offset=\"" << offset_ << "\" length=\"" << length_ << "\" at=\"0x";
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
    return util::awkward_index_getitem_at_nowrap<T>(ptr_.get(), offset_, at);
  }

  template <typename T>
  void IndexOf<T>::setitem_at_nowrap(int64_t at, T value) const {
    util::awkward_index_setitem_at_nowrap<T>(ptr_.get(), offset_, at, value);
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
    if (!(0 <= start  &&  start < length_  &&  0 <= stop  &&  stop <= length_)  &&  start != stop) {
      throw std::runtime_error("Index::getitem_range_nowrap with illegal start:stop for this length");
    }
    return IndexOf<T>(ptr_, offset_ + start*(start != stop), stop - start);
  }

  template <typename T>
  void IndexOf<T>::nbytes_part(std::map<size_t, int64_t>& largest) const {
    size_t x = (size_t)ptr_.get();
    auto it = largest.find(x);
    if (it == largest.end()  ||  it->second < (int64_t)(sizeof(T)*length_)) {
      largest[x] = (int64_t)(sizeof(T)*length_);
    }
  }

  template <typename T>
  const std::shared_ptr<Index> IndexOf<T>::shallow_copy() const {
    return std::make_shared<IndexOf<T>>(ptr_, offset_, length_);
  }

  template <>
  IndexOf<int64_t> IndexOf<int8_t>::to64() const {
    std::shared_ptr<int64_t> ptr(length_ == 0 ? nullptr : new int64_t[(size_t)length_], util::array_deleter<int64_t>());
    if (length_ != 0) {
      awkward_index8_to_index64(ptr.get(), &ptr_.get()[(size_t)offset_], length_);
    }
    return IndexOf<int64_t>(ptr, 0, length_);
  }

  template <>
  IndexOf<int64_t> IndexOf<uint8_t>::to64() const {
    std::shared_ptr<int64_t> ptr(length_ == 0 ? nullptr : new int64_t[(size_t)length_], util::array_deleter<int64_t>());
    if (length_ != 0) {
      awkward_indexU8_to_index64(ptr.get(), &ptr_.get()[(size_t)offset_], length_);
    }
    return IndexOf<int64_t>(ptr, 0, length_);
  }

  template <>
  IndexOf<int64_t> IndexOf<int32_t>::to64() const {
    std::shared_ptr<int64_t> ptr(length_ == 0 ? nullptr : new int64_t[(size_t)length_], util::array_deleter<int64_t>());
    if (length_ != 0) {
      awkward_index32_to_index64(ptr.get(), &ptr_.get()[(size_t)offset_], length_);
    }
    return IndexOf<int64_t>(ptr, 0, length_);
  }

  template <>
  IndexOf<int64_t> IndexOf<uint32_t>::to64() const {
    std::shared_ptr<int64_t> ptr(length_ == 0 ? nullptr : new int64_t[(size_t)length_], util::array_deleter<int64_t>());
    if (length_ != 0) {
      awkward_indexU32_to_index64(ptr.get(), &ptr_.get()[(size_t)offset_], length_);
    }
    return IndexOf<int64_t>(ptr, 0, length_);
  }

  template <>
  IndexOf<int64_t> IndexOf<int64_t>::to64() const {
    return IndexOf<int64_t>(ptr_, offset_, length_);
  }

  template <typename T>
  const IndexOf<T> IndexOf<T>::deep_copy() const {
    std::shared_ptr<T> ptr(length_ == 0 ? nullptr : new T[(size_t)length_], util::array_deleter<T>());
    if (length_ != 0) {
      memcpy(ptr.get(), &ptr_.get()[(size_t)offset_], sizeof(T)*((size_t)length_));
    }
    return IndexOf<T>(ptr, 0, length_);
  }

  template class IndexOf<int8_t>;
  template class IndexOf<uint8_t>;
  template class IndexOf<int32_t>;
  template class IndexOf<uint32_t>;
  template class IndexOf<int64_t>;
}
