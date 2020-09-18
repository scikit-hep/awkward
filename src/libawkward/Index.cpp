// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/Index.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/Index.cpp", line)

#include <cstring>
#include <iomanip>
#include <sstream>
#include <type_traits>

#define AWKWARD_INDEX_NO_EXTERN_TEMPLATE
#include "awkward/Slice.h"

#include "awkward/Index.h"

namespace awkward {
  Index::Form
  Index::str2form(const std::string& str) {
    if (strncmp(str.c_str(), "i8", str.length()) == 0) {
      return Index::Form::i8;
    }
    else if (strncmp(str.c_str(), "u8", str.length()) == 0) {
      return Index::Form::u8;
    }
    else if (strncmp(str.c_str(), "i32", str.length()) == 0) {
      return Index::Form::i32;
    }
    else if (strncmp(str.c_str(), "u32", str.length()) == 0) {
      return Index::Form::u32;
    }
    else if (strncmp(str.c_str(), "i64", str.length()) == 0) {
      return Index::Form::i64;
    }
    else {
      throw std::invalid_argument(
        std::string("unrecognized Index::Form: ") + str + FILENAME(__LINE__));
    }
  }

  const std::string
  Index::form2str(Index::Form form) {
    switch (form) {
    case Index::Form::i8:
      return "i8";
    case Index::Form::u8:
      return "u8";
    case Index::Form::i32:
      return "i32";
    case Index::Form::u32:
      return "u32";
    case Index::Form::i64:
      return "i64";
    default:
      throw std::runtime_error(
        std::string("unrecognized Index::Form") + FILENAME(__LINE__));
    }
  }

  Index::~Index() = default;

  template <typename T>
  IndexOf<T>::IndexOf(const std::shared_ptr<T>& ptr,
                      int64_t offset,
                      int64_t length,
                      kernel::lib ptr_lib)
      : ptr_(ptr)
      , ptr_lib_(ptr_lib)
      , offset_(offset)
      , length_(length) { }

  template <typename T>
  const std::shared_ptr<T>
  IndexOf<T>::ptr() const {
    return ptr_;
  }

  template <typename T>
  T*
  IndexOf<T>::data() const {
    return ptr_.get() + offset_;
  }

  template <typename T>
  IndexOf<T>::IndexOf(int64_t length, kernel::lib ptr_lib)
    : ptr_(kernel::malloc<T>(ptr_lib, length * (int64_t)sizeof(T)))
    , ptr_lib_(ptr_lib)
    , offset_(0)
    , length_(length) { }

  template <typename T>
  int64_t
  IndexOf<T>::offset() const {
    return offset_;
  }

  template <typename T>
  int64_t
  IndexOf<T>::length() const {
    return length_;
  }

  template <typename T>
  const std::string
  IndexOf<T>::classname() const {
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
  const std::string
  IndexOf<T>::tostring() const {
    return tostring_part("", "", "");
  }

  template <typename T>
  const std::string
  IndexOf<T>::tostring_part(const std::string& indent,
                            const std::string& pre,
                            const std::string& post) const {
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
      for (int64_t i = 0; i < 5; i++) {
        if (i != 0) {
          out << " ";
        }
        out << (int64_t) getitem_at_nowrap(i);
      }
      out << " ... ";
      for (int64_t i = length_ - 5; i < length_; i++) {
        if (i != length_ - 5) {
          out << " ";
        }
        out << (int64_t) getitem_at_nowrap(i);
      }
    }
    out << "]\" offset=\"" << offset_ << "\" length=\"" << length_
        << "\" at=\"0x" << std::hex << std::setw(12) << std::setfill('0')
        << reinterpret_cast<ssize_t>(ptr_.get());
    if (ptr_lib_ == kernel::lib::cpu) {
      out << "\"/>" << post;
    }
    else {
      out << "\">";
      out << kernel::lib_tostring(ptr_lib_,
                                  ptr_.get(),
                                  "\n" + indent + std::string("    "),
                                  "",
                                  "\n");
      out << indent << "</" << classname() << ">" << post;
    }
    return out.str();
  }

  template <typename T>
  Index::Form
  IndexOf<T>::form() const {
    if (std::is_same<T, int8_t>::value) {
      return Index::Form::i8;
    }
    else if (std::is_same<T, uint8_t>::value) {
      return Index::Form::u8;
    }
    else if (std::is_same<T, int32_t>::value) {
      return Index::Form::i32;
    }
    else if (std::is_same<T, uint32_t>::value) {
      return Index::Form::u32;
    }
    else if (std::is_same<T, int64_t>::value) {
      return Index::Form::i64;
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized Index specialization") + FILENAME(__LINE__));
    }
  }

  template <typename T>
  T
  IndexOf<T>::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length_;
    }
    if (!(0 <= regular_at  &&  regular_at < length_)) {
      util::handle_error(failure("index out of range", kSliceNone, at, FILENAME_C(__LINE__)),
                         classname(),
                         nullptr);
    }
    return getitem_at_nowrap(regular_at);
  }

  template <typename T>
  T
  IndexOf<T>::getitem_at_nowrap(int64_t at) const {
    return kernel::index_getitem_at_nowrap<T>(
      ptr_lib(),
      data(),
      at);
  }

  template <typename T>
  void
  IndexOf<T>::setitem_at_nowrap(int64_t at, T value) const {
    kernel::index_setitem_at_nowrap<T>(
      ptr_lib(),
      data(),
      at,
      value);
  }

  template <typename T>
  IndexOf<T>
  IndexOf<T>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), length_);
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template <typename T>
  IndexOf<T>
  IndexOf<T>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    if (!(0 <= start  &&  start < length_  &&  0 <= stop  &&  stop <= length_)
        &&  start != stop) {
      throw std::runtime_error(
        std::string("Index::getitem_range_nowrap with illegal start:stop for this length")
        + FILENAME(__LINE__));
    }
    return IndexOf<T>(ptr_, offset_ + start*(start != stop), stop - start, ptr_lib_);
  }

  template <typename T>
  void
  IndexOf<T>::nbytes_part(std::map<size_t, int64_t>& largest) const {
    size_t x = (size_t)ptr_.get();
    auto it = largest.find(x);
    if (it == largest.end()  ||  it->second < (int64_t)(sizeof(T))*length_) {
      largest[x] = (int64_t)(sizeof(T))*length_;
    }
  }

  template <typename T>
  const std::shared_ptr<Index>
  IndexOf<T>::shallow_copy() const {
    return std::make_shared<IndexOf<T>>(ptr_, offset_, length_, ptr_lib_);
  }

  template <>
  IndexOf<int64_t> IndexOf<int8_t>::to64() const {
    std::shared_ptr<int64_t> ptr(
      length_ == 0 ? nullptr : new int64_t[(size_t)length_],
      kernel::array_deleter<int64_t>());
    if (length_ != 0) {
      struct Error err = kernel::Index_to_Index64<int8_t>(
        kernel::lib::cpu,   // DERIVE
        ptr.get(),
        &ptr_.get()[(size_t)offset_],
        length_);
      util::handle_error(err);
    }
    return IndexOf<int64_t>(ptr, 0, length_, kernel::lib::cpu);   // DERIVE
  }

  template <>
  IndexOf<int64_t> IndexOf<uint8_t>::to64() const {
    std::shared_ptr<int64_t> ptr(
      length_ == 0 ? nullptr : new int64_t[(size_t)length_],
      kernel::array_deleter<int64_t>());
    if (length_ != 0) {
      struct Error err = kernel::Index_to_Index64<uint8_t>(
        kernel::lib::cpu,   // DERIVE
        ptr.get(),
        &ptr_.get()[(size_t)offset_],
        length_);
      util::handle_error(err);
    }
    return IndexOf<int64_t>(ptr, 0, length_, kernel::lib::cpu);   // DERIVE
  }

  template <>
  IndexOf<int64_t> IndexOf<int32_t>::to64() const {
    std::shared_ptr<int64_t> ptr(
      length_ == 0 ? nullptr : new int64_t[(size_t)length_],
      kernel::array_deleter<int64_t>());
    if (length_ != 0) {
      struct Error err = kernel::Index_to_Index64<int32_t>(
        kernel::lib::cpu,   // DERIVE
        ptr.get(),
        &ptr_.get()[(size_t)offset_],
        length_);
      util::handle_error(err);
    }
    return IndexOf<int64_t>(ptr, 0, length_, kernel::lib::cpu);   // DERIVE
  }

  template <>
  IndexOf<int64_t> IndexOf<uint32_t>::to64() const {
    std::shared_ptr<int64_t> ptr(
      length_ == 0 ? nullptr : new int64_t[(size_t)length_],
      kernel::array_deleter<int64_t>());
    if (length_ != 0) {
      struct Error err = kernel::Index_to_Index64<uint32_t>(
        kernel::lib::cpu,   // DERIVE
        ptr.get(),
        &ptr_.get()[(size_t)offset_],
        length_);
      util::handle_error(err);
    }
    return IndexOf<int64_t>(ptr, 0, length_, kernel::lib::cpu);   // DERIVE
  }

  template <>
  IndexOf<int64_t> IndexOf<int64_t>::to64() const {
    return IndexOf<int64_t>(ptr_, offset_, length_, ptr_lib_);
  }

  template <typename T>
  const IndexOf<T>
  IndexOf<T>::deep_copy() const {
    std::shared_ptr<T> ptr(
      length_ == 0 ? nullptr : new T[(size_t)length_],
      kernel::array_deleter<T>());
    if (length_ != 0) {
      memcpy(ptr.get(),
             &ptr_.get()[(size_t)offset_],
             sizeof(T)*((size_t)length_));
    }
    return IndexOf<T>(ptr, 0, length_, ptr_lib_);
  }

  template<typename T>
  kernel::lib IndexOf<T>::ptr_lib() const {
    return ptr_lib_;
  }

  template<typename T>
  const IndexOf<T>
  IndexOf<T>::copy_to(kernel::lib ptr_lib) const {
    if (ptr_lib == ptr_lib_) {
      return IndexOf<T>(ptr_, offset_, length_, ptr_lib);
    }
    else {
      int64_t bytelength = (offset_ + length_) * (int64_t)sizeof(T);
      std::shared_ptr<T> ptr = kernel::malloc<T>(ptr_lib, bytelength);
      Error err = kernel::copy_to(ptr_lib,
                                  ptr_lib_,
                                  ptr.get(),
                                  ptr_.get(),
                                  bytelength);
      util::handle_error(err);
      return IndexOf<T>(ptr, offset_, length_, ptr_lib);
    }
  }

  template class EXPORT_TEMPLATE_INST IndexOf<int8_t>;
  template class EXPORT_TEMPLATE_INST IndexOf<uint8_t>;
  template class EXPORT_TEMPLATE_INST IndexOf<int32_t>;
  template class EXPORT_TEMPLATE_INST IndexOf<uint32_t>;
  template class EXPORT_TEMPLATE_INST IndexOf<int64_t>;
}
