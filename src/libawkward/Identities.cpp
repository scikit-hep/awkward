// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <cstring>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/Slice.h"

#define AWKWARD_IDENTITIES_NO_EXTERN_TEMPLATE
#include "awkward/Identities.h"

namespace awkward {
  std::atomic<Identities::Ref> numrefs{0};

  Identities::Ref
  Identities::newref() {
    return numrefs++;
  }

  IdentitiesPtr
  Identities::none() {
    return IdentitiesPtr(nullptr);
  }

  Identities::Identities(const Ref ref,
                         const FieldLoc& fieldloc,
                         int64_t offset,
                         int64_t width,
                         int64_t length)
      : ref_(ref)
      , fieldloc_(fieldloc)
      , offset_(offset)
      , width_(width)
      , length_(length) { }

  Identities::~Identities() = default;

  const Identities::Ref
  Identities::ref() const {
    return ref_;
  }

  const Identities::FieldLoc
  Identities::fieldloc() const {
    return fieldloc_;
  }

  const int64_t
  Identities::offset() const {
    return offset_;
  }

  const int64_t
  Identities::width() const {
    return width_;
  }

  const int64_t
  Identities::length() const {
    return length_;
  }

  template <typename T>
  IdentitiesOf<T>::IdentitiesOf(const Ref ref,
                                const FieldLoc& fieldloc,
                                int64_t width,
                                int64_t length)
      : Identities(ref, fieldloc, 0, width, length)
      , ptr_(std::shared_ptr<T>(
          length*width == 0 ? nullptr : new T[(size_t)(length*width)],
          util::array_deleter<T>())) { }

  template <typename T>
  IdentitiesOf<T>::IdentitiesOf(const Ref ref,
                                const FieldLoc& fieldloc,
                                int64_t offset,
                                int64_t width,
                                int64_t length,
                                const std::shared_ptr<T> ptr)
      : Identities(ref, fieldloc, offset, width, length)
      , ptr_(ptr) { }

  template <typename T>
  const std::shared_ptr<T>
  IdentitiesOf<T>::ptr() const {
    return ptr_;
  }

  template <typename T>
  const std::string
  IdentitiesOf<T>::classname() const {
    if (std::is_same<T, int32_t>::value) {
      return "Identities32";
    }
    else if (std::is_same<T, int64_t>::value) {
      return "Identities64";
    }
    else {
      return "UnrecognizedIdentities";
    }
  }

  template <typename T>
  const std::string
  IdentitiesOf<T>::identity_at(int64_t at) const {
    std::stringstream out;
    for (int64_t i = 0;  i < width_;  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << ptr_.get()[offset_ + at*width_ + i];
      for (auto pair : fieldloc_) {
        if (pair.first == i) {
          out << ", " << util::quote(pair.second, true);
        }
      }
    }
    return out.str();
  }

  template <typename T>
  const IdentitiesPtr
  IdentitiesOf<T>::to64() const {
    if (std::is_same<T, int64_t>::value) {
      return shallow_copy();
    }
    else if (std::is_same<T, int32_t>::value) {
      IdentitiesPtr out = std::make_shared<Identities64>(ref_,
                                                         fieldloc_,
                                                         width_,
                                                         length_);
      Identities64* raw = reinterpret_cast<Identities64*>(out.get());
      kernel::identities_to_identities64<int32_t>(
        raw->ptr().get(),
        reinterpret_cast<int32_t*>(ptr_.get()),
        length_,
        width_);
      return out;
    }
  }

  template <typename T>
  const std::string
  IdentitiesOf<T>::tostring_part(const std::string& indent,
                                 const std::string& pre,
                                 const std::string& post) const {
    std::stringstream out;
    std::string name = "Unrecognized Identities";
    if (std::is_same<T, int32_t>::value) {
      name = "Identities32";
    }
    else if (std::is_same<T, int64_t>::value) {
      name = "Identities64";
    }
    out << indent << pre << "<" << name << " ref=\"" << ref_
        << "\" fieldloc=\"[";
    for (size_t i = 0;  i < fieldloc_.size();  i++) {
      if (i != 0) {
        out << " ";
      }
      out << "(" << fieldloc_[i].first << ", "
          << util::quote(fieldloc_[i].second, false) << ")";
    }
    out << "]\" width=\"" << width_ << "\" offset=\"" << offset_
        << "\" length=\"" << length_ << "\" at=\"0x";
    out << std::hex << std::setw(12) << std::setfill('0')
        << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
    return out.str();
  }

  template <typename T>
  const IdentitiesPtr
  IdentitiesOf<T>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    if (!(0 <= start  &&  start < length_  &&  0 <= stop  &&  stop <= length_)
        &&  start != stop) {
      throw std::runtime_error(
        "Identities::getitem_range_nowrap with illegal start:stop "
        "for this length");
    }
    return std::make_shared<IdentitiesOf<T>>(
      ref_,
      fieldloc_,
      offset_ + width_*start*(start != stop),
      width_,
      (stop - start),
      ptr_);
  }

  template <typename T>
  void
  IdentitiesOf<T>::nbytes_part(std::map<size_t, int64_t>& largest) const {
    size_t x = (size_t)ptr_.get();
    auto it = largest.find(x);
    if (it == largest.end()  ||
        it->second < (int64_t)(sizeof(T)*length_*width_)) {
      largest[x] = (int64_t)(sizeof(T)*length_*width_);
    }
  }

  template <typename T>
  const IdentitiesPtr
  IdentitiesOf<T>::shallow_copy() const {
    return std::make_shared<IdentitiesOf<T>>(ref_,
                                             fieldloc_,
                                             offset_,
                                             width_,
                                             length_,
                                             ptr_);
  }

  template <typename T>
  const IdentitiesPtr
  IdentitiesOf<T>::deep_copy() const {
    std::shared_ptr<T> ptr(length_ == 0 ? nullptr : new T[(size_t)length_],
                           util::array_deleter<T>());
    if (length_ != 0) {
      memcpy(ptr.get(),
             &ptr_.get()[(size_t)offset_],
             sizeof(T)*((size_t)length_));
    }
    return std::make_shared<IdentitiesOf<T>>(ref_,
                                             fieldloc_,
                                             0,
                                             width_,
                                             length_,
                                             ptr);
  }

  template <typename T>
  const IdentitiesPtr
  IdentitiesOf<T>::getitem_carry_64(const Index64& carry) const {
    IdentitiesPtr out = std::make_shared<IdentitiesOf<T>>(ref_,
                                                          fieldloc_,
                                                          width_,
                                                          carry.length());
    IdentitiesOf<T>* rawout = reinterpret_cast<IdentitiesOf<T>*>(out.get());

    if (std::is_same<T, int32_t>::value) {
      struct Error err = kernel::identities_getitem_carry_64<int32_t>(
        reinterpret_cast<int32_t*>(rawout->ptr().get()),
        reinterpret_cast<int32_t*>(ptr_.get()),
        carry.ptr().get(),
        carry.length(),
        offset_,
        width_,
        length_);
      util::handle_error(err, classname(), nullptr);
    }
    else if (std::is_same<T, int64_t>::value) {
      struct Error err = kernel::identities_getitem_carry_64<int64_t>(
        reinterpret_cast<int64_t*>(rawout->ptr().get()),
        reinterpret_cast<int64_t*>(ptr_.get()),
        carry.ptr().get(),
        carry.length(),
        offset_,
        width_,
        length_);
      util::handle_error(err, classname(), nullptr);
    }
    else {
      throw std::runtime_error("unrecognized Identities specialization");
    }

    return out;
  }

  const std::string
  Identities::tostring() const {
    return tostring_part("", "", "");
  }

  template <typename T>
  const IdentitiesPtr
  IdentitiesOf<T>::withfieldloc(const FieldLoc& fieldloc) const {
    return std::make_shared<IdentitiesOf<T>>(ref_,
                                             fieldloc,
                                             offset_,
                                             width_,
                                             length_,
                                             ptr_);
  }

  template <typename T>
  int64_t
  IdentitiesOf<T>::value(int64_t row, int64_t col) const {
    return (int64_t)ptr_.get()[offset_ + row*width_ + col];
  }

  template <typename T>
  const std::vector<T>
  IdentitiesOf<T>::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length_;
    }
    if (!(0 <= regular_at  &&  regular_at < length_)) {
      util::handle_error(
        failure("index out of range", kSliceNone, at), classname(), nullptr);
    }
    return getitem_at_nowrap(regular_at);
  }

  template <typename T>
  const std::vector<T>
  IdentitiesOf<T>::getitem_at_nowrap(int64_t at) const {
    if (!(0 <= at  &&  at < length_)) {
      throw std::runtime_error(
        "Identities::getitem_at_nowrap with illegal index for this length");
    }
    std::vector<T> out;
    for (size_t i = (size_t)(offset_ + at);
         i < (size_t)(offset_ + at + width_);
         i++) {
      out.push_back(ptr_.get()[i]);
    }
    return out;
  }

  template <typename T>
  const IdentitiesPtr
  IdentitiesOf<T>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), length_);
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template class EXPORT_SYMBOL IdentitiesOf<int32_t>;
  template class EXPORT_SYMBOL IdentitiesOf<int64_t>;
}
