// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/Slice.h"

#include "awkward/Identity.h"

namespace awkward {
  std::atomic<Identity::Ref> numrefs{0};

  Identity::Ref Identity::newref() {
    return numrefs++;
  }

  template <typename T>
  const std::string IdentityOf<T>::classname() const {
    if (std::is_same<T, int32_t>::value) {
      return "Identity32";
    }
    else if (std::is_same<T, int64_t>::value) {
      return "Identity64";
    }
    else {
      return "UnrecognizedIdentity";
    }
  }

  template <typename T>
  const std::string IdentityOf<T>::location(int64_t where) const {
    std::stringstream out;
    int64_t fieldi = 0;
    int64_t widthi = 0;
    for (int64_t bothi = 0;  bothi < (int64_t)fieldloc_.size() + width_;  bothi++) {
      if (bothi != 0) {
        out << ", ";
      }
      if (fieldi < (int64_t)fieldloc_.size()  &&  fieldloc_[(size_t)fieldi].first == bothi) {
        out << "\"" << fieldloc_[(size_t)fieldi].second << "\"";
        fieldi++;
      }
      else {
        out << ptr_.get()[offset_ + where*width_ + widthi];
        widthi++;
      }
    }
    return out.str();
  }

  template <typename T>
  const std::shared_ptr<Identity> IdentityOf<T>::to64() const {
    if (std::is_same<T, int64_t>::value) {
      return shallow_copy();
    }
    else if (std::is_same<T, int32_t>::value) {
      Identity64* raw = new Identity64(ref_, fieldloc_, width_, length_);
      std::shared_ptr<Identity> out(raw);
      awkward_identity32_to_identity64(raw->ptr().get(), reinterpret_cast<int32_t*>(ptr_.get()), length_, width_);
      return out;
    }
  }

  template <typename T>
  const std::string IdentityOf<T>::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    std::stringstream out;
    std::string name = "Unrecognized Identity";
    if (std::is_same<T, int32_t>::value) {
      name = "Identity32";
    }
    else if (std::is_same<T, int64_t>::value) {
      name = "Identity64";
    }
    out << indent << pre << "<" << name << " ref=\"" << ref_ << "\" fieldloc=\"[";
    for (size_t i = 0;  i < fieldloc_.size();  i++) {
      if (i != 0) {
        out << " ";
      }
      out << "(" << fieldloc_[i].first << ", " << util::quote(fieldloc_[i].second, false) << ")";
    }
    out << "]\" width=\"" << width_ << "\" offset=\"" << offset_ << "\" length=\"" << length_ << "\" at=\"0x";
    out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
    return out.str();
  }

  template <typename T>
  const std::shared_ptr<Identity> IdentityOf<T>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    assert(0 <= start  &&  start < length_  &&  0 <= stop  &&  stop < length_);
    return std::shared_ptr<Identity>(new IdentityOf<T>(ref_, fieldloc_, offset_ + width_*start*(start != stop), width_, (stop - start), ptr_));
  }

  template <typename T>
  const std::shared_ptr<Identity> IdentityOf<T>::shallow_copy() const {
    return std::shared_ptr<Identity>(new IdentityOf<T>(ref_, fieldloc_, offset_, width_, length_, ptr_));
  }

  template <typename T>
  const std::shared_ptr<Identity> IdentityOf<T>::getitem_carry_64(const Index64& carry) const {
    IdentityOf<T>* rawout = new IdentityOf<T>(ref_, fieldloc_, width_, carry.length());
    std::shared_ptr<Identity> out(rawout);

    if (std::is_same<T, int32_t>::value) {
      struct Error err = awkward_identity32_getitem_carry_64(
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
      struct Error err = awkward_identity64_getitem_carry_64(
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
      throw std::runtime_error("unrecognized Identity specialization");
    }

    return out;
  }

  const std::string Identity::tostring() const {
    return tostring_part("", "", "");
  }

  template <typename T>
  const std::shared_ptr<Identity> IdentityOf<T>::withfieldloc(const FieldLoc& fieldloc) const {
    return std::shared_ptr<Identity>(new IdentityOf<T>(ref_, fieldloc, offset_, width_, length_, ptr_));
  }

  template <typename T>
  const std::vector<T> IdentityOf<T>::getitem_at(int64_t at) const {
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
  const std::vector<T> IdentityOf<T>::getitem_at_nowrap(int64_t at) const {
    assert(0 <= at  &&  at < length_);
    std::vector<T> out;
    for (size_t i = (size_t)(offset_ + at);  i < (size_t)(offset_ + at + width_);  i++) {
      out.push_back(ptr_.get()[i]);
    }
    return out;
  }

  template <typename T>
  const std::shared_ptr<Identity> IdentityOf<T>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), length_);
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template class IdentityOf<int32_t>;
  template class IdentityOf<int64_t>;
}
