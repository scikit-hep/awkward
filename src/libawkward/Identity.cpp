// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"

#include "awkward/Identity.h"

using namespace awkward;

std::atomic<Identity::Ref> numrefs{0};

Identity::Ref Identity::newref() {
  return numrefs++;
}

template <typename T>
const std::shared_ptr<Identity> IdentityOf<T>::to64() const {
  if (std::is_same<T, int64_t>::value) {
    return shallow_copy();
  }
  else if (std::is_same<T, int32_t>::value) {
    Identity64* raw = new Identity64(ref_, fieldloc_, width_, length_);
    std::shared_ptr<Identity> out(raw);
    awkward_identity32_to_identity64(raw->ptr().get(), reinterpret_cast<int32_t*>(ptr_.get()), length_);
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
  out << indent << pre << "<" << name << " ref=\"" << ref() << "\" fieldloc=\"[";
  for (size_t i = 0;  i < fieldloc().size();  i++) {
    if (i != 0) {
      out << " ";
    }
    out << "(" << fieldloc()[i].first << ", '" << fieldloc()[i].second << "')";
  }
  out << "]\" width=\"" << width() << "\" length=\"" << length() << "\" at=\"0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
  return out.str();
}

template <typename T>
const std::string IdentityOf<T>::tostring() const {
  return tostring_part("", "", "");
}

template <typename T>
const std::shared_ptr<Identity> IdentityOf<T>::getitem_range(int64_t start, int64_t stop) const {
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

  Error assign_err = kNoError;
  if (std::is_same<T, int32_t>::value) {
    assign_err = awkward_identity32_getitem_carry_64(
      reinterpret_cast<int32_t*>(rawout->ptr().get()),
      reinterpret_cast<int32_t*>(ptr_.get()),
      carry.ptr().get(),
      carry.length(),
      offset_,
      width_,
      length_);
  }
  else if (std::is_same<T, int64_t>::value) {
    assign_err = awkward_identity64_getitem_carry_64(
      reinterpret_cast<int64_t*>(rawout->ptr().get()),
      reinterpret_cast<int64_t*>(ptr_.get()),
      carry.ptr().get(),
      carry.length(),
      offset_,
      width_,
      length_);
  }
  else {
    throw std::runtime_error("unrecognized Identity specialization");
  }
  HANDLE_ERROR(assign_err)

  return out;
}

template <typename T>
const std::vector<T> IdentityOf<T>::get(int64_t at) const {
  assert(0 <= at < length_);
  std::vector<T> out;
  for (size_t i = (size_t)(offset() + at);  i < (size_t)(offset() + at + width());  i++) {
    out.push_back(ptr_.get()[i]);
  }
  return out;
}

namespace awkward {
  template class IdentityOf<int32_t>;
  template class IdentityOf<int64_t>;
}
