// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"
#include "awkward/NumpyArray.h"

using namespace awkward;

ssize_t NumpyArray::ndim() const {
  return shape_.size();
}

bool NumpyArray::isscalar() const {
  return ndim() == 0;
}

bool NumpyArray::isempty() const {
  for (auto x : shape_) {
    if (x == 0) return true;
  }
  return false;  // false for isscalar(), too
}

bool NumpyArray::iscompact() const {
  ssize_t x = itemsize_;
  for (ssize_t i = ndim() - 1;  i >= 0;  i--) {
    if (x != strides_[i]) return false;
    x *= shape_[i];
  }
  return true;  // true for isscalar(), too
}

void* NumpyArray::byteptr() const {
  return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_);
}

ssize_t NumpyArray::bytelength() const {
  if (isscalar()) {
    return itemsize_;
  }
  else {
    return shape_[0]*strides_[0];
  }
}

uint8_t NumpyArray::getbyte(ssize_t at) const {
  return *reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_ + at);
}

void NumpyArray::setid(const std::shared_ptr<Identity> id) {
  id_ = id;
}

void NumpyArray::setid() {
  assert(!isscalar());
  Identity32* id32 = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
  std::shared_ptr<Identity> newid(id32);
  Error err = awkward_identity_new32(length(), id32->ptr().get());
  HANDLE_ERROR(err);
  setid(newid);
}

template <typename T>
void tostring_as(std::stringstream& out, T* ptr, int64_t length) {
  if (length <= 10) {
    for (int64_t i = 0;  i < length;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << ptr[i];
    }
  }
  else {
    for (int64_t i = 0;  i < 5;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << ptr[i];
    }
    out << " ... ";
    for (int64_t i = length - 5;  i < length;  i++) {
      if (i != length - 5) {
        out << " ";
      }
      out << ptr[i];
    }
  }
}

const std::string NumpyArray::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
  assert(!isscalar());
  std::stringstream out;
  out << indent << pre << "<NumpyArray format=\"" << format_ << "\" shape=\"";
  for (ssize_t i = 0;  i < ndim();  i++) {
    if (i != 0) {
      out << " ";
    }
    out << shape_[i];
  }
  out << "\" ";
  if (!iscompact()) {
    out << "strides=\"";
    for (ssize_t i = 0;  i < ndim();  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << strides_[i];
    }
    out << "\" ";
  }
  out << "data=\"";
#ifdef _MSC_VER
  if (ndim() == 1  &&  format_.compare("l") == 0) {
#else
  if (ndim() == 1  &&  format_.compare("i") == 0) {
#endif
    tostring_as<int32_t>(out, reinterpret_cast<int32_t*>(byteptr()), length());
  }
#ifdef _MSC_VER
  else if (ndim() == 1  &&  format_.compare("q") == 0) {
#else
  else if (ndim() == 1  &&  format_.compare("l") == 0) {
#endif
    tostring_as<int64_t>(out, reinterpret_cast<int64_t*>(byteptr()), length());
  }
  else if (ndim() == 1  &&  format_.compare("f") == 0) {
    tostring_as<float>(out, reinterpret_cast<float*>(byteptr()), length());
  }
  else if (ndim() == 1  &&  format_.compare("d") == 0) {
    tostring_as<double>(out, reinterpret_cast<double*>(byteptr()), length());
  }
  else {
    ssize_t len = bytelength();
    if (len <= 32) {
      for (ssize_t i = 0;  i < len;  i++) {
        if (i != 0  &&  i % 4 == 0) {
          out << " ";
        }
        out << std::hex << std::setw(2) << std::setfill('0') << int(getbyte(i));
      }
    }
    else {
      for (ssize_t i = 0;  i < 16;  i++) {
        if (i != 0  &&  i % 4 == 0) {
          out << " ";
        }
        out << std::hex << std::setw(2) << std::setfill('0') << int(getbyte(i));
      }
      out << " ... ";
      for (ssize_t i = len - 16;  i < len;  i++) {
        if (i != len - 16  &&  i % 4 == 0) {
          out << " ";
        }
        out << std::hex << std::setw(2) << std::setfill('0') << int(getbyte(i));
      }
    }
  }
  out << "\" at=\"0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get());
  if (id_.get() == nullptr) {
    out << "\"/>" << post;
  }
  else {
    out << "\">\n";
    out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
    out << indent << "</NumpyArray>" << post;
  }
  return out.str();
}

int64_t NumpyArray::length() const {
  if (isscalar()) {
    return -1;
  }
  else {
    return (int64_t)shape_[0];
  }
}

const std::shared_ptr<Content> NumpyArray::shallow_copy() const {
  return std::shared_ptr<Content>(new NumpyArray(id_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_));
}

const std::shared_ptr<Content> NumpyArray::get(int64_t at) const {
  assert(!isscalar());
  ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)at);
  const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
  const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
  return std::shared_ptr<Content>(new NumpyArray(Identity::none(), ptr_, shape, strides, byteoffset, itemsize_, format_));
}

const std::shared_ptr<Content> NumpyArray::slice(int64_t start, int64_t stop) const {
  assert(!isscalar());
  ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)start);
  std::vector<ssize_t> shape;
  shape.push_back((ssize_t)(stop - start));
  shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
  std::shared_ptr<Identity> id(nullptr);
  if (id_.get() != nullptr) {
    id = id_.get()->slice(start, stop);
  }
  return std::shared_ptr<Content>(new NumpyArray(id, ptr_, shape, strides_, byteoffset, itemsize_, format_));
}

const std::pair<int64_t, int64_t> NumpyArray::minmax_depth() const {
  return std::pair<int64_t, int64_t>((int64_t)shape_.size(), (int64_t)shape_.size());
}

#include <iostream>

std::string stupid(std::vector<ssize_t> v) {
  std::string out("[");
  for (size_t i = 0;  i < v.size();  i++) {
    if (i != 0) {
      out += std::string(" ");
    }
    out += std::to_string(v[i]);
  }
  return out + std::string("]");
}

const std::vector<ssize_t> shape2strides(const std::vector<ssize_t>& shape, ssize_t itemsize) {
  std::vector<ssize_t> out;
  for (auto dim = shape.rbegin();  dim != shape.rend();  ++dim) {
    out.insert(out.begin(), itemsize);
    itemsize *= *dim;
  }
  return out;
}

const std::shared_ptr<Content> NumpyArray::getitem(const Slice& slice) const {
  std::shared_ptr<SliceItem> head = slice.head();
  Slice tail = slice.tail();
  return getitem_next(head, tail);
}

const std::shared_ptr<Content> NumpyArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail) const {
  if (head.get() == nullptr) {
    return shallow_copy();
  }

  else if (SliceAt* h = dynamic_cast<SliceAt*>(head.get())) {
    if (isscalar()) {
      throw std::invalid_argument("too many dimensions in index for this array");
    }
    std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
    ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)h->at());
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    std::shared_ptr<Content> next(new NumpyArray(Identity::none(), ptr_, shape, shape2strides(shape, itemsize_), byteoffset, itemsize_, format_));
    return next.get()->getitem_next(nexthead, nexttail);
  }

  else if (SliceStartStop* h = dynamic_cast<SliceStartStop*>(head.get())) {
    if (isscalar()) {
      throw std::invalid_argument("too many dimensions in index for this array");
    }
    std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
    ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)h->start());
    if (tail.length() == 0) {
      shape.insert(shape.begin(), (ssize_t)(h->stop() - h->start()));
      return std::shared_ptr<Content>(new NumpyArray(Identity::none(), ptr_, shape, shape2strides(shape, itemsize_), byteoffset, itemsize_, format_));
    }
    else {
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      Index64 nextcarry(h->stop() - h->start());
      int64_t step = (shape_.size() == 1 ? 1 : (int64_t)shape_[1]);
      for (int64_t i = 0;  i < nextcarry.length();  i++) {
        nextcarry.ptr().get()[i] = step*i;
      }
      std::shared_ptr<Content> next(new NumpyArray(Identity::none(), ptr_, shape, shape2strides(shape, itemsize_), byteoffset, itemsize_, format_));
      return next.get()->getitem_next(nexthead, nexttail, nextcarry);
    }
  }

  else {
    assert(false);
  }
}

const std::shared_ptr<Content> NumpyArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& carry) const {
  if (head.get() == nullptr) {
    std::vector<ssize_t> shape = { (ssize_t)carry.length() };
    int64_t skip = itemsize_;
    if (shape_.size() != 0) {
      shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
      skip = strides_[0];
    }
    uint8_t* src = reinterpret_cast<uint8_t*>(ptr_.get());
    uint8_t* dst = new uint8_t[(size_t)(carry.length()*skip)];

    for (int64_t i = 0;  i < carry.length();  i++) {
      std::memcpy(&dst[(size_t)(skip*i)], &src[(size_t)(byteoffset_ + skip*carry.get(i))], skip);
    }
    std::shared_ptr<uint8_t> ptr(dst, awkward::util::array_deleter<uint8_t>());
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), ptr, shape, shape2strides(shape, itemsize_), 0, itemsize_, format_));
  }

  else if (SliceAt* h = dynamic_cast<SliceAt*>(head.get())) {
    if (isscalar()) {
      throw std::invalid_argument("too many dimensions in index for this array");
    }
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    Index64 nextcarry(carry.length());
    for (int64_t i = 0;  i < nextcarry.length();  i++) {
      nextcarry.ptr().get()[i] = carry.ptr().get()[i] + h->at();
    }
    std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
    std::shared_ptr<Content> next(new NumpyArray(Identity::none(), ptr_, shape, shape2strides(shape, itemsize_), byteoffset_, itemsize_, format_));
    return next.get()->getitem_next(nexthead, nexttail, nextcarry);
  }

  else if (SliceStartStop* h = dynamic_cast<SliceStartStop*>(head.get())) {
    std::cout << "SliceStartStop" << std::endl;

    throw std::runtime_error("FIXME");
  }

  else {
    assert(false);
  }
}




// const std::shared_ptr<Content> NumpyArray::getitem(Slice& slice) const {
//   std::shared_ptr<SliceItem> head = slice.head();
//   Slice tail = slice.tail();
//   return getitem_next(head, tail, std::shared_ptr<Index>(nullptr));
// }
//
// const std::shared_ptr<Content> NumpyArray::getitem_next(std::shared_ptr<SliceItem> head, Slice& tail, std::shared_ptr<Index> carry) const {
//   assert(!isscalar());
//   if (head.get() == nullptr) {
//     return shallow_copy();
//   }
//   else if (SliceAt* x = dynamic_cast<SliceAt*>(head.get())) {
//     if (carry.get() == nullptr) {
//       int64_t at = x->at();
//       if (at < 0) {
//         at += length();
//       }
//       if (at < 0  ||  at >= length()) {
//         throw std::invalid_argument("integer index out of range");
//       }
//       ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)at);
//       const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
//       const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
//       std::shared_ptr<Content> next = std::shared_ptr<Content>(new NumpyArray(Identity::none(), ptr_, shape, strides, byteoffset, itemsize_, format_));
//       std::shared_ptr<SliceItem> nexthead = tail.head();
//       Slice nexttail = tail.tail();
//       return next.get()->getitem_next(nexthead, nexttail, std::shared_ptr<Index>(nullptr));
//     }
//     else if (Index32* carry32 = dynamic_cast<Index32*>(carry.get())) {
//       ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)at);
//       std::vector<ssize_t> shape;
//       shape.push_back(carry32->length());
//       shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
//       std::vector<ssize_t> strides;
//       if (strides_.size() == 1) {
//         strides.push_back(carry32->length()*itemsize_);
//       }
//       else {
//         strides.push_back(carry32->length()*strides_[1]);
//       }
//       uint8_t* ptr = new uint8_t[(size_t)length];
//
//       std::shared_ptr<Content> next = std::shared_ptr<Content>(new NumpyArray(Identity::none(), std::shared_ptr<void>(ptr, awkward::util::array_deleter<void>()), ...));
//
//
//
//     }
//   }
//   else if (SliceStartStop* x = dynamic_cast<SliceStartStop*>(head.get())) {
//     int64_t start = x->start();
//     if (start == Slice::none()) {
//       start = 0;
//     }
//     if (start < 0) {
//       start += length();
//     }
//     if (start < 0) {
//       start = 0;
//     }
//     if (start > length()) {
//       start = length();
//     }
//     int64_t stop = x->stop();
//     if (stop == Slice::none()) {
//       stop = length();
//     }
//     if (stop < 0) {
//       stop += length();
//     }
//     if (stop < 0) {
//       stop = 0;
//     }
//     if (stop > length()) {
//       stop = length();
//     }
//     // ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)start);
//     // std::vector<ssize_t> shape;
//     // shape.push_back((ssize_t)(stop - start));
//     // shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
//     // std::shared_ptr<Identity> id(nullptr);
//     // if (id_.get() != nullptr) {
//     //   id = id_.get()->slice(start, stop);
//     // }
//     // return std::shared_ptr<Content>(new NumpyArray(id, ptr_, shape, strides_, byteoffset, itemsize_, format_));
//
//     std::vector<ssize_t> shape;
//     for (size_t i = 1;  i < shape_.size();  i++) {
//       if (i == 1) {
//         shape.push_back(shape_[i] * (stop - start));
//       }
//       else {
//         shape.push_back(shape_[i]);
//       }
//     }
//
//     std::cout << "before " << stupid(shape_) << " after " << stupid(shape) << std::endl;
//
//     std::vector<ssize_t> strides;
//     strides.insert(strides.end(), strides_.begin() + 1, strides_.end());
//     ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)start);
//
//     Index32 nextcarry(stop - start);
//     int32_t* carryptr = nextcarry.ptr().get();
//     for (int64_t i = 0;  i < (stop - start);  i++) {
//       carryptr[i] = shape_[1]*i;
//       std::cout << "and " << carryptr[i] << std::endl;
//     }
//     std::shared_ptr<Content> next = std::shared_ptr<Content>(new NumpyArray(Identity::none(), ptr_, shape, strides, byteoffset, itemsize_, format_));
//
//     std::shared_ptr<SliceItem> nexthead = tail.head();
//     Slice nexttail = tail.tail();
//     return next.get()->getitem_next(nexthead, nexttail, nextcarry.shallow_copy());
//     return next;
//
//   }
//   else if (SliceStartStopStep* x = dynamic_cast<SliceStartStopStep*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceByteMask* x = dynamic_cast<SliceByteMask*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceIndex32* x = dynamic_cast<SliceIndex32*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceIndex64* x = dynamic_cast<SliceIndex64*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceEllipsis* x = dynamic_cast<SliceEllipsis*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceNewAxis* x = dynamic_cast<SliceNewAxis*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else {
//     assert(false);
//   }
// }
