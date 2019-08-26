// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

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

byte NumpyArray::getbyte(ssize_t at) const {
  return *reinterpret_cast<byte*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_ + at);
}

const std::string NumpyArray::repr(const std::string indent, const std::string pre, const std::string post) const {
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
  out << "data=\"" << std::hex << std::setw(2) << std::setfill('0');
  ssize_t len = bytelength();
  if (len <= 32) {
    for (ssize_t i = 0;  i < len;  i++) {
      if (i != 0  &&  i % 4 == 0) {
        out << " ";
      }
      out << int(getbyte(i));
    }
  }
  else {
    for (ssize_t i = 0;  i < 16;  i++) {
      if (i != 0  &&  i % 4 == 0) {
        out << " ";
      }
      out << int(getbyte(i));
    }
    out << " ... ";
    for (ssize_t i = len - 16;  i < len;  i++) {
      if (i != len - 16  &&  i % 4 == 0) {
        out << " ";
      }
      out << int(getbyte(i));
    }
  }
  out << "\" at=\"0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
  return out.str();
}

IndexType NumpyArray::length() const {
  if (isscalar()) {
    return -1;
  }
  else {
    return shape_[shape_.size() - 1];
  }
}

std::shared_ptr<Content> NumpyArray::shallow_copy() const {
  return std::shared_ptr<Content>(new NumpyArray(ptr_, shape_, strides_, byteoffset_, itemsize_, format_));
}

std::shared_ptr<Content> NumpyArray::get(AtType at) const {
  assert(!isscalar());
  ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)at);
  const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
  const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
  return std::shared_ptr<Content>(new NumpyArray(ptr_, shape, strides, byteoffset, itemsize_, format_));
}

std::shared_ptr<Content> NumpyArray::slice(AtType start, AtType stop) const {
  assert(!isscalar());
  ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)start);
  std::vector<ssize_t> shape;
  shape.push_back((ssize_t)(stop - start));
  shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
  return std::shared_ptr<Content>(new NumpyArray(ptr_, shape, strides_, byteoffset, itemsize_, format_));
}
