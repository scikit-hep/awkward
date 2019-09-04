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
  std::shared_ptr<Identity> id(nullptr);
  if (id_.get() != nullptr) {
    id = id_.get()->slice(at, at + 1);
  }
  return std::shared_ptr<Content>(new NumpyArray(id, ptr_, shape, strides, byteoffset, itemsize_, format_));
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

const std::shared_ptr<Content> getitem_next(SliceItem& head, Slice& tail, std::shared_ptr<Index> carry) {
  return std::shared_ptr<Content>(nullptr);
}
