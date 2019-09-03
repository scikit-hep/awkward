// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"
#include "awkward/RawArray.h"

using namespace awkward;

template <typename T>
bool RawArrayOf<T>::isempty() const {
  return length_ == 0;
}

template <typename T>
bool RawArrayOf<T>::iscompact() const {
  return sizeof(T) == stride_;
}

template <typename T>
ssize_t RawArrayOf<T>::byteoffset() const {
  return (ssize_t)stride_*(ssize_t)offset_;
}

template <typename T>
void* RawArrayOf<T>::byteptr() const {
  return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset());
}

template <typename T>
ssize_t RawArrayOf<T>::bytelength() const {
  return (ssize_t)stride_*(ssize_t)length_;
}

template <typename T>
uint8_t RawArrayOf<T>::getbyte(ssize_t at) const {
  return *reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get()) + (ssize_t)stride_*(ssize_t)(offset_ + at));
}

template <typename T>
void RawArrayOf<T>::setid(const std::shared_ptr<Identity> id) {
  id_ = id;
}

template <typename T>
void RawArrayOf<T>::setid() {
  assert(!isscalar());
  Identity32* id32 = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
  std::shared_ptr<Identity> newid(id32);
  Error err = awkward_identity_new32(length(), id32->ptr().get());
  HANDLE_ERROR(err);
  setid(newid);
}

template <typename T>
const std::string RawArrayOf<T>::repr(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  out << indent << pre << "<RawArray of=\"" << typeid(T).name() << "\" length=\"" << length_ << "\" stride=\"" << stride_ << "\" data=\"" << std::hex << std::setw(2) << std::setfill('0');
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
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get());
  if (id_.get() == nullptr) {
    out << "\"/>" << post;
  }
  else {
    out << "\">\n";
    out << id_.get()->repr(indent + std::string("    "), "", "\n");
    out << indent << "</RawArray>" << post;
  }
  return out.str();
}

template <typename T>
const int64_t RawArrayOf<T>::length() const {
  return length_;
}

template <typename T>
const std::shared_ptr<Content> RawArrayOf<T>::shallow_copy() const {
  return std::shared_ptr<Content>(new RawArrayOf<T>(id_, ptr_, offset_, length_, stride_));
}

template <typename T>
const std::shared_ptr<Content> RawArrayOf<T>::get(int64_t at) const {
  return slice(at, at + 1);
}

template <typename T>
const std::shared_ptr<Content> RawArrayOf<T>::slice(int64_t start, int64_t stop) const {
  std::shared_ptr<Identity> id(nullptr);
  if (id_.get() != nullptr) {
    id = id_.get()->slice(start, stop);
  }
  return std::shared_ptr<Content>(new RawArrayOf<T>(id, ptr_, offset_ + start, stop - start, stride_));
}

template <typename T>
const std::pair<int64_t, int64_t> RawArrayOf<T>::minmax_depth() const {
  return std::pair<int64_t, int64_t>(1, 1);
}

template <typename T>
T* RawArrayOf<T>::borrow(int64_t at) const {
  return reinterpret_cast<T*>(reinterpret_cast<ssize_t>(ptr_.get()) + (ssize_t)stride_*(ssize_t)(offset_ + at));
}
