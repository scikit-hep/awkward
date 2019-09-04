// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RAWARRAY_H_
#define AWKWARD_RAWARRAY_H_

#include <cassert>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <typeinfo>

#include "awkward/cpu-kernels/util.h"
#include "awkward/cpu-kernels/identity.h"
#include "awkward/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  template <typename T>
  class RawArrayOf: public Content {
  public:
    RawArrayOf<T>(const std::shared_ptr<Identity> id, const std::shared_ptr<T> ptr, const int64_t offset, const int64_t length, const int64_t stride)
        : id_(id)
        , ptr_(ptr)
        , offset_(offset)
        , length_(length)
        , stride_(stride) {
          assert(sizeof(T) <= stride);
        }

    RawArrayOf<T>(const std::shared_ptr<Identity> id, const std::shared_ptr<T> ptr, const int64_t length)
        : id_(id)
        , ptr_(ptr)
        , offset_(0)
        , length_(length)
        , stride_(sizeof(T)) { }

    RawArrayOf<T>(const std::shared_ptr<Identity> id, const int64_t length)
        : id_(id)
        , ptr_(std::shared_ptr<T>(new T[(size_t)length], awkward::util::array_deleter<T>()))
        , offset_(0)
        , length_(length)
        , stride_(sizeof(T)) { }

    const std::shared_ptr<T> ptr() const { return ptr_; }
    const int64_t offset() const { return offset_; }
    const int64_t stride() const { return stride_; }

    bool isempty() const { return length_ == 0; }
    bool iscompact() const { return sizeof(T) == stride_; }
    ssize_t byteoffset() const { return (ssize_t)stride_*(ssize_t)offset_; }
    void* byteptr() const { return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset()); }
    ssize_t bytelength() const { return (ssize_t)stride_*(ssize_t)length_; }
    uint8_t getbyte(ssize_t at) const { return *reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get()) + (ssize_t)(byteoffset() + at)); }

    virtual const std::shared_ptr<Identity> id() const { return id_; }
    virtual void setid() {
      Identity32* id32 = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
      std::shared_ptr<Identity> newid(id32);
      Error err = awkward_identity_new32(length(), id32->ptr().get());
      HANDLE_ERROR(err);
      setid(newid);
    }
    virtual void setid(const std::shared_ptr<Identity> id) { id_ = id; }
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const {
      std::stringstream out;
      out << indent << pre << "<RawArray of=\"" << typeid(T).name() << "\" length=\"" << length_ << "\" stride=\"" << stride_ << "\" data=\"";
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
      out << "\" at=\"0x";
      out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get());
      if (id_.get() == nullptr) {
        out << "\"/>" << post;
      }
      else {
        out << "\">\n";
        out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
        out << indent << "</RawArray>" << post;
      }
      return out.str();
    }
    virtual int64_t length() const { return length_; }
    virtual const std::shared_ptr<Content> shallow_copy() const { return std::shared_ptr<Content>(new RawArrayOf<T>(id_, ptr_, offset_, length_, stride_)); }
    virtual const std::shared_ptr<Content> get(int64_t at) const { return slice(at, at + 1); }
    virtual const std::shared_ptr<Content> slice(int64_t start, int64_t stop) const {
      std::shared_ptr<Identity> id(nullptr);
      if (id_.get() != nullptr) {
        id = id_.get()->slice(start, stop);
      }
      return std::shared_ptr<Content>(new RawArrayOf<T>(id, ptr_, offset_ + start, stop - start, stride_));
    }
    virtual const std::pair<int64_t, int64_t> minmax_depth() const { return std::pair<int64_t, int64_t>(1, 1); }

    // const std::shared_ptr<Content> getitem_next(SliceItem& head, Slice& tail, std::shared_ptr<Index> advanced) {
    //   return std::shared_ptr<Content>(nullptr);
    // }

    T* borrow(int64_t at) const { return reinterpret_cast<T*>(reinterpret_cast<ssize_t>(ptr_.get()) + (ssize_t)stride_*(ssize_t)(offset_ + at)); }

  private:
    std::shared_ptr<Identity> id_;
    const std::shared_ptr<T> ptr_;
    const int64_t offset_;
    const int64_t length_;
    const int64_t stride_;
  };
}

#endif // AWKWARD_RAWARRAY_H_
