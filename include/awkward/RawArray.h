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
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  template <typename T>
  class RawArrayOf: public Content {
  public:
    RawArrayOf<T>(const std::shared_ptr<Identity> id, const std::shared_ptr<T> ptr, const int64_t offset, const int64_t length, const int64_t itemsize)
        : id_(id)
        , ptr_(ptr)
        , offset_(offset)
        , length_(length)
        , itemsize_(itemsize) {
          assert(sizeof(T) == itemsize);
        }

    RawArrayOf<T>(const std::shared_ptr<Identity> id, const std::shared_ptr<T> ptr, const int64_t length)
        : id_(id)
        , ptr_(ptr)
        , offset_(0)
        , length_(length)
        , itemsize_(sizeof(T)) { }

    RawArrayOf<T>(const std::shared_ptr<Identity> id, const int64_t length)
        : id_(id)
        , ptr_(std::shared_ptr<T>(new T[(size_t)length], awkward::util::array_deleter<T>()))
        , offset_(0)
        , length_(length)
        , itemsize_(sizeof(T)) { }

    const std::shared_ptr<T> ptr() const { return ptr_; }
    const int64_t offset() const { return offset_; }
    const int64_t itemsize() const { return itemsize_; }

    bool isempty() const { return length_ == 0; }
    ssize_t byteoffset() const { return (ssize_t)itemsize_*(ssize_t)offset_; }
    uint8_t* byteptr() const { return reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset()); }
    ssize_t bytelength() const { return (ssize_t)itemsize_*(ssize_t)length_; }
    uint8_t getbyte(ssize_t at) const { return *reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get()) + (ssize_t)(byteoffset() + at)); }

    T* borrow(int64_t at) const { return reinterpret_cast<T*>(reinterpret_cast<ssize_t>(ptr_.get()) + (ssize_t)itemsize_*(ssize_t)(offset_ + at)); }

    virtual const std::string classname() const { return std::string("RawArrayOf<") + std::string(typeid(T).name()) + std::string(">"); }

    virtual const std::shared_ptr<Identity> id() const { return id_; }
    virtual void setid() {
      if (length() <= kMaxInt32) {
        Identity32* rawid = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
        std::shared_ptr<Identity> newid(rawid);
        awkward_new_identity32(rawid->ptr().get(), length());
        setid(newid);
      }
      else {
        Identity64* rawid = new Identity64(Identity::newref(), Identity::FieldLoc(), 1, length());
        std::shared_ptr<Identity> newid(rawid);
        awkward_new_identity64(rawid->ptr().get(), length());
        setid(newid);
      }
    }
    virtual void setid(const std::shared_ptr<Identity> id) {
      if (id.get() != nullptr  &&  length() != id.get()->length()) {
        throw std::invalid_argument("content and its id must have the same length");
      }
      id_ = id;
    }

    const std::string tostring() { return tostring_part("", "", ""); }
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const {
      std::stringstream out;
      out << indent << pre << "<RawArray of=\"" << typeid(T).name() << "\" length=\"" << length_ << "\" itemsize=\"" << itemsize_ << "\" data=\"";
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

    virtual const std::shared_ptr<Content> shallow_copy() const { return std::shared_ptr<Content>(new RawArrayOf<T>(id_, ptr_, offset_, length_, itemsize_)); }

    virtual void checksafe() const {
      if (id_.get() != nullptr  &&  id_.get()->length() < length_) {
        util::handle_error(failure("len(id) < len(array)", kSliceNone, kSliceNone), id_.get()->classname(), nullptr);
      }
    }

    virtual const std::shared_ptr<Content> getitem_at(int64_t at) const {
      int64_t regular_at = at;
      if (regular_at < 0) {
        regular_at += length_;
      }
      if (!(0 <= regular_at  &&  regular_at < length_)) {
        util::handle_error(failure("index out of range", kSliceNone, at), classname(), id_.get());
      }
      return getitem_at_unsafe(regular_at);
    }

    virtual const std::shared_ptr<Content> getitem_at_unsafe(int64_t at) const {
      return getitem_range_unsafe(at, at + 1);
    }

    virtual const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const {
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), length_);
      if (id_.get() != nullptr  &&  regular_stop > id_.get()->length()) {
        util::handle_error(failure("index out of range", kSliceNone, stop), id_.get()->classname(), nullptr);
      }
      return getitem_range_unsafe(regular_start, regular_stop);
    }

    virtual const std::shared_ptr<Content> getitem_range_unsafe(int64_t start, int64_t stop) const {
      std::shared_ptr<Identity> id(nullptr);
      if (id_.get() != nullptr) {
        id = id_.get()->getitem_range_unsafe(start, stop);
      }
      return std::shared_ptr<Content>(new RawArrayOf<T>(id, ptr_, offset_ + start, stop - start, itemsize_));
    }

    virtual const std::shared_ptr<Content> getitem(const Slice& where) const {
      std::shared_ptr<SliceItem> nexthead = where.head();
      Slice nexttail = where.tail();
      Index64 nextadvanced(0);
      return getitem_next(nexthead, nexttail, nextadvanced);
    }

    const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
      if (tail.length() != 0) {
        throw std::invalid_argument("too many indexes for array");
      }

      if (head.get() == nullptr) {
        return shallow_copy();
      }

      else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
        return getitem_at(at->at());
      }

      else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
        if (range->step() == Slice::none()  ||  range->step() == 1) {
          return getitem_range(range->start(), range->stop());
        }
        else {
          int64_t start = range->start();
          int64_t stop = range->stop();
          int64_t step = range->step();
          if (step == Slice::none()) {
            step = 1;
          }
          else if (step == 0) {
            throw std::invalid_argument("slice step must not be 0");
          }
          awkward_regularize_rangeslice(&start, &stop, step > 0, range->hasstart(), range->hasstop(), length_);

          int64_t numer = abs(start - stop);
          int64_t denom = abs(step);
          int64_t d = numer / denom;
          int64_t m = numer % denom;
          int64_t lenhead = d + (m != 0 ? 1 : 0);

          Index64 nextcarry(lenhead);
          int64_t* nextcarryptr = nextcarry.ptr().get();
          for (int64_t i = 0;  i < lenhead;  i++) {
            nextcarryptr[i] = start + step*i;
          }

          return carry(nextcarry);
        }
      }

      else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
        throw std::runtime_error("ellipsis");
      }

      else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
        throw std::runtime_error("newaxis");
      }

      else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
        if (array->shape().size() != 1) {
          throw std::runtime_error("array.ndim != 1");
        }
        if (advanced.length() == 0) {
          Index64 flathead = array->ravel();
          Error err = awkward_regularize_arrayslice_64(
            flathead.ptr().get(),
            flathead.length(),
            length_);
          util::handle_error(err, classname(), id_.get());
          return carry(flathead);
        }
        else {
          throw std::runtime_error("advanced array");
        }
      }

      else {
        throw std::runtime_error("unrecognized slice item type");
      }
    }

    virtual const std::shared_ptr<Content> carry(const Index64& carry) const {
      std::shared_ptr<T> ptr(new T[(size_t)carry.length()], awkward::util::array_deleter<T>());
      Error err = awkward_numpyarray_getitem_next_null_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        carry.length(),
        itemsize_,
        byteoffset(),
        carry.ptr().get());
      util::handle_error(err, classname(), id_.get());

      std::shared_ptr<Identity> id(nullptr);
      if (id_.get() != nullptr) {
        id = id_.get()->getitem_carry_64(carry);
      }

      return std::shared_ptr<Content>(new RawArrayOf<T>(id, ptr, 0, carry.length(), itemsize_));
    }

    virtual const std::pair<int64_t, int64_t> minmax_depth() const { return std::pair<int64_t, int64_t>(1, 1); }

  private:
    std::shared_ptr<Identity> id_;
    const std::shared_ptr<T> ptr_;
    const int64_t offset_;
    const int64_t length_;
    const int64_t itemsize_;
  };
}

#endif // AWKWARD_RAWARRAY_H_
