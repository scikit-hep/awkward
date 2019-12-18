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
#include "awkward/type/PrimitiveType.h"
#include "awkward/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  void tojson_boolean(ToJson& builder, bool* array, int64_t length) {
    for (int i = 0;  i < length;  i++) {
      builder.boolean((bool)array[i]);
    }
  }

  template <typename T>
  void tojson_integer(ToJson& builder, T* array, int64_t length) {
    for (int i = 0;  i < length;  i++) {
      builder.integer((int64_t)array[i]);
    }
  }

  template <typename T>
  void tojson_real(ToJson& builder, T* array, int64_t length) {
    for (int i = 0;  i < length;  i++) {
      builder.real((double)array[i]);
    }
  }

  template <typename T>
  class RawArrayOf: public Content {
  public:
    RawArrayOf<T>(const std::shared_ptr<Identity> id, const std::shared_ptr<Type> type, const std::shared_ptr<T> ptr, const int64_t offset, const int64_t length, const int64_t itemsize)
        : Content(id, type)
        , ptr_(ptr)
        , offset_(offset)
        , length_(length)
        , itemsize_(itemsize) {
          assert(sizeof(T) == itemsize);
        }

    RawArrayOf<T>(const std::shared_ptr<Identity> id, const std::shared_ptr<Type> type, const std::shared_ptr<T> ptr, const int64_t length)
        : Content(id, type)
        , ptr_(ptr)
        , offset_(0)
        , length_(length)
        , itemsize_(sizeof(T)) { }

    RawArrayOf<T>(const std::shared_ptr<Identity> id, const std::shared_ptr<Type> type, const int64_t length)
        : Content(id, type)
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

    const std::string classname() const override { return std::string("RawArrayOf<") + std::string(typeid(T).name()) + std::string(">"); }

    void setid() override {
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
    void setid(const std::shared_ptr<Identity> id) override {
      if (id != nullptr  &&  length() != id.get()->length()) {
        throw std::invalid_argument("content and its id must have the same length");
      }
      id_ = id;
    }

    const std::string tostring() { return tostring_part("", "", ""); }
    const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const override {
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

    void tojson_part(ToJson& builder) const override {
      if (std::is_same<T, double>::value) {
        tojson_real(builder, reinterpret_cast<double*>(byteptr()), length());
      }
      else if (std::is_same<T, float>::value) {
        tojson_real(builder, reinterpret_cast<float*>(byteptr()), length());
      }
      else if (std::is_same<T, int64_t>::value) {
        tojson_integer(builder, reinterpret_cast<int64_t*>(byteptr()), length());
      }
      else if (std::is_same<T, uint64_t>::value) {
        tojson_integer(builder, reinterpret_cast<uint64_t*>(byteptr()), length());
      }
      else if (std::is_same<T, int32_t>::value) {
        tojson_integer(builder, reinterpret_cast<int32_t*>(byteptr()), length());
      }
      else if (std::is_same<T, uint32_t>::value) {
        tojson_integer(builder, reinterpret_cast<uint32_t*>(byteptr()), length());
      }
      else if (std::is_same<T, int16_t>::value) {
        tojson_integer(builder, reinterpret_cast<int16_t*>(byteptr()), length());
      }
      else if (std::is_same<T, uint16_t>::value) {
        tojson_integer(builder, reinterpret_cast<uint16_t*>(byteptr()), length());
      }
      else if (std::is_same<T, int8_t>::value) {
        tojson_integer(builder, reinterpret_cast<int8_t*>(byteptr()), length());
      }
      else if (std::is_same<T, uint8_t>::value) {
        tojson_integer(builder, reinterpret_cast<uint8_t*>(byteptr()), length());
      }
      else if (std::is_same<T, bool>::value) {
        tojson_boolean(builder, reinterpret_cast<bool*>(byteptr()), length());
      }
      else {
        throw std::invalid_argument(std::string("cannot convert RawArrayOf<") + typeid(T).name() + std::string("> into JSON"));
      }
    }

    const std::shared_ptr<Type> innertype(bool bare) const override {
      if (std::is_same<T, double>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::float64));
      }
      else if (std::is_same<T, float>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::float32));
      }
      else if (std::is_same<T, int64_t>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::int64));
      }
      else if (std::is_same<T, uint64_t>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::uint64));
      }
      else if (std::is_same<T, int32_t>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::int32));
      }
      else if (std::is_same<T, uint32_t>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::uint32));
      }
      else if (std::is_same<T, int16_t>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::int16));
      }
      else if (std::is_same<T, uint16_t>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::uint16));
      }
      else if (std::is_same<T, int8_t>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::int8));
      }
      else if (std::is_same<T, uint8_t>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::uint8));
      }
      else if (std::is_same<T, bool>::value) {
        return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::boolean));
      }
      else {
        throw std::invalid_argument(std::string("RawArrayOf<") + typeid(T).name() + std::string("> cannot be expressed as a PrimitiveType"));
      }
    }

    void settype_part(const std::shared_ptr<Type> type) override {
      if (accepts(type)) {
        type_ = type;
      }
      else {
        throw std::invalid_argument(std::string("provided type is incompatible with array: ") + type.get()->compare(baretype()));
      }
    }

    bool accepts(const std::shared_ptr<Type> type) override {
      std::shared_ptr<Type> check = type.get()->level();
      if (std::is_same<T, double>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::float64)), false);
      }
      else if (std::is_same<T, float>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::float32)), false);
      }
      else if (std::is_same<T, int64_t>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::int64)), false);
      }
      else if (std::is_same<T, uint64_t>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::uint64)), false);
      }
      else if (std::is_same<T, int32_t>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::int32)), false);
      }
      else if (std::is_same<T, uint32_t>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::uint32)), false);
      }
      else if (std::is_same<T, int16_t>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::int16)), false);
      }
      else if (std::is_same<T, uint16_t>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::uint16)), false);
      }
      else if (std::is_same<T, int8_t>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::int8)), false);
      }
      else if (std::is_same<T, uint8_t>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::uint8)), false);
      }
      else if (std::is_same<T, bool>::value) {
        return check.get()->equal(std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::boolean)), false);
      }
      else {
        return false;
      }
    }

    int64_t length() const override { return length_; }

    const std::shared_ptr<Content> shallow_copy() const override { return std::shared_ptr<Content>(new RawArrayOf<T>(id_, type_, ptr_, offset_, length_, itemsize_)); }

    void check_for_iteration() const override {
      if (id_.get() != nullptr  &&  id_.get()->length() < length_) {
        util::handle_error(failure("len(id) < len(array)", kSliceNone, kSliceNone), id_.get()->classname(), nullptr);
      }
    }

    const std::shared_ptr<Content> getitem_nothing() const override {
      return getitem_range_nowrap(0, 0);
    }

    const std::shared_ptr<Content> getitem_at(int64_t at) const override {
      int64_t regular_at = at;
      if (regular_at < 0) {
        regular_at += length_;
      }
      if (!(0 <= regular_at  &&  regular_at < length_)) {
        util::handle_error(failure("index out of range", kSliceNone, at), classname(), id_.get());
      }
      return getitem_at_nowrap(regular_at);
    }

    const std::shared_ptr<Content> getitem_at_nowrap(int64_t at) const override {
      return getitem_range_nowrap(at, at + 1);
    }

    const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const override {
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), length_);
      if (id_.get() != nullptr  &&  regular_stop > id_.get()->length()) {
        util::handle_error(failure("index out of range", kSliceNone, stop), id_.get()->classname(), nullptr);
      }
      return getitem_range_nowrap(regular_start, regular_stop);
    }

    const std::shared_ptr<Content> getitem_range_nowrap(int64_t start, int64_t stop) const override {
      std::shared_ptr<Identity> id(nullptr);
      if (id_.get() != nullptr) {
        id = id_.get()->getitem_range_nowrap(start, stop);
      }
      return std::shared_ptr<Content>(new RawArrayOf<T>(id, type_, ptr_, offset_ + start, stop - start, itemsize_));
    }

    const std::shared_ptr<Content> getitem_field(const std::string& key) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname() + std::string(" by field name"));
    }

    const std::shared_ptr<Content> getitem_fields(const std::vector<std::string>& keys) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname() + std::string(" by field name"));
    }

    const std::shared_ptr<Content> getitem(const Slice& where) const override {
      std::shared_ptr<SliceItem> nexthead = where.head();
      Slice nexttail = where.tail();
      Index64 nextadvanced(0);
      return getitem_next(nexthead, nexttail, nextadvanced);
    }

    const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const override {
      if (tail.length() != 0) {
        throw std::invalid_argument("too many indexes for array");
      }
      return Content::getitem_next(head, tail, advanced);
    }

    const std::shared_ptr<Content> carry(const Index64& carry) const override {
      std::shared_ptr<T> ptr(new T[(size_t)carry.length()], awkward::util::array_deleter<T>());
      struct Error err = awkward_numpyarray_getitem_next_null_64(
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

      return std::shared_ptr<Content>(new RawArrayOf<T>(id, type_, ptr, 0, carry.length(), itemsize_));
    }

    const std::pair<int64_t, int64_t> minmax_depth() const override {
      return std::pair<int64_t, int64_t>(1, 1);
    }

    int64_t numfields() const override { return -1; }

    int64_t fieldindex(const std::string& key) const override {
      throw std::invalid_argument("array contains no Records");
    }

    const std::string key(int64_t fieldindex) const override {
      throw std::invalid_argument("array contains no Records");
    }

    bool haskey(const std::string& key) const override {
      throw std::invalid_argument("array contains no Records");
    }

    const std::vector<std::string> keyaliases(int64_t fieldindex) const override {
      throw std::invalid_argument("array contains no Records");
    }

    const std::vector<std::string> keyaliases(const std::string& key) const override {
      throw std::invalid_argument("array contains no Records");
    }

    const std::vector<std::string> keys() const override {
      throw std::invalid_argument("array contains no Records");
    }

  protected:
    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override {
      return getitem_at(at.at());
    }

    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override {
      if (range.step() == Slice::none()  ||  range.step() == 1) {
        return getitem_range(range.start(), range.stop());
      }
      else {
        int64_t start = range.start();
        int64_t stop = range.stop();
        int64_t step = range.step();
        if (step == Slice::none()) {
          step = 1;
        }
        else if (step == 0) {
          throw std::invalid_argument("slice step must not be 0");
        }
        awkward_regularize_rangeslice(&start, &stop, step > 0, range.hasstart(), range.hasstop(), length_);

        int64_t numer = std::abs(start - stop);
        int64_t denom = std::abs(step);
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

    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override {
      assert(advanced.length() == 0);
      if (array.shape().size() != 1) {
        throw std::runtime_error("array.ndim != 1");
      }
      Index64 flathead = array.ravel();
      struct Error err = awkward_regularize_arrayslice_64(
        flathead.ptr().get(),
        flathead.length(),
        length_);
      util::handle_error(err, classname(), id_.get());
      return carry(flathead);
    }

    const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const override {
      throw std::invalid_argument(field.tostring() + std::string(" is not a valid slice type for ") + classname());
    }

    const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const override {
      throw std::invalid_argument(fields.tostring() + std::string(" is not a valid slice type for ") + classname());
    }

  private:
    const std::shared_ptr<T> ptr_;
    const int64_t offset_;
    const int64_t length_;
    const int64_t itemsize_;
  };
}

#endif // AWKWARD_RAWARRAY_H_
