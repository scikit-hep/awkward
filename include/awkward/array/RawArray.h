// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RAWARRAY_H_
#define AWKWARD_RAWARRAY_H_

#include <cstring>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <typeinfo>

#include "awkward/cpu-kernels/util.h"
#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"

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
  class EXPORT_SYMBOL RawArrayOf: public Content {
  public:
    RawArrayOf<T>(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::shared_ptr<T>& ptr, const int64_t offset, const int64_t length, const int64_t itemsize)
        : Content(identities, parameters)
        , ptr_(ptr)
        , offset_(offset)
        , length_(length)
        , itemsize_(itemsize) {
      if (sizeof(T) != itemsize) {
        throw std::invalid_argument("sizeof(T) != itemsize");
      }
    }

    RawArrayOf<T>(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::shared_ptr<T>& ptr, const int64_t length)
        : Content(identities, parameters)
        , ptr_(ptr)
        , offset_(0)
        , length_(length)
        , itemsize_(sizeof(T)) { }

    RawArrayOf<T>(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const int64_t length)
        : Content(identities, parameters)
        , ptr_(std::shared_ptr<T>(new T[(size_t)length], util::array_deleter<T>()))
        , offset_(0)
        , length_(length)
        , itemsize_(sizeof(T)) { }

    const std::shared_ptr<T> ptr() const {
      return ptr_;
    }

    const int64_t offset() const {
      return offset_;
    }

    const int64_t itemsize() const {
      return itemsize_;
    }

    bool isempty() const {
      return length_ == 0;
    }

    ssize_t byteoffset() const {
      return (ssize_t)itemsize_*(ssize_t)offset_;
    }

    uint8_t* byteptr() const {
      return reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset());
    }
    ssize_t bytelength() const {
      return (ssize_t)itemsize_*(ssize_t)length_;
    }
    uint8_t getbyte(ssize_t at) const {
      return *reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get()) + (ssize_t)(byteoffset() + at));
    }

    T* borrow(int64_t at) const {
      return reinterpret_cast<T*>(reinterpret_cast<ssize_t>(ptr_.get()) + (ssize_t)itemsize_*(ssize_t)(offset_ + at));
    }

    const std::string classname() const override {
      return std::string("RawArrayOf<") + std::string(typeid(T).name()) + std::string(">");
    }

    void setidentities() override {
      if (length() <= kMaxInt32) {
        std::shared_ptr<Identities> newidentities = std::make_shared<Identities32>(Identities::newref(), Identities::FieldLoc(), 1, length());
        Identities32* rawidentities = reinterpret_cast<Identities32*>(newidentities.get());
        awkward_new_identities32(rawidentities->ptr().get(), length());
        setidentities(newidentities);
      }
      else {
        std::shared_ptr<Identities> newidentities = std::make_shared<Identities64>(Identities::newref(), Identities::FieldLoc(), 1, length());
        Identities64* rawidentities = reinterpret_cast<Identities64*>(newidentities.get());
        awkward_new_identities64(rawidentities->ptr().get(), length());
        setidentities(newidentities);
      }
    }

    void setidentities(const std::shared_ptr<Identities>& identities) override {
      if (identities.get() != nullptr  &&  length() != identities.get()->length()) {
        throw std::invalid_argument("content and its identities must have the same length");
      }
      identities_ = identities;
    }

    const std::shared_ptr<Type> type(const std::map<std::string, std::string>& typestrs) const override {
      if (std::is_same<T, double>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::float64);
      }
      else if (std::is_same<T, float>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::float32);
      }
      else if (std::is_same<T, int64_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::int64);
      }
      else if (std::is_same<T, uint64_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::uint64);
      }
      else if (std::is_same<T, int32_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::int32);
      }
      else if (std::is_same<T, uint32_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::uint32);
      }
      else if (std::is_same<T, int16_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::int16);
      }
      else if (std::is_same<T, uint16_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::uint16);
      }
      else if (std::is_same<T, int8_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::int8);
      }
      else if (std::is_same<T, uint8_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::uint8);
      }
      else if (std::is_same<T, bool>::value) {
        return std::make_shared<PrimitiveType>(parameters_, util::gettypestr(parameters_, typestrs), PrimitiveType::boolean);
      }
      else {
        throw std::invalid_argument(std::string("RawArrayOf<") + typeid(T).name() + std::string("> does not have a known type"));
      }
    }

    const std::string tostring() {
      return tostring_part("", "", "");
    }

    const std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override {
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
      if (identities_.get() == nullptr  &&  parameters_.empty()) {
        out << "\"/>" << post;
      }
      else {
        out << "\">\n";
        out << identities_.get()->tostring_part(indent + std::string("    "), "", "\n");
        if (!parameters_.empty()) {
          out << parameters_tostring(indent + std::string("    "), "", "\n");
        }
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

    int64_t length() const override {
      return length_;
    }

    void nbytes_part(std::map<size_t, int64_t>& largest) const override {
      size_t x = (size_t)ptr_.get();
      auto it = largest.find(x);
      if (it == largest.end()  ||  it->second < (int64_t)(sizeof(T)*length_)) {
        largest[x] = (int64_t)(sizeof(T)*length_);
      }
      if (identities_.get() != nullptr) {
        identities_.get()->nbytes_part(largest);
      }
    }

    const std::shared_ptr<Content> shallow_copy() const override {
      return std::make_shared<RawArrayOf<T>>(identities_, parameters_, ptr_, offset_, length_, itemsize_);
    }

    const std::shared_ptr<Content> deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const override {
      std::shared_ptr<T> ptr = ptr_;
      int64_t offset = offset_;
      if (copyarrays) {
        ptr = std::shared_ptr<T>(new T[(size_t)length_], util::array_deleter<T>());
        memcpy(ptr.get(), &ptr_.get()[(size_t)offset_], sizeof(T)*((size_t)length_));
        offset = 0;
      }
      std::shared_ptr<Identities> identities = identities_;
      if (copyidentities  &&  identities_.get() != nullptr) {
        identities = identities_.get()->deep_copy();
      }
      return std::make_shared<RawArrayOf<T>>(identities, parameters_, ptr, offset, length_, itemsize_);
    }

    void check_for_iteration() const override {
      if (identities_.get() != nullptr  &&  identities_.get()->length() < length_) {
        util::handle_error(failure("len(identities) < len(array)", kSliceNone, kSliceNone), identities_.get()->classname(), nullptr);
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
        util::handle_error(failure("index out of range", kSliceNone, at), classname(), identities_.get());
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
      if (identities_.get() != nullptr  &&  regular_stop > identities_.get()->length()) {
        util::handle_error(failure("index out of range", kSliceNone, stop), identities_.get()->classname(), nullptr);
      }
      return getitem_range_nowrap(regular_start, regular_stop);
    }

    const std::shared_ptr<Content> getitem_range_nowrap(int64_t start, int64_t stop) const override {
      std::shared_ptr<Identities> identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_range_nowrap(start, stop);
      }
      return std::make_shared<RawArrayOf<T>>(identities, parameters_, ptr_, offset_ + start, stop - start, itemsize_);
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

    const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const override {
      if (tail.length() != 0) {
        throw std::invalid_argument("too many indexes for array");
      }
      return Content::getitem_next(head, tail, advanced);
    }

    const std::shared_ptr<Content> carry(const Index64& carry) const override {
      std::shared_ptr<T> ptr(new T[(size_t)carry.length()], util::array_deleter<T>());
      struct Error err = awkward_numpyarray_getitem_next_null_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        carry.length(),
        itemsize_,
        byteoffset(),
        carry.ptr().get());
      util::handle_error(err, classname(), identities_.get());

      std::shared_ptr<Identities> identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_carry_64(carry);
      }

      return std::make_shared<RawArrayOf<T>>(identities, parameters_, ptr, 0, carry.length(), itemsize_);
    }

    const std::string purelist_parameter(const std::string& key) const override {
      return parameter(key);
    }

    bool purelist_isregular() const override {
      return true;
    }

    int64_t purelist_depth() const override {
      return 1;
    }

    const std::pair<int64_t, int64_t> minmax_depth() const override {
      return std::pair<int64_t, int64_t>(1, 1);
    }

    const std::pair<bool, int64_t> branch_depth() const {
      return std::pair<bool, int64_t>(false, 1);
    }

    int64_t numfields() const override {
      return -1;
    }

    int64_t fieldindex(const std::string& key) const override {
      throw std::invalid_argument(std::string("key ") + util::quote(key, true) + std::string(" does not exist (data are not records)"));
    }

    const std::string key(int64_t fieldindex) const override {
      throw std::invalid_argument(std::string("fieldindex \"") + std::to_string(fieldindex) + std::string("\" does not exist (data are not records)"));
    }

    bool haskey(const std::string& key) const override {
      return false;
    }

    const std::vector<std::string> keys() const override {
      return std::vector<std::string>();
    }

    // operations

    const std::string validityerror(const std::string& path) const override {
      return std::string();
    }

    const std::shared_ptr<Content> sizes(int64_t axis, int64_t depth) const override {
      int64_t toaxis = axis_wrap_if_negative(axis);
      if (toaxis == depth) {
        Index64 out(1);
        out.ptr().get()[0] = length();
        return std::make_shared<RawArrayOf<int64_t>>(Identities::none(), util::Parameters(), out.ptr(), 0, 1, sizeof(int64_t));
      }
      else {
        throw std::invalid_argument("'axis' out of range for 'sizes'");
      }
    }

    const std::shared_ptr<Content> flatten(int64_t axis) const override {
      throw std::invalid_argument("RawArray cannot be flattened because it is one-dimentional");
    }

    bool mergeable(const std::shared_ptr<Content>& other, bool mergebool) const override {
      if (dynamic_cast<EmptyArray*>(other.get())) {
        return true;
      }
      else if (IndexedArray32* rawother = dynamic_cast<IndexedArray32*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (IndexedArrayU32* rawother = dynamic_cast<IndexedArrayU32*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (IndexedArray64* rawother = dynamic_cast<IndexedArray64*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (IndexedOptionArray32* rawother = dynamic_cast<IndexedOptionArray32*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (IndexedOptionArray64* rawother = dynamic_cast<IndexedOptionArray64*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }

      if (RawArrayOf<T>* rawother = dynamic_cast<RawArrayOf<T>*>(other.get())) {
        return true;
      }
      else {
        return false;
      }
    }

    const std::shared_ptr<Content> merge(const std::shared_ptr<Content>& other) const override {
      if (dynamic_cast<EmptyArray*>(other.get())) {
        return shallow_copy();
      }
      else if (IndexedArray32* rawother = dynamic_cast<IndexedArray32*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (IndexedArrayU32* rawother = dynamic_cast<IndexedArrayU32*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (IndexedArray64* rawother = dynamic_cast<IndexedArray64*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (IndexedOptionArray32* rawother = dynamic_cast<IndexedOptionArray32*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (IndexedOptionArray64* rawother = dynamic_cast<IndexedOptionArray64*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }

      if (RawArrayOf<T>* rawother = dynamic_cast<RawArrayOf<T>*>(other.get())) {
        std::shared_ptr<T> ptr = std::shared_ptr<T>(new T[(size_t)(length_ + rawother->length())], util::array_deleter<T>());
        memcpy(ptr.get(), &ptr_.get()[(size_t)offset_], sizeof(T)*((size_t)length_));
        memcpy(&ptr.get()[(size_t)length_], &rawother->ptr().get()[(size_t)rawother->offset()], sizeof(T)*((size_t)rawother->length()));
        return std::make_shared<RawArrayOf<T>>(Identities::none(), util::Parameters(), ptr, 0, length_ + rawother->length(), itemsize_);
      }
      else {
        throw std::invalid_argument(std::string("cannot merge ") + classname() + std::string(" with ") + other.get()->classname());
      }
    }

    const std::shared_ptr<SliceItem> asslice() const override {
      throw std::invalid_argument("cannot use RawArray as a slice");
    }

    const std::shared_ptr<Content> reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const override {
      throw std::runtime_error("FIXME: Raw:reduce_next");
    }

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
      if (advanced.length() != 0) {
        throw std::runtime_error("RawArray::getitem_next(SliceAt): advanced.length() != 0");
      }
      if (array.shape().size() != 1) {
        throw std::runtime_error("array.ndim != 1");
      }
      Index64 flathead = array.ravel();
      struct Error err = awkward_regularize_arrayslice_64(
        flathead.ptr().get(),
        flathead.length(),
        length_);
      util::handle_error(err, classname(), identities_.get());
      return carry(flathead);
    }

    const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname() + std::string(" by a field name because it has no fields"));
    }

    const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname() + std::string(" by field names because it has no fields"));
    }

    const std::shared_ptr<Content> getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname() + std::string(" by a jagged array because it is one-dimensional"));
    }

    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const override {
      throw std::runtime_error("undefined operation: RawArray::getitem_next_jagged(array)");
    }

    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const override {
      throw std::runtime_error("undefined operation: RawArray::getitem_next_jagged(missing)");
    }

    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const override {
      throw std::runtime_error("undefined operation: RawArray::getitem_next_jagged(jagged)");
    }

  private:
    const std::shared_ptr<T> ptr_;
    const int64_t offset_;
    const int64_t length_;
    const int64_t itemsize_;
  };
}

#endif // AWKWARD_RAWARRAY_H_
