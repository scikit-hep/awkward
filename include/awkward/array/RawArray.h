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
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"

namespace awkward {
  /// @brief Internal function to fill JSON with boolean values.
  void
    tojson_boolean(ToJson& builder, bool* array, int64_t length) {
    for (int i = 0;  i < length;  i++) {
      builder.boolean((bool)array[i]);
    }
  }

  /// @brief Internal function to fill JSON with integer values.
  template <typename T>
  void
    tojson_integer(ToJson& builder, T* array, int64_t length) {
    for (int i = 0;  i < length;  i++) {
      builder.integer((int64_t)array[i]);
    }
  }

  /// @brief Internal function to fill JSON with floating-point values.
  template <typename T>
  void
    tojson_real(ToJson& builder, T* array, int64_t length) {
    for (int i = 0;  i < length;  i++) {
      builder.real((double)array[i]);
    }
  }

  /// @class RawArrayOf
  ///
  /// @brief Represents a one-dimensional array of type `T`, usable purely in
  /// C++ and implemented entirely in include/awkward/array/RawArray.h.
  ///
  /// See #RawArrayOf for the meaning of each parameter.
  ///
  /// Arrays of any type `T` can be passed through Awkward Array operations,
  /// even slicing, but operations that have to interpret the array's values
  /// (such as reducers like "sum" and "max") only work on numeric types:
  ///
  ///  - 32-bit and 64-bit floating point numbers
  ///  - 8-bit, 16-bit, 32-bit, and 64-bit signed and unsigned integers
  ///  - 8-bit booleans
  template <typename T>
  class EXPORT_SYMBOL RawArrayOf: public Content {
  public:
    /// @brief Creates a RawArray from a full set of parameters.
    ///
    /// @param identities Optional Identities for each element of the array
    /// (may be `nullptr`).
    /// @param parameters String-to-JSON map that augments the meaning of this
    /// array.
    /// @param ptr Reference-counted pointer to the array buffer.
    /// @param offset Location of item zero in the buffer, relative to
    /// #ptr, measured in the number of elements. We keep this information in
    /// two parameters (#ptr and #offset) rather than moving #ptr so that
    /// #ptr can be reference counted among all arrays that use the same
    /// buffer.
    /// @param length Number of elements in the array.
    /// @param itemsize Number of bytes per item; should agree with the format.
    RawArrayOf<T>(const IdentitiesPtr& identities,
                  const util::Parameters& parameters,
                  const std::shared_ptr<T>& ptr,
                  const int64_t offset,
                  const int64_t length,
                  const int64_t itemsize)
        : Content(identities, parameters)
        , ptr_(ptr)
        , offset_(offset)
        , length_(length)
        , itemsize_(itemsize) {
      if (sizeof(T) != itemsize) {
        throw std::invalid_argument("sizeof(T) != itemsize");
      }
    }

    /// @brief Creates a RawArray without having to specify #itemsize.
    ///
    /// The #itemsize is computed as `sizeof(T)`.
    RawArrayOf<T>(const IdentitiesPtr& identities,
                  const util::Parameters& parameters,
                  const std::shared_ptr<T>& ptr,
                  const int64_t length)
        : Content(identities, parameters)
        , ptr_(ptr)
        , offset_(0)
        , length_(length)
        , itemsize_(sizeof(T)) { }

    /// @brief Creates a RawArray without providing a #ptr to data and without
    /// having to specify #itemsize.
    ///
    /// This constructor allocates a new buffer with `itemsize * length` bytes.
    ///
    /// The #itemsize is computed as `sizeof(T)`.
    RawArrayOf<T>(const IdentitiesPtr& identities,
                  const util::Parameters& parameters,
                  const int64_t length)
        : Content(identities, parameters)
        , ptr_(std::shared_ptr<T>(new T[(size_t)length],
                                  util::array_deleter<T>()))
        , offset_(0)
        , length_(length)
        , itemsize_(sizeof(T)) { }

    /// @brief Reference-counted pointer to the array buffer.
    const std::shared_ptr<T>
      ptr() const {
      return ptr_;
    }

    /// @brief Location of item zero in the buffer, relative to
    /// #ptr, measured in the number of elements.
    ///
    /// We keep this information in two parameters
    /// (#ptr and #offset) rather than moving #ptr so that #ptr can be
    /// reference counted among all arrays that use the same buffer.
    const int64_t
      offset() const {
      return offset_;
    }

    /// @brief Number of bytes per item; should be `sizeof(T)`.
    const int64_t
      itemsize() const {
      return itemsize_;
    }

    /// @brief Location of item zero in the buffer, relative to
    /// `ptr`, measured in bytes, rather than number of elements; see #offset.
    ssize_t
      byteoffset() const {
      return (ssize_t)itemsize_*(ssize_t)offset_;
    }

    /// @brief An untyped pointer to item zero in the buffer.
    void*
      byteptr() const {
      return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get())
                                        + byteoffset());
    }

    /// @brief The length of the array in bytes.
    ssize_t
      bytelength() const {
      return (ssize_t)itemsize_*(ssize_t)length_;
    }

    /// @brief Dereferences a selected item as a `uint8_t`.
    uint8_t
      getbyte(ssize_t at) const {
      return *reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get())
                                         + (ssize_t)(byteoffset() + at));
    }

    /// @brief Dereferences a selected item as pointer to `T`.
    ///
    /// The name is a reminder that the reference is borrowed, not owned,
    /// and should not be deleted by the caller.
    T*
      borrow(int64_t at) const {
      return reinterpret_cast<T*>(
        reinterpret_cast<ssize_t>(ptr_.get()) +
        (ssize_t)itemsize_*(ssize_t)(offset_ + at));
    }

    /// @brief User-friendly name of this class: `"RawArray<T>"` where `T` is
    /// is `typeid(T).name()` (platform-dependent).
    const std::string
      classname() const override {
      return std::string("RawArrayOf<") + std::string(typeid(T).name()) +
        std::string(">");
    }

    void
      setidentities() override {
      if (length() <= kMaxInt32) {
        IdentitiesPtr newidentities =
          std::make_shared<Identities32>(Identities::newref(),
                                         Identities::FieldLoc(), 1, length());
        Identities32* rawidentities =
          reinterpret_cast<Identities32*>(newidentities.get());
        awkward_new_identities32(rawidentities->ptr().get(), length());
        setidentities(newidentities);
      }
      else {
        IdentitiesPtr newidentities =
          std::make_shared<Identities64>(Identities::newref(),
                                         Identities::FieldLoc(), 1, length());
        Identities64* rawidentities =
          reinterpret_cast<Identities64*>(newidentities.get());
        awkward_new_identities64(rawidentities->ptr().get(), length());
        setidentities(newidentities);
      }
    }

    void
      setidentities(const IdentitiesPtr& identities) override {
      if (identities.get() != nullptr  &&
          length() != identities.get()->length()) {
        throw std::invalid_argument(
          "content and its identities must have the same length");
      }
      identities_ = identities;
    }

    const TypePtr
      type(const util::TypeStrs& typestrs) const override {
      if (std::is_same<T, double>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::float64);
      }
      else if (std::is_same<T, float>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::float32);
      }
      else if (std::is_same<T, int64_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::int64);
      }
      else if (std::is_same<T, uint64_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::uint64);
      }
      else if (std::is_same<T, int32_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::int32);
      }
      else if (std::is_same<T, uint32_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::uint32);
      }
      else if (std::is_same<T, int16_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::int16);
      }
      else if (std::is_same<T, uint16_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::uint16);
      }
      else if (std::is_same<T, int8_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::int8);
      }
      else if (std::is_same<T, uint8_t>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::uint8);
      }
      else if (std::is_same<T, bool>::value) {
        return std::make_shared<PrimitiveType>(parameters_,
          util::gettypestr(parameters_, typestrs), PrimitiveType::boolean);
      }
      else {
        throw std::invalid_argument(std::string("RawArrayOf<") +
          typeid(T).name() + std::string("> does not have a known type"));
      }
    }

    const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const override {
      std::stringstream out;
      out << indent << pre << "<RawArray of=\"" << typeid(T).name()
          << "\" length=\"" << length_ << "\" itemsize=\"" << itemsize_
          << "\" data=\"";
      ssize_t len = bytelength();
      if (len <= 32) {
        for (ssize_t i = 0;  i < len;  i++) {
          if (i != 0  &&  i % 4 == 0) {
            out << " ";
          }
          out << std::hex << std::setw(2) << std::setfill('0')
              << int(getbyte(i));
        }
      }
      else {
        for (ssize_t i = 0;  i < 16;  i++) {
          if (i != 0  &&  i % 4 == 0) {
            out << " ";
          }
          out << std::hex << std::setw(2) << std::setfill('0')
              << int(getbyte(i));
        }
        out << " ... ";
        for (ssize_t i = len - 16;  i < len;  i++) {
          if (i != len - 16  &&  i % 4 == 0) {
            out << " ";
          }
          out << std::hex << std::setw(2) << std::setfill('0')
              << int(getbyte(i));
        }
      }
      out << "\" at=\"0x";
      out << std::hex << std::setw(12) << std::setfill('0')
          << reinterpret_cast<ssize_t>(ptr_.get());
      if (identities_.get() == nullptr  &&  parameters_.empty()) {
        out << "\"/>" << post;
      }
      else {
        out << "\">\n";
        out << identities_.get()->tostring_part(indent
                 + std::string("    "), "", "\n");
        if (!parameters_.empty()) {
          out << parameters_tostring(indent + std::string("    "), "", "\n");
        }
        out << indent << "</RawArray>" << post;
      }
      return out.str();
    }

    void
      tojson_part(ToJson& builder) const override {
      if (std::is_same<T, double>::value) {
        tojson_real(builder,
                    reinterpret_cast<double*>(byteptr()), length());
      }
      else if (std::is_same<T, float>::value) {
        tojson_real(builder,
                    reinterpret_cast<float*>(byteptr()), length());
      }
      else if (std::is_same<T, int64_t>::value) {
        tojson_integer(builder,
                       reinterpret_cast<int64_t*>(byteptr()), length());
      }
      else if (std::is_same<T, uint64_t>::value) {
        tojson_integer(builder,
                       reinterpret_cast<uint64_t*>(byteptr()), length());
      }
      else if (std::is_same<T, int32_t>::value) {
        tojson_integer(builder,
                       reinterpret_cast<int32_t*>(byteptr()), length());
      }
      else if (std::is_same<T, uint32_t>::value) {
        tojson_integer(builder,
                       reinterpret_cast<uint32_t*>(byteptr()), length());
      }
      else if (std::is_same<T, int16_t>::value) {
        tojson_integer(builder,
                       reinterpret_cast<int16_t*>(byteptr()), length());
      }
      else if (std::is_same<T, uint16_t>::value) {
        tojson_integer(builder,
                       reinterpret_cast<uint16_t*>(byteptr()), length());
      }
      else if (std::is_same<T, int8_t>::value) {
        tojson_integer(builder,
                       reinterpret_cast<int8_t*>(byteptr()), length());
      }
      else if (std::is_same<T, uint8_t>::value) {
        tojson_integer(builder,
                       reinterpret_cast<uint8_t*>(byteptr()), length());
      }
      else if (std::is_same<T, bool>::value) {
        tojson_boolean(builder,
                       reinterpret_cast<bool*>(byteptr()), length());
      }
      else {
        throw std::invalid_argument(std::string("cannot convert RawArrayOf<")
          + typeid(T).name() + std::string("> into JSON"));
      }
    }

    int64_t
      length() const override {
      return length_;
    }

    void
      nbytes_part(std::map<size_t, int64_t>& largest) const override {
      size_t x = (size_t)ptr_.get();
      auto it = largest.find(x);
      if (it == largest.end()  ||  it->second < (int64_t)(sizeof(T)*length_)) {
        largest[x] = (int64_t)(sizeof(T)*length_);
      }
      if (identities_.get() != nullptr) {
        identities_.get()->nbytes_part(largest);
      }
    }

    const ContentPtr
      shallow_copy() const override {
      return std::make_shared<RawArrayOf<T>>(identities_, parameters_, ptr_,
                                             offset_, length_, itemsize_);
    }

    const ContentPtr
      deep_copy(bool copyarrays,
                bool copyindexes,
                bool copyidentities) const override {
      std::shared_ptr<T> ptr = ptr_;
      int64_t offset = offset_;
      if (copyarrays) {
        ptr = std::shared_ptr<T>(new T[(size_t)length_],
                                 util::array_deleter<T>());
        memcpy(ptr.get(), &ptr_.get()[(size_t)offset_],
               sizeof(T)*((size_t)length_));
        offset = 0;
      }
      IdentitiesPtr identities = identities_;
      if (copyidentities  &&  identities_.get() != nullptr) {
        identities = identities_.get()->deep_copy();
      }
      return std::make_shared<RawArrayOf<T>>(identities,
                                             parameters_,
                                             ptr,
                                             offset,
                                             length_,
                                             itemsize_);
    }

    void
      check_for_iteration() const override {
      if (identities_.get() != nullptr  &&
          identities_.get()->length() < length_) {
        util::handle_error(failure("len(identities) < len(array)", kSliceNone,
          kSliceNone), identities_.get()->classname(), nullptr);
      }
    }

    const ContentPtr
      getitem_nothing() const override {
      return getitem_range_nowrap(0, 0);
    }

    const ContentPtr
      getitem_at(int64_t at) const override {
      int64_t regular_at = at;
      if (regular_at < 0) {
        regular_at += length_;
      }
      if (!(0 <= regular_at  &&  regular_at < length_)) {
        util::handle_error(failure("index out of range", kSliceNone, at),
                           classname(),
                           identities_.get());
      }
      return getitem_at_nowrap(regular_at);
    }

    const ContentPtr
      getitem_at_nowrap(int64_t at) const override {
      return getitem_range_nowrap(at, at + 1);
    }

    const ContentPtr
      getitem_range(int64_t start, int64_t stop) const override {
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, true,
        start != Slice::none(), stop != Slice::none(), length_);
      if (identities_.get() != nullptr  &&
          regular_stop > identities_.get()->length()) {
        util::handle_error(failure("index out of range", kSliceNone, stop),
          identities_.get()->classname(), nullptr);
      }
      return getitem_range_nowrap(regular_start, regular_stop);
    }

    const ContentPtr
      getitem_range_nowrap(int64_t start, int64_t stop) const override {
      IdentitiesPtr identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_range_nowrap(start, stop);
      }
      return std::make_shared<RawArrayOf<T>>(identities,
                                             parameters_,
                                             ptr_,
                                             offset_ + start,
                                             stop - start,
                                             itemsize_);
    }

    const ContentPtr
      getitem_field(const std::string& key) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname()
        + std::string(" by field name"));
    }

    const ContentPtr
      getitem_fields(const std::vector<std::string>& keys) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname()
        + std::string(" by field name"));
    }

    const ContentPtr
      getitem(const Slice& where) const override {
      SliceItemPtr nexthead = where.head();
      Slice nexttail = where.tail();
      Index64 nextadvanced(0);
      return getitem_next(nexthead, nexttail, nextadvanced);
    }

    const ContentPtr
      getitem_next(const SliceItemPtr& head,
                   const Slice& tail,
                   const Index64& advanced) const override {
      if (tail.length() != 0) {
        throw std::invalid_argument("too many indexes for array");
      }
      return Content::getitem_next(head, tail, advanced);
    }

    const ContentPtr carry(const Index64& carry) const override {
      std::shared_ptr<T> ptr(new T[(size_t)carry.length()],
                             util::array_deleter<T>());
      struct Error err = awkward_numpyarray_getitem_next_null_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        carry.length(),
        itemsize_,
        byteoffset(),
        carry.ptr().get());
      util::handle_error(err, classname(), identities_.get());

      IdentitiesPtr identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_carry_64(carry);
      }

      return std::make_shared<RawArrayOf<T>>(identities,
                                             parameters_,
                                             ptr,
                                             0,
                                             carry.length(),
                                             itemsize_);
    }

    const std::string
      purelist_parameter(const std::string& key) const override {
      return parameter(key);
    }

    bool
      purelist_isregular() const override {
      return true;
    }

    int64_t
      purelist_depth() const override {
      return 1;
    }

    const std::pair<int64_t, int64_t>
      minmax_depth() const override {
      return std::pair<int64_t, int64_t>(1, 1);
    }

    const std::pair<bool, int64_t>
      branch_depth() const override {
      return std::pair<bool, int64_t>(false, 1);
    }

    int64_t
      numfields() const override {
      return -1;
    }

    int64_t
      fieldindex(const std::string& key) const override {
      throw std::invalid_argument(std::string("key ") + util::quote(key, true)
        + std::string(" does not exist (data are not records)"));
    }

    const std::string
      key(int64_t fieldindex) const override {
      throw std::invalid_argument(std::string("fieldindex \"")
        + std::to_string(fieldindex)
        + std::string("\" does not exist (data are not records)"));
    }

    bool
      haskey(const std::string& key) const override {
      return false;
    }

    const std::vector<std::string>
      keys() const override {
      return std::vector<std::string>();
    }

    // operations

    const std::string
      validityerror(const std::string& path) const override {
      return std::string();
    }

    /// @copydoc Content::shallow_simplify()
    ///
    /// For {@link RawArrayOf RawArray}, this method returns #shallow_copy
    /// (pass-through).
    const ContentPtr
      shallow_simplify() const override {
      return shallow_copy();
    }

    const ContentPtr
      num(int64_t axis, int64_t depth) const override {
      int64_t toaxis = axis_wrap_if_negative(axis);
      if (toaxis == depth) {
        Index64 out(1);
        out.setitem_at_nowrap(0, length());
        return std::make_shared<RawArrayOf<int64_t>>(Identities::none(),
                                                     util::Parameters(),
                                                     out.ptr(),
                                                     0,
                                                     1,
                                                     sizeof(int64_t));
      }
      else {
        throw std::invalid_argument("'axis' out of range for 'num'");
      }
    }

    const std::pair<Index64, ContentPtr>
      offsets_and_flattened(int64_t axis, int64_t depth) const override {
      int64_t toaxis = axis_wrap_if_negative(axis);
      if (toaxis == depth) {
        throw std::invalid_argument("axis=0 not allowed for flatten");
      }
      else {
        throw std::invalid_argument("axis out of range for flatten");
      }
    }

    bool
      mergeable(const ContentPtr& other, bool mergebool) const override {
      if (dynamic_cast<EmptyArray*>(other.get())) {
        return true;
      }
      else if (IndexedArray32* rawother =
               dynamic_cast<IndexedArray32*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (IndexedArrayU32* rawother =
               dynamic_cast<IndexedArrayU32*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (IndexedArray64* rawother =
               dynamic_cast<IndexedArray64*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (IndexedOptionArray32* rawother =
               dynamic_cast<IndexedOptionArray32*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (IndexedOptionArray64* rawother =
               dynamic_cast<IndexedOptionArray64*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (ByteMaskedArray* rawother =
               dynamic_cast<ByteMaskedArray*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (BitMaskedArray* rawother =
               dynamic_cast<BitMaskedArray*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }
      else if (UnmaskedArray* rawother =
               dynamic_cast<UnmaskedArray*>(other.get())) {
        return mergeable(rawother->content(), mergebool);
      }

      if (RawArrayOf<T>* rawother =
          dynamic_cast<RawArrayOf<T>*>(other.get())) {
        return true;
      }
      else {
        return false;
      }
    }

    const ContentPtr
      merge(const ContentPtr& other) const override {
      if (dynamic_cast<EmptyArray*>(other.get())) {
        return shallow_copy();
      }
      else if (IndexedArray32* rawother =
               dynamic_cast<IndexedArray32*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (IndexedArrayU32* rawother =
               dynamic_cast<IndexedArrayU32*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (IndexedArray64* rawother =
               dynamic_cast<IndexedArray64*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (IndexedOptionArray32* rawother =
               dynamic_cast<IndexedOptionArray32*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (IndexedOptionArray64* rawother =
               dynamic_cast<IndexedOptionArray64*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (ByteMaskedArray* rawother =
               dynamic_cast<ByteMaskedArray*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (BitMaskedArray* rawother =
               dynamic_cast<BitMaskedArray*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }
      else if (UnmaskedArray* rawother =
               dynamic_cast<UnmaskedArray*>(other.get())) {
        return rawother->reverse_merge(shallow_copy());
      }

      if (RawArrayOf<T>* rawother =
          dynamic_cast<RawArrayOf<T>*>(other.get())) {
        std::shared_ptr<T> ptr =
          std::shared_ptr<T>(new T[(size_t)(length_ + rawother->length())],
                             util::array_deleter<T>());
        memcpy(ptr.get(),
               &ptr_.get()[(size_t)offset_],
               sizeof(T)*((size_t)length_));
        memcpy(&ptr.get()[(size_t)length_],
               &rawother->ptr().get()[(size_t)rawother->offset()],
               sizeof(T)*((size_t)rawother->length()));
        return std::make_shared<RawArrayOf<T>>(Identities::none(),
                                               util::Parameters(),
                                               ptr,
                                               0,
                                               length_ + rawother->length(),
                                               itemsize_);
      }
      else {
        throw std::invalid_argument(std::string("cannot merge ") + classname()
          + std::string(" with ") + other.get()->classname());
      }
    }

    const SliceItemPtr
      asslice() const override {
      throw std::invalid_argument("cannot use RawArray as a slice");
    }

    const ContentPtr
      fillna(const ContentPtr& value) const override {
      return shallow_copy();
    }

    const ContentPtr
      rpad(int64_t target, int64_t axis, int64_t depth) const override {
      int64_t toaxis = axis_wrap_if_negative(axis);
      if (toaxis != depth) {
        throw std::invalid_argument("axis exceeds the depth of this array");
      }
      if (target < length()) {
        return shallow_copy();
      }
      else {
        return rpad_and_clip(target, toaxis, depth);
      }
    }

    const ContentPtr
      rpad_and_clip(int64_t target,
                    int64_t axis,
                    int64_t depth) const override {
      int64_t toaxis = axis_wrap_if_negative(axis);
      if (toaxis != depth) {
        throw std::invalid_argument("axis exceeds the depth of this array");
      }
      Index64 index(target);
      struct Error err = awkward_index_rpad_and_clip_axis0_64(
        index.ptr().get(),
        target,
        length());
      util::handle_error(err, classname(), identities_.get());

      return std::make_shared<IndexedOptionArray64>(Identities::none(),
                                                    util::Parameters(),
                                                    index,
                                                    shallow_copy());
    }

    const ContentPtr
      reduce_next(const Reducer& reducer,
                  int64_t negaxis,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength,
                  bool mask,
                  bool keepdims) const override {
      throw std::runtime_error("FIXME: RawArray:reduce_next");
    }

    const ContentPtr
      sort_next(int64_t negaxis,
                const Index64& starts,
                const Index64& parents,
                int64_t outlength,
                bool ascending,
                bool stable) const override {
      throw std::runtime_error("FIXME: RawArray:sort_next");
    }

    const ContentPtr
      localindex(int64_t axis, int64_t depth) const override {
      int64_t toaxis = axis_wrap_if_negative(axis);
      if (axis == depth) {
        return localindex_axis0();
      }
      else {
        throw std::invalid_argument("'axis' out of range for localindex");
      }
    }

    const ContentPtr
      choose(int64_t n,
             bool diagonal,
             const util::RecordLookupPtr& recordlookup,
             const util::Parameters& parameters,
             int64_t axis,
             int64_t depth) const override {
      if (n < 1) {
        throw std::invalid_argument("in choose, 'n' must be at least 1");
      }
      int64_t toaxis = axis_wrap_if_negative(axis);
      if (toaxis == depth) {
        return choose_axis0(n, diagonal, recordlookup, parameters);
      }
      else {
        throw std::invalid_argument("'axis' out of range for choose");
      }
    }

    const ContentPtr
      getitem_next(const SliceAt& at,
                   const Slice& tail,
                   const Index64& advanced) const override {
      return getitem_at(at.at());
    }

    const ContentPtr
      getitem_next(const SliceRange& range,
                   const Slice& tail,
                   const Index64& advanced) const override {
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
        awkward_regularize_rangeslice(&start, &stop, step > 0,
          range.hasstart(), range.hasstop(), length_);

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

    const ContentPtr
      getitem_next(const SliceArray64& array,
                   const Slice& tail,
                   const Index64& advanced) const override {
      if (advanced.length() != 0) {
        throw std::runtime_error(
          "RawArray::getitem_next(SliceAt): advanced.length() != 0");
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

    const ContentPtr
      getitem_next(const SliceField& field,
                   const Slice& tail,
                   const Index64& advanced) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname()
        + std::string(" by a field name because it has no fields"));
    }

    const ContentPtr
      getitem_next(const SliceFields& fields,
                   const Slice& tail,
                   const Index64& advanced) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname()
        + std::string(" by field names because it has no fields"));
    }

    const ContentPtr
      getitem_next(const SliceJagged64& jagged,
                   const Slice& tail,
                   const Index64& advanced) const override {
      throw std::invalid_argument(std::string("cannot slice ") + classname()
        + std::string(" by a jagged array because it is one-dimensional"));
    }

    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceArray64& slicecontent,
                          const Slice& tail) const override {
      throw std::runtime_error(
        "undefined operation: RawArray::getitem_next_jagged(array)");
    }

    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceMissing64& slicecontent,
                          const Slice& tail) const override {
      throw std::runtime_error(
        "undefined operation: RawArray::getitem_next_jagged(missing)");
    }

    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceJagged64& slicecontent,
                          const Slice& tail) const override {
      throw std::runtime_error(
        "undefined operation: RawArray::getitem_next_jagged(jagged)");
    }

  private:
    const std::shared_ptr<T> ptr_;
    const int64_t offset_;
    const int64_t length_;
    const int64_t itemsize_;
  };
}

#endif // AWKWARD_RAWARRAY_H_
