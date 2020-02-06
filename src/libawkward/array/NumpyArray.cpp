// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/type/RegularType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/util.h"

#include "awkward/array/NumpyArray.h"

namespace awkward {
  const std::shared_ptr<Type> NumpyArray::unwrap_regulartype(const std::shared_ptr<Type>& type, const std::vector<ssize_t>& shape) {
    std::shared_ptr<Type> out = type;
    for (size_t i = 1;  i < shape.size();  i++) {
      if (RegularType* raw = dynamic_cast<RegularType*>(out.get())) {
        if (raw->size() == (int64_t)shape[i]) {
          out = raw->type();
        }
        else {
          throw std::invalid_argument(std::string("NumpyArray cannot be converted to type ") + type.get()->tostring() + std::string(" because shape does not match sizes of RegularTypes"));
        }
      }
      else {
        throw std::invalid_argument(std::string("NumpyArray cannot be converted to type ") + type.get()->tostring() + std::string(" because shape does not match level of RegularType nesting"));
      }
    }
    return out;
  }

  const std::unordered_map<std::type_index, std::string> NumpyArray::format_map = {
    { typeid(int8_t), "b"},
    { typeid(uint8_t), "B"},
 #ifdef _MSC_VER
    { typeid(int32_t), "l"},
    { typeid(uint32_t), "L"},
    { typeid(int64_t), "q"}
 #else
    { typeid(int32_t), "i"},
    { typeid(uint32_t), "I"},
    { typeid(int64_t), "l"}
 #endif
 };

  NumpyArray::NumpyArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::shared_ptr<void>& ptr, const std::vector<ssize_t>& shape, const std::vector<ssize_t>& strides, ssize_t byteoffset, ssize_t itemsize, const std::string format)
      : Content(identities, parameters)
      , ptr_(ptr)
      , shape_(shape)
      , strides_(strides)
      , byteoffset_(byteoffset)
      , itemsize_(itemsize)
      , format_(format) {
    if (shape.size() != strides.size()) {
      throw std::runtime_error(std::string("len(shape), which is ") + std::to_string(shape.size()) + std::string(", must be equal to len(strides), which is ") + std::to_string(strides.size()));
    }
  }

  NumpyArray::NumpyArray(const Index8 index)
    : NumpyArray(index, format_map.at(std::type_index(typeid(int8_t)))) { }

  NumpyArray::NumpyArray(const IndexU8 index)
    : NumpyArray(index, format_map.at(std::type_index(typeid(uint8_t)))) { }

  NumpyArray::NumpyArray(const Index32 index)
    : NumpyArray(index, format_map.at(std::type_index(typeid(int32_t)))) { }

  NumpyArray::NumpyArray(const IndexU32 index)
    : NumpyArray(index, format_map.at(std::type_index(typeid(uint32_t)))) { }

  NumpyArray::NumpyArray(const Index64 index)
    : NumpyArray(index, format_map.at(std::type_index(typeid(int64_t)))) { }

  NumpyArray::NumpyArray(const Index8 index, const std::string& format)
    : NumpyArray(Identities::none(), util::Parameters(), index.ptr(), std::vector<ssize_t>({ (ssize_t)index.length() }), std::vector<ssize_t>({ (ssize_t)sizeof(int8_t) }), 0, sizeof(int8_t), format) { }

  NumpyArray::NumpyArray(const IndexU8 index, const std::string& format)
    : NumpyArray(Identities::none(), util::Parameters(), index.ptr(), std::vector<ssize_t>({ (ssize_t)index.length() }), std::vector<ssize_t>({ (ssize_t)sizeof(uint8_t) }), 0, sizeof(uint8_t), format) { }

  NumpyArray::NumpyArray(const Index32 index, const std::string& format)
    : NumpyArray(Identities::none(), util::Parameters(), index.ptr(), std::vector<ssize_t>({ (ssize_t)index.length() }), std::vector<ssize_t>({ (ssize_t)sizeof(int32_t) }), 0, sizeof(int32_t), format) { }

  NumpyArray::NumpyArray(const IndexU32 index, const std::string& format)
    : NumpyArray(Identities::none(), util::Parameters(), index.ptr(), std::vector<ssize_t>({ (ssize_t)index.length() }), std::vector<ssize_t>({ (ssize_t)sizeof(uint32_t) }), 0, sizeof(uint32_t), format) { }

  NumpyArray::NumpyArray(const Index64 index, const std::string& format)
    : NumpyArray(Identities::none(), util::Parameters(), index.ptr(), std::vector<ssize_t>({ (ssize_t)index.length() }), std::vector<ssize_t>({ (ssize_t)sizeof(int64_t) }), 0, sizeof(int64_t), format) { }

  const std::shared_ptr<void> NumpyArray::ptr() const {
    return ptr_;
  }

  const std::vector<ssize_t> NumpyArray::shape() const {
    return shape_;
  }

  const std::vector<ssize_t> NumpyArray::strides() const {
    return strides_;
  }

  ssize_t NumpyArray::byteoffset() const {
    return byteoffset_;
  }

  ssize_t NumpyArray::itemsize() const {
    return itemsize_;
  }

  const std::string NumpyArray::format() const {
    return format_;
  }

  ssize_t NumpyArray::ndim() const {
    return (ssize_t)shape_.size();
  }

  bool NumpyArray::isempty() const {
    for (auto x : shape_) {
      if (x == 0) {
        return true;
      }
    }
    return false;  // false for isscalar(), too
  }

  void* NumpyArray::byteptr() const {
    return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_);
  }

  void* NumpyArray::byteptr(ssize_t at) const {
    return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_ + at);
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
    return *reinterpret_cast<uint8_t*>(byteptr(at));
  }

  int8_t NumpyArray::getint8(ssize_t at) const  {
    return *reinterpret_cast<int8_t*>(byteptr(at));
  }

  uint8_t NumpyArray::getuint8(ssize_t at) const {
    return *reinterpret_cast<uint8_t*>(byteptr(at));
  }

  int16_t NumpyArray::getint16(ssize_t at) const {
    return *reinterpret_cast<int16_t*>(byteptr(at));
  }

  uint16_t NumpyArray::getuint16(ssize_t at) const {
    return *reinterpret_cast<uint16_t*>(byteptr(at));
  }

  int32_t NumpyArray::getint32(ssize_t at) const {
    return *reinterpret_cast<int32_t*>(byteptr(at));
  }

  uint32_t NumpyArray::getuint32(ssize_t at) const {
    return *reinterpret_cast<uint32_t*>(byteptr(at));
  }

  int64_t NumpyArray::getint64(ssize_t at) const {
    return *reinterpret_cast<int64_t*>(byteptr(at));
  }

  uint64_t NumpyArray::getuint64(ssize_t at) const {
    return *reinterpret_cast<uint64_t*>(byteptr(at));
  }

  float_t NumpyArray::getfloat(ssize_t at) const {
    return *reinterpret_cast<float*>(byteptr(at));
  }

  double_t NumpyArray::getdouble(ssize_t at) const {
    return *reinterpret_cast<double*>(byteptr(at));
  }

  const std::shared_ptr<Content> NumpyArray::toRegularArray() const {
    if (isscalar()) {
      return shallow_copy();
    }
    NumpyArray contiguous_self = contiguous();
    std::vector<ssize_t> flatshape({ 1 });
    for (auto x : shape_) {
      flatshape[0] = flatshape[0] * x;
    }
    std::vector<ssize_t> flatstrides({ itemsize_ });
    std::shared_ptr<Content> out = std::make_shared<NumpyArray>(identities_, parameters_, contiguous_self.ptr(), flatshape, flatstrides, contiguous_self.byteoffset(), contiguous_self.itemsize(), contiguous_self.format());
    for (int64_t i = (int64_t)shape_.size() - 1;  i > 0;  i--) {
      out = std::make_shared<RegularArray>(Identities::none(), util::Parameters(), out, shape_[(size_t)i]);
    }
    return out;
  }

  bool NumpyArray::isscalar() const {
    return ndim() == 0;
  }

  const std::string NumpyArray::classname() const {
    return "NumpyArray";
  }

  void NumpyArray::setidentities(const std::shared_ptr<Identities>& identities) {
    if (identities.get() != nullptr  &&  length() != identities.get()->length()) {
      util::handle_error(failure("content and its identities must have the same length", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    identities_ = identities;
  }

  void NumpyArray::setidentities() {
    assert(!isscalar());
    if (length() <= kMaxInt32) {
      std::shared_ptr<Identities> newidentities = std::make_shared<Identities32>(Identities::newref(), Identities::FieldLoc(), 1, length());
      Identities32* rawidentities = reinterpret_cast<Identities32*>(newidentities.get());
      struct Error err = awkward_new_identities32(rawidentities->ptr().get(), length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
    else {
      std::shared_ptr<Identities> newidentities = std::make_shared<Identities64>(Identities::newref(), Identities::FieldLoc(), 1, length());
      Identities64* rawidentities = reinterpret_cast<Identities64*>(newidentities.get());
      struct Error err = awkward_new_identities64(rawidentities->ptr().get(), length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  template <typename T>
  void tostring_as(std::stringstream& out, T* ptr, int64_t length) {
    if (length <= 10) {
      for (int64_t i = 0;  i < length;  i++) {
        if (i != 0) {
          out << " ";
        }
        if (std::is_same<T, bool>::value) {
          out << (ptr[i] ? "true" : "false");
        }
        else {
          out << ptr[i];
        }
      }
    }
    else {
      for (int64_t i = 0;  i < 5;  i++) {
        if (i != 0) {
          out << " ";
        }
        if (std::is_same<T, bool>::value) {
          out << (ptr[i] ? "true" : "false");
        }
        else {
          out << ptr[i];
        }
      }
      out << " ... ";
      for (int64_t i = length - 5;  i < length;  i++) {
        if (i != length - 5) {
          out << " ";
        }
        if (std::is_same<T, bool>::value) {
          out << (ptr[i] ? "true" : "false");
        }
        else {
          out << ptr[i];
        }
      }
    }
  }

  const std::shared_ptr<Type> NumpyArray::type() const {
    std::shared_ptr<Type> out;
    if (format_.compare("d") == 0) {
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::float64);
    }
    else if (format_.compare("f") == 0) {
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::float32);
    }
#ifdef _MSC_VER
    else if (format_.compare("q") == 0) {
#else
    else if (format_.compare("l") == 0) {
#endif
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::int64);
    }
#ifdef _MSC_VER
    else if (format_.compare("Q") == 0) {
#else
    else if (format_.compare("L") == 0) {
#endif
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::uint64);
    }
#ifdef _MSC_VER
    else if (format_.compare("l") == 0) {
#else
    else if (format_.compare("i") == 0) {
#endif
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::int32);
    }
#ifdef _MSC_VER
    else if (format_.compare("L") == 0) {
#else
    else if (format_.compare("I") == 0) {
#endif
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::uint32);
    }
    else if (format_.compare("h") == 0) {
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::int16);
    }
    else if (format_.compare("H") == 0) {
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::uint16);
    }
    else if (format_.compare("b") == 0) {
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::int8);
    }
    else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::uint8);
    }
    else if (format_.compare("?") == 0) {
      out = std::make_shared<PrimitiveType>(parameters_, PrimitiveType::boolean);
    }
    else {
      throw std::invalid_argument(std::string("Numpy format \"") + format_ + std::string("\" cannot be expressed as a PrimitiveType"));
    }
    for (std::size_t i = shape_.size() - 1;  i > 0;  i--) {
      out = std::make_shared<RegularType>(util::Parameters(), out, (int64_t)shape_[i]);
    }
    return out;
  }

  const std::shared_ptr<Content> NumpyArray::astype(const std::shared_ptr<Type>& type) const {
    // FIXME: if the unwrapped_type does not match the format_, actually convert it!
    // Maybe also change the shape_ if there's a different RegularType nesting (less strict than unwrap_regulartype).
    std::shared_ptr<Type> unwrapped_type = unwrap_regulartype(type, shape_);
    return std::make_shared<NumpyArray>(identities_, unwrapped_type.get()->parameters(), ptr_, shape_, strides_, byteoffset_, itemsize_, format_);
  }

  const std::string NumpyArray::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    assert(!isscalar());
    std::stringstream out;
    out << indent << pre << "<" << classname() << " format=" << util::quote(format_, true) << " shape=\"";
    for (std::size_t i = 0;  i < shape_.size();  i++) {
      if (i != 0) {
        out << " ";
      }
      out << shape_[i];
    }
    out << "\" ";
    if (!iscontiguous()) {
      out << "strides=\"";
      for (std::size_t i = 0;  i < shape_.size();  i++) {
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
    else if (ndim() == 1  &&  format_.compare("?") == 0) {
      tostring_as<bool>(out, reinterpret_cast<bool*>(byteptr()), length());
    }
    else {
      out << "0x ";
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
    if (identities_.get() == nullptr  &&  parameters_.empty()) {
      out << "\"/>" << post;
    }
    else {
      out << "\">\n";
      if (identities_.get() != nullptr) {
        out << identities_.get()->tostring_part(indent + std::string("    "), "", "\n");
      }
      if (!parameters_.empty()) {
        out << parameters_tostring(indent + std::string("    "), "", "\n");
      }
      out << indent << "</" << classname() << ">" << post;
    }
    return out.str();
  }

  void NumpyArray::tojson_part(ToJson& builder) const {
    check_for_iteration();
    if (parameter_equals("__array__", "\"char\"")) {
      tojson_string(builder);
    }
    else if (format_.compare("d") == 0) {
      tojson_real<double>(builder);
    }
    else if (format_.compare("f") == 0) {
      tojson_real<float>(builder);
    }
#ifdef _MSC_VER
    else if (format_.compare("q") == 0) {
#else
    else if (format_.compare("l") == 0) {
#endif
      tojson_integer<int64_t>(builder);
    }
#ifdef _MSC_VER
    else if (format_.compare("Q") == 0) {
#else
    else if (format_.compare("L") == 0) {
#endif
      tojson_integer<uint64_t>(builder);
    }
#ifdef _MSC_VER
      else if (format_.compare("l") == 0) {
#else
      else if (format_.compare("i") == 0) {
#endif
      tojson_integer<int32_t>(builder);
    }
#ifdef _MSC_VER
    else if (format_.compare("L") == 0) {
#else
    else if (format_.compare("I") == 0) {
#endif
      tojson_integer<uint32_t>(builder);
    }
    else if (format_.compare("h") == 0) {
      tojson_integer<int16_t>(builder);
    }
    else if (format_.compare("H") == 0) {
      tojson_integer<uint16_t>(builder);
    }
    else if (format_.compare("b") == 0) {
      tojson_integer<int8_t>(builder);
    }
    else if (format_.compare("B") == 0) {
      tojson_integer<uint8_t>(builder);
    }
    else if (format_.compare("?") == 0) {
      tojson_boolean(builder);
    }
    else {
      throw std::invalid_argument(std::string("cannot convert Numpy format \"") + format_ + std::string("\" into JSON"));
    }
  }

  void NumpyArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    int64_t len = 1;
    if (!shape_.empty()) {
      len = shape_[0];
    }
    size_t x = (size_t)ptr_.get();
    auto it = largest.find(x);
    if (it == largest.end()  ||  it->second < (int64_t)(itemsize_*len)) {
      largest[x] = (int64_t)(itemsize_*len);
    }
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t NumpyArray::length() const {
    if (isscalar()) {
      return -1;   // just like Record, which is also a scalar
    }
    else {
      return (int64_t)shape_[0];
    }
  }

  const std::shared_ptr<Content> NumpyArray::shallow_copy() const {
    return std::make_shared<NumpyArray>(identities_, parameters_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_);
  }

  const std::shared_ptr<Content> NumpyArray::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    std::shared_ptr<void> ptr = ptr_;
    std::vector<ssize_t> shape = shape_;
    std::vector<ssize_t> strides = strides_;
    ssize_t byteoffset = byteoffset_;
    if (copyarrays) {
      NumpyArray tmp = contiguous();
      ptr = tmp.ptr();
      shape = tmp.shape();
      strides = tmp.strides();
      byteoffset = tmp.byteoffset();
    }
    std::shared_ptr<Identities> identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<NumpyArray>(identities, parameters_, ptr, shape, strides, byteoffset, itemsize_, format_);
  }

  void NumpyArray::check_for_iteration() const {
    if (identities_.get() != nullptr  &&  identities_.get()->length() < shape_[0]) {
      util::handle_error(failure("len(identities) < len(array)", kSliceNone, kSliceNone), identities_.get()->classname(), nullptr);
    }
  }

  const std::shared_ptr<Content> NumpyArray::getitem_nothing() const {
    const std::vector<ssize_t> shape({ 0 });
    const std::vector<ssize_t> strides({ itemsize_ });
    std::shared_ptr<Identities> identities;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(0, 0);
    }
    return std::make_shared<NumpyArray>(identities, parameters_, ptr_, shape, strides, byteoffset_, itemsize_, format_);
  }

  const std::shared_ptr<Content> NumpyArray::getitem_at(int64_t at) const {
    assert(!isscalar());
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += shape_[0];
    }
    if (regular_at < 0  ||  regular_at >= shape_[0]) {
      util::handle_error(failure("index out of range", kSliceNone, at), classname(), identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const std::shared_ptr<Content> NumpyArray::getitem_at_nowrap(int64_t at) const {
    ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)at);
    const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
    const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
    std::shared_ptr<Identities> identities;
    if (identities_.get() != nullptr) {
      if (at >= identities_.get()->length()) {
        util::handle_error(failure("index out of range", kSliceNone, at), identities_.get()->classname(), nullptr);
      }
      identities = identities_.get()->getitem_range_nowrap(at, at + 1);
    }
    return std::make_shared<NumpyArray>(identities, parameters_, ptr_, shape, strides, byteoffset, itemsize_, format_);
  }

  const std::shared_ptr<Content> NumpyArray::getitem_range(int64_t start, int64_t stop) const {
    assert(!isscalar());
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), shape_[0]);
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  const std::shared_ptr<Content> NumpyArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)start);
    std::vector<ssize_t> shape;
    shape.push_back((ssize_t)(stop - start));
    shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
    std::shared_ptr<Identities> identities;
    if (identities_.get() != nullptr) {
      if (stop > identities_.get()->length()) {
        util::handle_error(failure("index out of range", kSliceNone, stop), identities_.get()->classname(), nullptr);
      }
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<NumpyArray>(identities, parameters_, ptr_, shape, strides_, byteoffset, itemsize_, format_);
  }

  const std::shared_ptr<Content> NumpyArray::getitem_field(const std::string& key) const {
    throw std::invalid_argument(std::string("cannot slice ") + classname() + std::string(" by field name"));
  }

  const std::shared_ptr<Content> NumpyArray::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::invalid_argument(std::string("cannot slice ") + classname() + std::string(" by field name"));
  }

  const std::shared_ptr<Content> NumpyArray::getitem(const Slice& where) const {
    assert(!isscalar());

    if (!where.isadvanced()  &&  identities_.get() == nullptr) {
      std::vector<ssize_t> nextshape = { 1 };
      nextshape.insert(nextshape.end(), shape_.begin(), shape_.end());
      std::vector<ssize_t> nextstrides = { shape_[0]*strides_[0] };
      nextstrides.insert(nextstrides.end(), strides_.begin(), strides_.end());
      NumpyArray next(identities_, parameters_, ptr_, nextshape, nextstrides, byteoffset_, itemsize_, format_);

      std::shared_ptr<SliceItem> nexthead = where.head();
      Slice nexttail = where.tail();
      NumpyArray out = next.getitem_bystrides(nexthead, nexttail, 1);

      std::vector<ssize_t> outshape(out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides(out.strides_.begin() + 1, out.strides_.end());
      return std::make_shared<NumpyArray>(out.identities_, out.parameters_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
    }

    else {
      NumpyArray safe = contiguous();   // maybe become_contiguous() to change in-place?

      std::vector<ssize_t> nextshape = { 1 };
      nextshape.insert(nextshape.end(), safe.shape_.begin(), safe.shape_.end());
      std::vector<ssize_t> nextstrides = { safe.shape_[0]*safe.strides_[0] };
      nextstrides.insert(nextstrides.end(), safe.strides_.begin(), safe.strides_.end());
      NumpyArray next(safe.identities_, safe.parameters_, safe.ptr_, nextshape, nextstrides, safe.byteoffset_, itemsize_, format_);

      std::shared_ptr<SliceItem> nexthead = where.head();
      Slice nexttail = where.tail();
      Index64 nextcarry(1);
      nextcarry.ptr().get()[0] = 0;
      Index64 nextadvanced(0);
      NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, 1, next.strides_[0], true);

      std::vector<ssize_t> outshape(out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides(out.strides_.begin() + 1, out.strides_.end());
      return std::make_shared<NumpyArray>(out.identities_, out.parameters_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
    }
  }

  const std::shared_ptr<Content> NumpyArray::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    assert(!isscalar());
    Index64 carry(shape_[0]);
    struct Error err = awkward_carry_arange_64(carry.ptr().get(), shape_[0]);
    util::handle_error(err, classname(), identities_.get());
    return getitem_next(head, tail, carry, advanced, shape_[0], strides_[0], false).shallow_copy();
  }

  const std::shared_ptr<Content> NumpyArray::carry(const Index64& carry) const {
    assert(!isscalar());

    std::shared_ptr<void> ptr(new uint8_t[(size_t)(carry.length()*strides_[0])], util::array_deleter<uint8_t>());
    struct Error err = awkward_numpyarray_getitem_next_null_64(
      reinterpret_cast<uint8_t*>(ptr.get()),
      reinterpret_cast<uint8_t*>(ptr_.get()),
      carry.length(),
      strides_[0],
      byteoffset_,
      carry.ptr().get());
    util::handle_error(err, classname(), identities_.get());

    std::shared_ptr<Identities> identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }

    std::vector<ssize_t> shape = { (ssize_t)carry.length() };
    shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
    return std::make_shared<NumpyArray>(identities, parameters_, ptr, shape, strides_, 0, itemsize_, format_);
  }

  const std::string NumpyArray::purelist_parameter(const std::string& key) const {
    return parameter(key);
  }

  bool NumpyArray::purelist_isregular() const {
    return true;
  }

  int64_t NumpyArray::purelist_depth() const {
    return (int64_t)shape_.size();
  }

  const std::pair<int64_t, int64_t> NumpyArray::minmax_depth() const {
    return std::pair<int64_t, int64_t>((int64_t)shape_.size(), (int64_t)shape_.size());
  }

  int64_t NumpyArray::numfields() const { return -1; }

  int64_t NumpyArray::fieldindex(const std::string& key) const {
    throw std::invalid_argument(std::string("key ") + util::quote(key, true) + std::string(" does not exist (data are not records)"));
  }

  const std::string NumpyArray::key(int64_t fieldindex) const {
    throw std::invalid_argument(std::string("fieldindex \"") + std::to_string(fieldindex) + std::string("\" does not exist (data are not records)"));
  }

  bool NumpyArray::haskey(const std::string& key) const {
    return false;
  }

  const std::vector<std::string> NumpyArray::keys() const {
    return std::vector<std::string>();
  }

  const Index64 NumpyArray::count64() const {
    if (ndim() < 1) {
      throw std::invalid_argument(std::string("NumpyArray cannot be counted because it has ") + std::to_string(ndim()) + std::string(" dimensions"));
    }
    else if (ndim() == 1) {
      Index64 tocount(1);
      tocount.ptr().get()[0] = length();
      return tocount;
    }
    int64_t len = length();
    Index64 tocount(len);
    struct Error err = awkward_regulararray_count(
      tocount.ptr().get(),
      (int64_t)shape_[1],
      len);
    util::handle_error(err, classname(), identities_.get());
    return tocount;
  }

  const std::shared_ptr<Content> NumpyArray::count(int64_t axis) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    ssize_t offset = (ssize_t)toaxis;
    if (offset > ndim()) {
      throw std::invalid_argument(std::string("NumpyArray cannot be counted in axis ") + std::to_string(offset) + (" because it has ") + std::to_string(ndim()) + std::string(" dimensions"));
    }

#ifdef _MSC_VER
    std::string format = "q";
#else
    std::string format = "l";
#endif
    if (offset == 0) {
      Index64 tocount = count64();
      std::vector<ssize_t> shape({ (ssize_t)tocount.length() });
      std::vector<ssize_t> strides({ (ssize_t)sizeof(int64_t) });

      return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), tocount.ptr(), shape, strides, 0, sizeof(int64_t), format);
    }
    else if (offset + 1 == ndim()) {
      Index64 tocount(1);
      tocount.ptr().get()[0] = std::accumulate(std::begin(shape_), std::end(shape_), 1, std::multiplies<ssize_t>());
      std::vector<ssize_t> shape({ (ssize_t)tocount.length() });
      std::vector<ssize_t> strides({ (ssize_t)sizeof(int64_t) });

      return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), tocount.ptr(), shape, strides, 0, sizeof(int64_t), format);
    }
    else {
      // From studies/flatten.py:
      // # content = NumpyArray(self.ptr, self.shape[1:], self.strides[1:], self.offset).count(axis - 1)
      // # index = [0] * self.shape[0] * self.shape[1]
      // # return RegularArray(IndexedArray(index, content), self.shape[1])

      std::vector<ssize_t> nextshape = std::vector<ssize_t>(std::begin(shape_) + 1, std::end(shape_));
      std::vector<ssize_t> nextstrides = std::vector<ssize_t>(std::begin(strides_) + 1, std::end(strides_));

      std::shared_ptr<Content> content = (NumpyArray(identities_, parameters_, ptr_, nextshape, nextstrides, byteoffset_, itemsize_, format_)).count(offset - 1);

      int64_t len = shape_[0]*shape_[1];
      Index64 tocount(len);
      struct Error err = awkward_regulararray_count(
        tocount.ptr().get(),
        (int64_t)0,
        len);
      util::handle_error(err, classname(), identities_.get());

      std::shared_ptr<IndexedArray64> indexed = std::make_shared<IndexedArray64>(Identities::none(), util::Parameters(), tocount, content);

      return std::make_shared<RegularArray>(Identities::none(), util::Parameters(), indexed, shape_[1]);
    }
  }

  const std::vector<ssize_t> flatten_shape(const std::vector<ssize_t> shape) {
    if (shape.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      std::vector<ssize_t> out = { shape[0]*shape[1] };
      out.insert(out.end(), shape.begin() + 2, shape.end());
      return out;
    }
  }

  const std::vector<ssize_t> flatten_strides(const std::vector<ssize_t> strides) {
    if (strides.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      return std::vector<ssize_t>(strides.begin() + 1, strides.end());
    }
  }

  const std::vector<ssize_t> flatten_shape(const std::vector<ssize_t>& shape, int64_t axis) {
    if (shape.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      ssize_t offset = (ssize_t)axis;
      std::vector<ssize_t> out;
      const auto& indx = std::begin(shape) + offset;
      if (indx > std::begin(shape)) {
        out.insert(std::end(out), std::begin(shape), indx);
      }
      out.emplace_back(shape[offset]*shape[offset + 1]);
      out.insert(std::end(out), indx + 2, std::end(shape));
      return out;
    }
  }

  const std::vector<ssize_t> flatten_strides(const std::vector<ssize_t>& strides, int64_t axis) {
    if (strides.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      ssize_t offset = (ssize_t)axis;
      std::vector<ssize_t> out;
      const auto& indx = std::begin(strides) + offset;
      if (indx > std::begin(strides)) {
        out.insert(std::end(out), std::begin(strides), indx);
      }
      out.insert(std::end(out), indx + 1, std::end(strides));
      return out;
    }
  }

  const std::shared_ptr<Content> NumpyArray::flatten(int64_t axis) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (shape_.size() <= 1) {
      throw std::invalid_argument(std::string("NumpyArray cannot be flattened because it has ") + std::to_string(ndim()) + std::string(" dimensions"));
    }
    if (toaxis >= (int64_t)shape_.size() - 1) {
      throw std::invalid_argument(std::string("NumpyArray cannot be flattened because axis is ") + std::to_string(axis) + std::string(" exeeds its ") + std::to_string(ndim()) + std::string(" dimensions"));
    }
    if (iscontiguous()) {
      return std::make_shared<NumpyArray>(identities_, parameters_, ptr_, flatten_shape(shape_, toaxis), flatten_strides(strides_, toaxis), byteoffset_, itemsize_, format_);
    }
    else {
      return contiguous().flatten(toaxis);
    }
  }

  bool NumpyArray::mergeable(const std::shared_ptr<Content>& other, bool mergebool) const {
    if (!parameters_equal(other.get()->parameters())) {
      return false;
    }

    if (dynamic_cast<EmptyArray*>(other.get())  ||
        dynamic_cast<UnionArray8_32*>(other.get())  ||
        dynamic_cast<UnionArray8_U32*>(other.get())  ||
        dynamic_cast<UnionArray8_64*>(other.get())) {
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

    if (ndim() == 0) {
      return false;
    }

    if (NumpyArray* rawother = dynamic_cast<NumpyArray*>(other.get())) {
      if (ndim() != rawother->ndim()) {
        return false;
      }

      std::string other_format = rawother->format();

      if (!mergebool  &&  ((format_.compare("?") == 0  &&  other_format.compare("?") != 0)  ||  (format_.compare("?") != 0  &&  other_format.compare("?") == 0))) {
        return false;
      }

      if (!(format_.compare("d") == 0  ||  format_.compare("f") == 0  ||  format_.compare("q") == 0  ||  format_.compare("Q") == 0  ||  format_.compare("l") == 0  ||  format_.compare("L") == 0  ||  format_.compare("i") == 0  ||  format_.compare("I") == 0  ||  format_.compare("h") == 0  ||  format_.compare("H") == 0  ||  format_.compare("b") == 0  ||  format_.compare("B") == 0  ||  format_.compare("c") == 0  ||  format_.compare("?") == 0  ||
          other_format.compare("d") == 0  ||  other_format.compare("f") == 0  ||  other_format.compare("q") == 0  ||  other_format.compare("Q") == 0  ||  other_format.compare("l") == 0  ||  other_format.compare("L") == 0  ||  other_format.compare("i") == 0  ||  other_format.compare("I") == 0  ||  other_format.compare("h") == 0  ||  other_format.compare("H") == 0  ||  other_format.compare("b") == 0  ||  other_format.compare("B") == 0  ||  other_format.compare("c") == 0  ||  other_format.compare("?") == 0)) {
        return false;
      }

      std::vector<ssize_t> other_shape = rawother->shape();
      for (int64_t i = ((int64_t)shape_.size()) - 1;  i > 0;  i--) {
        if (shape_[(size_t)i] != other_shape[(size_t)i]) {
          return false;
        }
      }

      return true;
    }
    else {
      return false;
    }
  }

  const std::shared_ptr<Content> NumpyArray::merge(const std::shared_ptr<Content>& other) const {
    if (!parameters_equal(other.get()->parameters())) {
      return merge_as_union(other);
    }

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
    else if (UnionArray8_32* rawother = dynamic_cast<UnionArray8_32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnionArray8_U32* rawother = dynamic_cast<UnionArray8_U32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnionArray8_64* rawother = dynamic_cast<UnionArray8_64*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }

    if (ndim() == 0) {
      throw std::invalid_argument("cannot merge Numpy scalars");
    }

    NumpyArray contiguous_self = contiguous();
    if (NumpyArray* rawother = dynamic_cast<NumpyArray*>(other.get())) {
      if (ndim() != rawother->ndim()) {
        throw std::invalid_argument("cannot merge arrays with different shapes");
      }

      std::string other_format = rawother->format();

      ssize_t itemsize;
      std::string format;
      if (format_.compare("d") == 0  ||  format_.compare("f") == 0  ||  other_format.compare("d") == 0  ||  other_format.compare("f") == 0) {
        itemsize = 8;
        format = "d";
      }
#ifdef _MSC_VER
      else if (format_.compare("Q") == 0  &&  other_format.compare("Q") == 0) {
        itemsize = 8;
        format = "Q";
#else
      else if (format_.compare("L") == 0  &&  other_format.compare("L") == 0) {
        itemsize = 8;
        format = "L";
#endif
      }
      else if (format_.compare("q") == 0  ||  format_.compare("Q") == 0  ||  format_.compare("l") == 0  ||  format_.compare("L") == 0  ||  format_.compare("i") == 0  ||  format_.compare("I") == 0  ||  format_.compare("h") == 0  ||  format_.compare("H") == 0  ||  format_.compare("b") == 0  ||  format_.compare("B") == 0  ||  format_.compare("c") == 0  ||  other_format.compare("q") == 0  ||  other_format.compare("Q") == 0  ||  other_format.compare("l") == 0  ||  other_format.compare("L") == 0  ||  other_format.compare("i") == 0  ||  other_format.compare("I") == 0  ||  other_format.compare("h") == 0  ||  other_format.compare("H") == 0  ||  other_format.compare("b") == 0  ||  other_format.compare("B") == 0  ||  other_format.compare("c") == 0) {
        itemsize = 8;
#ifdef _MSC_VER
        format = "q";
#else
        format = "l";
#endif
      }
      else if (format_.compare("?") == 0  &&  other_format.compare("?") == 0) {
        itemsize = 1;
        format = "?";
      }
      else {
        throw std::invalid_argument(std::string("cannot merge Numpy format \"") + format_ + std::string("\" with \"") + other_format + std::string("\""));
      }

      std::vector<ssize_t> other_shape = rawother->shape();
      std::vector<ssize_t> shape;
      std::vector<ssize_t> strides;
      shape.push_back(shape_[0] + other_shape[0]);
      strides.push_back(itemsize);
      int64_t self_flatlength = shape_[0];
      int64_t other_flatlength = other_shape[0];
      for (int64_t i = ((int64_t)shape_.size()) - 1;  i > 0;  i--) {
        if (shape_[(size_t)i] != other_shape[(size_t)i]) {
          throw std::invalid_argument("cannot merge arrays with different shapes");
        }
        shape.insert(shape.begin() + 1, shape_[(size_t)i]);
        strides.insert(strides.begin(), strides[0]*shape_[(size_t)i]);
        self_flatlength *= (int64_t)shape_[(size_t)i];
        other_flatlength *= (int64_t)shape_[(size_t)i];
      }

      std::shared_ptr<void> ptr(new uint8_t[(size_t)(itemsize*(self_flatlength + other_flatlength))], util::array_deleter<uint8_t>());

      NumpyArray contiguous_other = rawother->contiguous();

      int64_t self_offset = contiguous_self.byteoffset() / contiguous_self.itemsize();
      int64_t other_offset = contiguous_other.byteoffset() / contiguous_other.itemsize();

      struct Error err;
      if (format.compare("d") == 0) {
        if (format_.compare("d") == 0) {
          err = awkward_numpyarray_fill_todouble_fromdouble(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<double*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("f") == 0) {
          err = awkward_numpyarray_fill_todouble_fromfloat(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<float*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
#ifdef _MSC_VER
        else if (format_.compare("q") == 0) {
#else
        else if (format_.compare("l") == 0) {
#endif
          err = awkward_numpyarray_fill_todouble_from64(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<int64_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
#ifdef _MSC_VER
          else if (format_.compare("Q") == 0) {
#else
          else if (format_.compare("L") == 0) {
#endif
          err = awkward_numpyarray_fill_todouble_fromU64(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<uint64_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
#ifdef _MSC_VER
          else if (format_.compare("l") == 0) {
#else
          else if (format_.compare("i") == 0) {
#endif
          err = awkward_numpyarray_fill_todouble_from32(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<int32_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
#ifdef _MSC_VER
          else if (format_.compare("L") == 0) {
#else
          else if (format_.compare("I") == 0) {
#endif
          err = awkward_numpyarray_fill_todouble_fromU32(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<uint32_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("h") == 0) {
          err = awkward_numpyarray_fill_todouble_from16(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<int16_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("H") == 0) {
          err = awkward_numpyarray_fill_todouble_fromU16(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<uint16_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("b") == 0) {
          err = awkward_numpyarray_fill_todouble_from8(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<int8_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
          err = awkward_numpyarray_fill_todouble_fromU8(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<uint8_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("?") == 0) {
          err = awkward_numpyarray_fill_todouble_frombool(reinterpret_cast<double*>(ptr.get()), 0, reinterpret_cast<bool*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else {
          throw std::invalid_argument(std::string("cannot merge Numpy format \"") + format_ + std::string("\" with \"") + other_format + std::string("\""));
        }
        util::handle_error(err, classname(), nullptr);

        if (other_format.compare("d") == 0) {
          err = awkward_numpyarray_fill_todouble_fromdouble(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<double*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("f") == 0) {
          err = awkward_numpyarray_fill_todouble_fromfloat(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<float*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
#ifdef _MSC_VER
        else if (other_format.compare("q") == 0) {
#else
        else if (other_format.compare("l") == 0) {
#endif
          err = awkward_numpyarray_fill_todouble_from64(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<int64_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
#ifdef _MSC_VER
          else if (other_format.compare("Q") == 0) {
#else
          else if (other_format.compare("L") == 0) {
#endif
          err = awkward_numpyarray_fill_todouble_fromU64(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<uint64_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
#ifdef _MSC_VER
          else if (other_format.compare("l") == 0) {
#else
          else if (other_format.compare("i") == 0) {
#endif
          err = awkward_numpyarray_fill_todouble_from32(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<int32_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
#ifdef _MSC_VER
          else if (other_format.compare("L") == 0) {
#else
          else if (other_format.compare("I") == 0) {
#endif
          err = awkward_numpyarray_fill_todouble_fromU32(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<uint32_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("h") == 0) {
          err = awkward_numpyarray_fill_todouble_from16(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<int16_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("H") == 0) {
          err = awkward_numpyarray_fill_todouble_fromU16(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<uint16_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("b") == 0) {
          err = awkward_numpyarray_fill_todouble_from8(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<int8_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("B") == 0  ||  format_.compare("c") == 0) {
          err = awkward_numpyarray_fill_todouble_fromU8(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<uint8_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("?") == 0) {
          err = awkward_numpyarray_fill_todouble_frombool(reinterpret_cast<double*>(ptr.get()), self_flatlength, reinterpret_cast<bool*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else {
          throw std::invalid_argument(std::string("cannot merge Numpy format \"") + format_ + std::string("\" with \"") + other_format + std::string("\""));
        }
        util::handle_error(err, classname(), nullptr);
      }

      else if (format.compare("Q") == 0  ||  format.compare("L") == 0) {
        err = awkward_numpyarray_fill_toU64_fromU64(reinterpret_cast<uint64_t*>(ptr.get()), 0, reinterpret_cast<uint64_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        util::handle_error(err, classname(), nullptr);
        err = awkward_numpyarray_fill_toU64_fromU64(reinterpret_cast<uint64_t*>(ptr.get()), self_flatlength, reinterpret_cast<uint64_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        util::handle_error(err, classname(), nullptr);
      }

      else if (itemsize == 8) {
#ifdef _MSC_VER
        if (format_.compare("q") == 0) {
#else
        if (format_.compare("l") == 0) {
#endif
          err = awkward_numpyarray_fill_to64_from64(reinterpret_cast<int64_t*>(ptr.get()), 0, reinterpret_cast<int64_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
#ifdef _MSC_VER
          else if (format_.compare("Q") == 0) {
#else
          else if (format_.compare("L") == 0) {
#endif
          err = awkward_numpyarray_fill_to64_fromU64(reinterpret_cast<int64_t*>(ptr.get()), 0, reinterpret_cast<uint64_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
#ifdef _MSC_VER
          else if (format_.compare("l") == 0) {
#else
          else if (format_.compare("i") == 0) {
#endif
          err = awkward_numpyarray_fill_to64_from32(reinterpret_cast<int64_t*>(ptr.get()), 0, reinterpret_cast<int32_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
#ifdef _MSC_VER
          else if (format_.compare("L") == 0) {
#else
          else if (format_.compare("I") == 0) {
#endif
          err = awkward_numpyarray_fill_to64_fromU32(reinterpret_cast<int64_t*>(ptr.get()), 0, reinterpret_cast<uint32_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("h") == 0) {
          err = awkward_numpyarray_fill_to64_from16(reinterpret_cast<int64_t*>(ptr.get()), 0, reinterpret_cast<int16_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("H") == 0) {
          err = awkward_numpyarray_fill_to64_fromU16(reinterpret_cast<int64_t*>(ptr.get()), 0, reinterpret_cast<uint16_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("b") == 0) {
          err = awkward_numpyarray_fill_to64_from8(reinterpret_cast<int64_t*>(ptr.get()), 0, reinterpret_cast<int8_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
          err = awkward_numpyarray_fill_to64_fromU8(reinterpret_cast<int64_t*>(ptr.get()), 0, reinterpret_cast<uint8_t*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else if (format_.compare("?") == 0) {
          err = awkward_numpyarray_fill_to64_frombool(reinterpret_cast<int64_t*>(ptr.get()), 0, reinterpret_cast<bool*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        }
        else {
          throw std::invalid_argument(std::string("cannot merge Numpy format \"") + format_ + std::string("\" with \"") + other_format + std::string("\""));
        }
        util::handle_error(err, classname(), nullptr);

#ifdef _MSC_VER
        if (other_format.compare("q") == 0) {
#else
        if (other_format.compare("l") == 0) {
#endif
          err = awkward_numpyarray_fill_to64_from64(reinterpret_cast<int64_t*>(ptr.get()), self_flatlength, reinterpret_cast<int64_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
#ifdef _MSC_VER
          else if (other_format.compare("Q") == 0) {
#else
          else if (other_format.compare("L") == 0) {
#endif
          err = awkward_numpyarray_fill_to64_fromU64(reinterpret_cast<int64_t*>(ptr.get()), self_flatlength, reinterpret_cast<uint64_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
#ifdef _MSC_VER
          else if (other_format.compare("l") == 0) {
#else
          else if (other_format.compare("i") == 0) {
#endif
          err = awkward_numpyarray_fill_to64_from32(reinterpret_cast<int64_t*>(ptr.get()), self_flatlength, reinterpret_cast<int32_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
#ifdef _MSC_VER
          else if (other_format.compare("L") == 0) {
#else
          else if (other_format.compare("I") == 0) {
#endif
          err = awkward_numpyarray_fill_to64_fromU32(reinterpret_cast<int64_t*>(ptr.get()), self_flatlength, reinterpret_cast<uint32_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("h") == 0) {
          err = awkward_numpyarray_fill_to64_from16(reinterpret_cast<int64_t*>(ptr.get()), self_flatlength, reinterpret_cast<int16_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("H") == 0) {
          err = awkward_numpyarray_fill_to64_fromU16(reinterpret_cast<int64_t*>(ptr.get()), self_flatlength, reinterpret_cast<uint16_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("b") == 0) {
          err = awkward_numpyarray_fill_to64_from8(reinterpret_cast<int64_t*>(ptr.get()), self_flatlength, reinterpret_cast<int8_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("B") == 0  ||  format_.compare("c") == 0) {
          err = awkward_numpyarray_fill_to64_fromU8(reinterpret_cast<int64_t*>(ptr.get()), self_flatlength, reinterpret_cast<uint8_t*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else if (other_format.compare("?") == 0) {
          err = awkward_numpyarray_fill_to64_frombool(reinterpret_cast<int64_t*>(ptr.get()), self_flatlength, reinterpret_cast<bool*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        }
        else {
          throw std::invalid_argument(std::string("cannot merge Numpy format \"") + format_ + std::string("\" with \"") + other_format + std::string("\""));
        }
        util::handle_error(err, classname(), nullptr);
      }

      else {
        err = awkward_numpyarray_fill_tobool_frombool(reinterpret_cast<bool*>(ptr.get()), 0, reinterpret_cast<bool*>(contiguous_self.ptr().get()), self_offset, self_flatlength);
        util::handle_error(err, classname(), nullptr);
        err = awkward_numpyarray_fill_tobool_frombool(reinterpret_cast<bool*>(ptr.get()), self_flatlength, reinterpret_cast<bool*>(contiguous_other.ptr().get()), other_offset, other_flatlength);
        util::handle_error(err, classname(), nullptr);
      }

      return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), ptr, shape, strides, 0, itemsize, format);
    }

    else {
      throw std::invalid_argument(std::string("cannot merge ") + classname() + std::string(" with ") + other.get()->classname());
    }
  }

  const std::shared_ptr<Content> NumpyArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("NumpyArray has its own getitem_next system");
  }

  const std::shared_ptr<Content> NumpyArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("NumpyArray has its own getitem_next system");
  }

  const std::shared_ptr<Content> NumpyArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("NumpyArray has its own getitem_next system");
  }

  const std::shared_ptr<Content> NumpyArray::getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("NumpyArray has its own getitem_next system");
  }

  const std::shared_ptr<Content> NumpyArray::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("NumpyArray has its own getitem_next system");
  }

  bool NumpyArray::iscontiguous() const {
    ssize_t x = itemsize_;
    for (ssize_t i = ndim() - 1;  i >= 0;  i--) {
      if (x != strides_[i]) return false;
      x *= shape_[i];
    }
    return true;  // true for isscalar(), too
  }

  void NumpyArray::become_contiguous() {
    if (!iscontiguous()) {
      NumpyArray x = contiguous();
      identities_ = x.identities_;
      ptr_ = x.ptr_;
      shape_ = x.shape_;
      strides_ = x.strides_;
      byteoffset_ = x.byteoffset_;
    }
  }

  const NumpyArray NumpyArray::contiguous() const {
    if (iscontiguous()) {
      return NumpyArray(identities_, parameters_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_);
    }
    else {
      Index64 bytepos(shape_[0]);
      struct Error err = awkward_numpyarray_contiguous_init_64(bytepos.ptr().get(), shape_[0], strides_[0]);
      util::handle_error(err, classname(), identities_.get());
      return contiguous_next(bytepos);
    }
  }

  const NumpyArray NumpyArray::contiguous_next(const Index64& bytepos) const {
    if (iscontiguous()) {
      std::shared_ptr<void> ptr(new uint8_t[(size_t)(bytepos.length()*strides_[0])], util::array_deleter<uint8_t>());
      struct Error err = awkward_numpyarray_contiguous_copy_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        bytepos.length(),
        strides_[0],
        byteoffset_,
        bytepos.ptr().get());
      util::handle_error(err, classname(), identities_.get());
      return NumpyArray(identities_, parameters_, ptr, shape_, strides_, 0, itemsize_, format_);
    }

    else if (shape_.size() == 1) {
      std::shared_ptr<void> ptr(new uint8_t[(size_t)(bytepos.length()*itemsize_)], util::array_deleter<uint8_t>());
      struct Error err = awkward_numpyarray_contiguous_copy_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        bytepos.length(),
        itemsize_,
        byteoffset_,
        bytepos.ptr().get());
      util::handle_error(err, classname(), identities_.get());
      std::vector<ssize_t> strides = { itemsize_ };
      return NumpyArray(identities_, parameters_, ptr, shape_, strides, 0, itemsize_, format_);
    }

    else {
      NumpyArray next(identities_, parameters_, ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);

      Index64 nextbytepos(bytepos.length()*shape_[1]);
      struct Error err = awkward_numpyarray_contiguous_next_64(
        nextbytepos.ptr().get(),
        bytepos.ptr().get(),
        bytepos.length(),
        (int64_t)shape_[1],
        (int64_t)strides_[1]);
      util::handle_error(err, classname(), identities_.get());

      NumpyArray out = next.contiguous_next(nextbytepos);
      std::vector<ssize_t> outstrides = { shape_[1]*out.strides_[0] };
      outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
      return NumpyArray(out.identities_, out.parameters_, out.ptr_, shape_, outstrides, out.byteoffset_, itemsize_, format_);
    }
  }

  const NumpyArray NumpyArray::getitem_bystrides(const std::shared_ptr<SliceItem>& head, const Slice& tail, int64_t length) const {
    if (head.get() == nullptr) {
      return NumpyArray(identities_, parameters_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_);
    }
    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      return getitem_bystrides(*at, tail, length);
    }
    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      return getitem_bystrides(*range, tail, length);
    }
    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_bystrides(*ellipsis, tail, length);
    }
    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_bystrides(*newaxis, tail, length);
    }
    else if (SliceField* field = dynamic_cast<SliceField*>(head.get())) {
      throw std::invalid_argument(field->tostring() + std::string(" is not a valid slice type for ") + classname());
    }
    else if (SliceFields* fields = dynamic_cast<SliceFields*>(head.get())) {
      throw std::invalid_argument(fields->tostring() + std::string(" is not a valid slice type for ") + classname());
    }
    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }

  const NumpyArray NumpyArray::getitem_bystrides(const SliceAt& at, const Slice& tail, int64_t length) const {
    if (ndim() < 2) {
      util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    int64_t i = at.at();
    if (i < 0) i += shape_[1];
    if (i < 0  ||  i >= shape_[1]) {
      util::handle_error(failure("index out of range", kSliceNone, at.at()), classname(), identities_.get());
    }

    ssize_t nextbyteoffset = byteoffset_ + ((ssize_t)i)*strides_[1];
    NumpyArray next(identities_, parameters_, ptr_, flatten_shape(shape_), flatten_strides(strides_), nextbyteoffset, itemsize_, format_);

    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = next.getitem_bystrides(nexthead, nexttail, length);

    std::vector<ssize_t> outshape = { (ssize_t)length };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    return NumpyArray(out.identities_, out.parameters_, out.ptr_, outshape, out.strides_, out.byteoffset_, itemsize_, format_);
  }

  const NumpyArray NumpyArray::getitem_bystrides(const SliceRange& range, const Slice& tail, int64_t length) const {
    if (ndim() < 2) {
      util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    int64_t start = range.start();
    int64_t stop = range.stop();
    int64_t step = range.step();
    if (step == Slice::none()) {
      step = 1;
    }
    awkward_regularize_rangeslice(&start, &stop, step > 0, range.hasstart(), range.hasstop(), (int64_t)shape_[1]);

    int64_t numer = std::abs(start - stop);
    int64_t denom = std::abs(step);
    int64_t d = numer / denom;
    int64_t m = numer % denom;
    int64_t lenhead = d + (m != 0 ? 1 : 0);

    ssize_t nextbyteoffset = byteoffset_ + ((ssize_t)start)*strides_[1];
    NumpyArray next(identities_, parameters_, ptr_, flatten_shape(shape_), flatten_strides(strides_), nextbyteoffset, itemsize_, format_);

    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = next.getitem_bystrides(nexthead, nexttail, length*lenhead);

    std::vector<ssize_t> outshape = { (ssize_t)length, (ssize_t)lenhead };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    std::vector<ssize_t> outstrides = { strides_[0], strides_[1]*((ssize_t)step) };
    outstrides.insert(outstrides.end(), out.strides_.begin() + 1, out.strides_.end());
    return NumpyArray(out.identities_, out.parameters_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
  }

  const NumpyArray NumpyArray::getitem_bystrides(const SliceEllipsis& ellipsis, const Slice& tail, int64_t length) const {
    std::pair<int64_t, int64_t> minmax = minmax_depth();
    assert(minmax.first == minmax.second);
    int64_t mindepth = minmax.first;

    if (tail.length() == 0  ||  mindepth - 1 == tail.dimlength()) {
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      return getitem_bystrides(nexthead, nexttail, length);
    }
    else {
      std::vector<std::shared_ptr<SliceItem>> tailitems = tail.items();
      std::vector<std::shared_ptr<SliceItem>> items = { std::make_shared<SliceEllipsis>() };
      items.insert(items.end(), tailitems.begin(), tailitems.end());

      std::shared_ptr<SliceItem> nexthead = std::make_shared<SliceRange>(Slice::none(), Slice::none(), 1);
      Slice nexttail(items);
      return getitem_bystrides(nexthead, nexttail, length);
    }
  }

  const NumpyArray NumpyArray::getitem_bystrides(const SliceNewAxis& newaxis, const Slice& tail, int64_t length) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = getitem_bystrides(nexthead, nexttail, length);

    std::vector<ssize_t> outshape = { (ssize_t)length, 1 };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    std::vector<ssize_t> outstrides = { out.strides_[0] };
    outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
    return NumpyArray(out.identities_, out.parameters_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
  }

  const NumpyArray NumpyArray::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const {
    if (head.get() == nullptr) {
      std::shared_ptr<void> ptr(new uint8_t[(size_t)(carry.length()*stride)], util::array_deleter<uint8_t>());
      struct Error err = awkward_numpyarray_getitem_next_null_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        carry.length(),
        stride,
        byteoffset_,
        carry.ptr().get());
      util::handle_error(err, classname(), identities_.get());

      std::shared_ptr<Identities> identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_carry_64(carry);
      }

      std::vector<ssize_t> shape = { (ssize_t)carry.length() };
      shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
      std::vector<ssize_t> strides = { (ssize_t)stride };
      strides.insert(strides.end(), strides_.begin() + 1, strides_.end());
      return NumpyArray(identities, parameters_, ptr, shape, strides, 0, itemsize_, format_);
    }

    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      return getitem_next(*at, tail, carry, advanced, length, stride, first);
    }
    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      return getitem_next(*range, tail, carry, advanced, length, stride, first);
    }
    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_next(*ellipsis, tail, carry, advanced, length, stride, first);
    }
    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_next(*newaxis, tail, carry, advanced, length, stride, first);
    }
    else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
      return getitem_next(*array, tail, carry, advanced, length, stride, first);
    }
    else if (SliceField* field = dynamic_cast<SliceField*>(head.get())) {
      throw std::invalid_argument(field->tostring() + std::string(" is not a valid slice type for ") + classname());
    }
    else if (SliceFields* fields = dynamic_cast<SliceFields*>(head.get())) {
      throw std::invalid_argument(fields->tostring() + std::string(" is not a valid slice type for ") + classname());
    }
    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }

  const NumpyArray NumpyArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const {
    if (ndim() < 2) {
      util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    NumpyArray next(first ? identities_ : Identities::none(), parameters_, ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();

    // if we had any array slices, this int would become an array
    assert(advanced.length() == 0);

    int64_t regular_at = at.at();
    if (regular_at < 0) {
      regular_at += shape_[1];
    }
    if (!(0 <= regular_at  &&  regular_at < shape_[1])) {
      util::handle_error(failure("index out of range", kSliceNone, at.at()), classname(), identities_.get());
    }

    Index64 nextcarry(carry.length());
    struct Error err = awkward_numpyarray_getitem_next_at_64(
      nextcarry.ptr().get(),
      carry.ptr().get(),
      carry.length(),
      shape_[1],   // because this is contiguous
      regular_at);
    util::handle_error(err, classname(), identities_.get());

    NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, advanced, length, next.strides_[0], false);

    std::vector<ssize_t> outshape = { (ssize_t)length };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    return NumpyArray(out.identities_, out.parameters_, out.ptr_, outshape, out.strides_, out.byteoffset_, itemsize_, format_);
  }

  const NumpyArray NumpyArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const {
    if (ndim() < 2) {
      util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    int64_t start = range.start();
    int64_t stop = range.stop();
    int64_t step = range.step();
    if (step == Slice::none()) {
      step = 1;
    }
    awkward_regularize_rangeslice(&start, &stop, step > 0, range.hasstart(), range.hasstop(), (int64_t)shape_[1]);

    int64_t numer = std::abs(start - stop);
    int64_t denom = std::abs(step);
    int64_t d = numer / denom;
    int64_t m = numer % denom;
    int64_t lenhead = d + (m != 0 ? 1 : 0);

    NumpyArray next(first ? identities_ : Identities::none(), parameters_, ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();

    if (advanced.length() == 0) {
      Index64 nextcarry(carry.length()*lenhead);
      struct Error err = awkward_numpyarray_getitem_next_range_64(
        nextcarry.ptr().get(),
        carry.ptr().get(),
        carry.length(),
        lenhead,
        shape_[1],   // because this is contiguous
        start,
        step);
      util::handle_error(err, classname(), identities_.get());

      NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, advanced, length*lenhead, next.strides_[0], false);
      std::vector<ssize_t> outshape = { (ssize_t)length, (ssize_t)lenhead };
      outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides = { (ssize_t)lenhead*out.strides_[0] };
      outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
      return NumpyArray(out.identities_, out.parameters_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
    }

    else {
      Index64 nextcarry(carry.length()*lenhead);
      Index64 nextadvanced(carry.length()*lenhead);
      struct Error err = awkward_numpyarray_getitem_next_range_advanced_64(
        nextcarry.ptr().get(),
        nextadvanced.ptr().get(),
        carry.ptr().get(),
        advanced.ptr().get(),
        carry.length(),
        lenhead,
        shape_[1],   // because this is contiguous
        start,
        step);
      util::handle_error(err, classname(), identities_.get());

      NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length*lenhead, next.strides_[0], false);
      std::vector<ssize_t> outshape = { (ssize_t)length, (ssize_t)lenhead };
      outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides = { (ssize_t)lenhead*out.strides_[0] };
      outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
      return NumpyArray(out.identities_, out.parameters_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
    }
  }

  const NumpyArray NumpyArray::getitem_next(const SliceEllipsis& ellipsis, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const {
    std::pair<int64_t, int64_t> minmax = minmax_depth();
    assert(minmax.first == minmax.second);
    int64_t mindepth = minmax.first;

    if (tail.length() == 0  ||  mindepth - 1 == tail.dimlength()) {
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      return getitem_next(nexthead, nexttail, carry, advanced, length, stride, false);
    }
    else {
      std::vector<std::shared_ptr<SliceItem>> tailitems = tail.items();
      std::vector<std::shared_ptr<SliceItem>> items = { std::make_shared<SliceEllipsis>() };
      items.insert(items.end(), tailitems.begin(), tailitems.end());
      std::shared_ptr<SliceItem> nexthead = std::make_shared<SliceRange>(Slice::none(), Slice::none(), 1);
      Slice nexttail(items);
      return getitem_next(nexthead, nexttail, carry, advanced, length, stride, false);
    }
  }

  const NumpyArray NumpyArray::getitem_next(const SliceNewAxis& newaxis, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = getitem_next(nexthead, nexttail, carry, advanced, length, stride, false);

    std::vector<ssize_t> outshape = { (ssize_t)length, 1 };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    std::vector<ssize_t> outstrides = { out.strides_[0] };
    outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
    return NumpyArray(out.identities_, out.parameters_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
  }

  const NumpyArray NumpyArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const {
    if (ndim() < 2) {
      util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    NumpyArray next(first ? identities_ : Identities::none(), parameters_, ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();

    Index64 flathead = array.ravel();
    struct Error err = awkward_regularize_arrayslice_64(
      flathead.ptr().get(),
      flathead.length(),
      shape_[1]);
    util::handle_error(err, classname(), identities_.get());

    if (advanced.length() == 0) {
      Index64 nextcarry(carry.length()*flathead.length());
      Index64 nextadvanced(carry.length()*flathead.length());
      struct Error err = awkward_numpyarray_getitem_next_array_64(
        nextcarry.ptr().get(),
        nextadvanced.ptr().get(),
        carry.ptr().get(),
        flathead.ptr().get(),
        carry.length(),
        flathead.length(),
        shape_[1]);   // because this is contiguous
      util::handle_error(err, classname(), identities_.get());

      NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length*flathead.length(), next.strides_[0], false);

      std::vector<ssize_t> outshape = { (ssize_t)length };
      std::vector<int64_t> arrayshape = array.shape();
      for (auto x = arrayshape.begin();  x != arrayshape.end();  ++x) {
        outshape.push_back((ssize_t)(*x));
      }
      outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());

      std::vector<ssize_t> outstrides(out.strides_.begin(), out.strides_.end());
      for (auto x = arrayshape.rbegin();  x != arrayshape.rend();  ++x) {
        outstrides.insert(outstrides.begin(), ((ssize_t)(*x))*outstrides[0]);
      }
      return NumpyArray(arrayshape.size() == 1 ? out.identities_ : Identities::none(), out.parameters_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
    }

    else {
      Index64 nextcarry(carry.length());
      struct Error err = awkward_numpyarray_getitem_next_array_advanced_64(
        nextcarry.ptr().get(),
        carry.ptr().get(),
        advanced.ptr().get(),
        flathead.ptr().get(),
        carry.length(),
        shape_[1]);   // because this is contiguous
      util::handle_error(err, classname(), identities_.get());

      NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, advanced, length*array.length(), next.strides_[0], false);

      std::vector<ssize_t> outshape = { (ssize_t)length };
      outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
      return NumpyArray(out.identities_, out.parameters_, out.ptr_, outshape, out.strides_, out.byteoffset_, itemsize_, format_);
    }
  }

  void NumpyArray::tojson_boolean(ToJson& builder) const {
    if (ndim() == 0) {
      bool* array = reinterpret_cast<bool*>(byteptr());
      builder.boolean(array[0]);
    }
    else if (ndim() == 1) {
      bool* array = reinterpret_cast<bool*>(byteptr());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        builder.boolean(array[i]);
      }
      builder.endlist();
    }
    else {
      const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
      const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)i);
        NumpyArray numpy(Identities::none(), util::Parameters(), ptr_, shape, strides, byteoffset, itemsize_, format_);
        numpy.tojson_boolean(builder);
      }
      builder.endlist();
    }
  }

  template <typename T>
  void NumpyArray::tojson_integer(ToJson& builder) const {
    if (ndim() == 0) {
      T* array = reinterpret_cast<T*>(byteptr());
      builder.integer(array[0]);
    }
    else if (ndim() == 1) {
      T* array = reinterpret_cast<T*>(byteptr());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        builder.integer(array[i]);
      }
      builder.endlist();
    }
    else {
      const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
      const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)i);
        NumpyArray numpy(Identities::none(), util::Parameters(), ptr_, shape, strides, byteoffset, itemsize_, format_);
        numpy.tojson_integer<T>(builder);
      }
      builder.endlist();
    }
  }

  template <typename T>
  void NumpyArray::tojson_real(ToJson& builder) const {
    if (ndim() == 0) {
      T* array = reinterpret_cast<T*>(byteptr());
      builder.real(array[0]);
    }
    else if (ndim() == 1) {
      T* array = reinterpret_cast<T*>(byteptr());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        builder.real(array[i]);
      }
      builder.endlist();
    }
    else {
      const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
      const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)i);
        NumpyArray numpy(Identities::none(), util::Parameters(), ptr_, shape, strides, byteoffset, itemsize_, format_);
        numpy.tojson_real<T>(builder);
      }
      builder.endlist();
    }
  }

  void NumpyArray::tojson_string(ToJson& builder) const {
    if (ndim() == 0) {
      char* array = reinterpret_cast<char*>(byteptr());
      builder.string(array, 1);
    }
    else if (ndim() == 1) {
      char* array = reinterpret_cast<char*>(byteptr());
      builder.string(array, length());
    }
    else {
      const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
      const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)i);
        NumpyArray numpy(Identities::none(), util::Parameters(), ptr_, shape, strides, byteoffset, itemsize_, format_);
        numpy.tojson_string(builder);
      }
      builder.endlist();
    }
  }
}
