// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/cpu-kernels/reducers.h"
#include "awkward/cpu-kernels/sorting.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/type/RegularType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/VirtualArray.h"
#include "awkward/util.h"

#include "awkward/array/NumpyArray.h"

namespace awkward {
  ////////// NumpyForm

  NumpyForm::NumpyForm(bool has_identities,
                       const util::Parameters& parameters,
                       const std::vector<int64_t>& inner_shape,
                       int64_t itemsize,
                       const std::string& format)
      : Form(has_identities, parameters)
      , inner_shape_(inner_shape)
      , itemsize_(itemsize)
      , format_(format) { }

  const std::vector<int64_t>
  NumpyForm::inner_shape() const {
    return inner_shape_;
  }

  int64_t
  NumpyForm::itemsize() const {
    return itemsize_;
  }

  const std::string
  NumpyForm::format() const {
    return format_;
  }

  const std::string
  NumpyForm::primitive() const {
    if (format_.compare("d") == 0) {
      return "float64";
    }
    else if (format_.compare("f") == 0) {
      return "float32";
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("q") == 0) {
#else
    else if (format_.compare("l") == 0) {
#endif
      return "int64";
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("Q") == 0) {
#else
    else if (format_.compare("L") == 0) {
#endif
      return "uint64";
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("l") == 0) {
#else
    else if (format_.compare("i") == 0) {
#endif
      return "int32";
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("L") == 0) {
#else
    else if (format_.compare("I") == 0) {
#endif
      return "uint32";
    }
    else if (format_.compare("h") == 0) {
      return "int16";
    }
    else if (format_.compare("H") == 0) {
      return "uint16";
    }
    else if (format_.compare("b") == 0) {
      return "int8";
    }
    else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
      return "uint8";
    }
    else if (format_.compare("?") == 0) {
      return "bool";
    }
    else {
      return "unknown";
    }
  }

  const TypePtr
  NumpyForm::type(const util::TypeStrs& typestrs) const {
    TypePtr out;
    if (format_.compare("d") == 0) {
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::float64);
    }
    else if (format_.compare("f") == 0) {
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::float32);
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("q") == 0) {
#else
    else if (format_.compare("l") == 0) {
#endif
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::int64);
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("Q") == 0) {
#else
    else if (format_.compare("L") == 0) {
#endif
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::uint64);
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("l") == 0) {
#else
    else if (format_.compare("i") == 0) {
#endif
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::int32);
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("L") == 0) {
#else
    else if (format_.compare("I") == 0) {
#endif
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::uint32);
    }
    else if (format_.compare("h") == 0) {
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::int16);
    }
    else if (format_.compare("H") == 0) {
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::uint16);
    }
    else if (format_.compare("b") == 0) {
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::int8);
    }
    else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::uint8);
    }
    else if (format_.compare("?") == 0) {
      out = std::make_shared<PrimitiveType>(
                parameters_,
                util::gettypestr(parameters_, typestrs),
                PrimitiveType::boolean);
    }
    else {
      throw std::invalid_argument(
        std::string("Numpy format \"") + format_
        + std::string("\" cannot be expressed as a PrimitiveType"));
    }
    for (int64_t i = ((int64_t)inner_shape_.size()) - 1;  i >= 0;  i--) {
      out = std::make_shared<RegularType>(
                util::Parameters(),
                util::gettypestr(parameters_, typestrs),
                out,
                inner_shape_[(size_t)i]);
    }
    return out;
  }

  const std::string
  NumpyForm::tostring() const {
    ToJsonPrettyString builder(-1);
    tojson_part(builder, false, true);
    return builder.tostring();
  }

  const std::string
  NumpyForm::tojson(bool pretty, bool verbose) const {
    if (pretty) {
      ToJsonPrettyString builder(-1);
      tojson_part(builder, verbose, true);
      return builder.tostring();
    }
    else {
      ToJsonString builder(-1);
      tojson_part(builder, verbose, true);
      return builder.tostring();
    }
  }

  void
  NumpyForm::tojson_part(ToJson& builder, bool verbose) const {
    return tojson_part(builder, verbose, false);
  }

  void
  NumpyForm::tojson_part(ToJson& builder, bool verbose, bool toplevel) const {
    std::string p = primitive();
    if (verbose  ||  toplevel  ||  p.empty()  ||  !inner_shape_.empty()) {
      builder.beginrecord();
      builder.field("class");
      builder.string("NumpyArray");
      if (verbose  ||  !inner_shape_.empty()) {
        builder.field("inner_shape");
        builder.beginlist();
        for (auto x : inner_shape_) {
          builder.integer(x);
        }
        builder.endlist();
      }
      builder.field("itemsize");
      builder.integer(itemsize_);
      builder.field("format");
      builder.string(format_);
      if (!p.empty()) {
        builder.field("primitive");
        builder.string(p);
      }
      else if (verbose) {
        builder.field("primitive");
        builder.null();
      }
      identities_tojson(builder, verbose);
      parameters_tojson(builder, verbose);
      builder.endrecord();
    }
    else {
      builder.string(p.c_str(), p.length());
    }
  }

  const FormPtr
  NumpyForm::shallow_copy() const {
    return std::make_shared<NumpyForm>(has_identities_,
                                       parameters_,
                                       inner_shape_,
                                       itemsize_,
                                       format_);
  }

  const std::string
  NumpyForm::purelist_parameter(const std::string& key) const {
    return parameter(key);
  }

  bool
  NumpyForm::purelist_isregular() const {
    return true;
  }

  int64_t
  NumpyForm::purelist_depth() const {
    return (int64_t)inner_shape_.size() + 1;
  }

  const std::pair<int64_t, int64_t>
  NumpyForm::minmax_depth() const {
    return std::pair<int64_t, int64_t>((int64_t)inner_shape_.size() + 1,
                                       (int64_t)inner_shape_.size() + 1);
  }

  const std::pair<bool, int64_t>
  NumpyForm::branch_depth() const {
    return std::pair<bool, int64_t>(false, (int64_t)inner_shape_.size() + 1);
  }

  int64_t
  NumpyForm::numfields() const {
    return -1;
  }

  int64_t
  NumpyForm::fieldindex(const std::string& key) const {
    throw std::invalid_argument(
      std::string("key ") + util::quote(key, true)
      + std::string(" does not exist (data are not records)"));
  }

  const std::string
  NumpyForm::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("fieldindex \"") + std::to_string(fieldindex)
      + std::string("\" does not exist (data are not records)"));
  }

  bool
  NumpyForm::haskey(const std::string& key) const {
    return false;
  }

  const std::vector<std::string>
  NumpyForm::keys() const {
    return std::vector<std::string>();
  }

  bool
  NumpyForm::equal(const FormPtr& other,
                   bool check_identities,
                   bool check_parameters,
                   bool compatibility_check) const {
    if (check_identities  &&
        has_identities_ != other.get()->has_identities()) {
      return false;
    }
    if (check_parameters  &&
        !util::parameters_equal(parameters_, other.get()->parameters())) {
      return false;
    }
    if (NumpyForm* t = dynamic_cast<NumpyForm*>(other.get())) {
      return (inner_shape_ == t->inner_shape()  &&
              itemsize_ == t->itemsize()  &&
              format_ == t->format());
    }
    else {
      return false;
    }
  }

  ////////// NumpyArray

  const std::unordered_map<std::type_index, std::string>
  NumpyArray::format_map = {
    { typeid(int8_t), "b"},
    { typeid(uint8_t), "B"},
#if defined _MSC_VER || defined __i386__
    { typeid(int32_t), "l"},
    { typeid(uint32_t), "L"},
    { typeid(int64_t), "q"}
#else
    { typeid(int32_t), "i"},
    { typeid(uint32_t), "I"},
    { typeid(int64_t), "l"}
#endif
 };

  NumpyArray::NumpyArray(const IdentitiesPtr& identities,
                         const util::Parameters& parameters,
                         const std::shared_ptr<void>& ptr,
                         const std::vector<ssize_t>& shape,
                         const std::vector<ssize_t>& strides,
                         ssize_t byteoffset,
                         ssize_t itemsize,
                         const std::string format,
                         const kernel::Lib ptr_lib)
      : Content(identities, parameters)
      , ptr_lib_(ptr_lib)
      , ptr_(ptr)
      , shape_(shape)
      , strides_(strides)
      , byteoffset_(byteoffset)
      , itemsize_(itemsize)
      , format_(format) {
    if (shape.size() != strides.size()) {
      throw std::invalid_argument(
        std::string("len(shape), which is ") + std::to_string(shape.size())
        + std::string(", must be equal to len(strides), which is ")
        + std::to_string(strides.size()));
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
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(int8_t) }),
                 0,
                 sizeof(int8_t),
                 format,
                 index.ptr_lib()) { }

  NumpyArray::NumpyArray(const IndexU8 index, const std::string& format)
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(uint8_t) }),
                 0,
                 sizeof(uint8_t),
                 format,
                 index.ptr_lib()) { }

  NumpyArray::NumpyArray(const Index32 index, const std::string& format)
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(int32_t) }),
                 0,
                 sizeof(int32_t),
                 format,
                 index.ptr_lib()) { }

  NumpyArray::NumpyArray(const IndexU32 index, const std::string& format)
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(uint32_t) }),
                 0,
                 sizeof(uint32_t),
                 format,
                 index.ptr_lib()) { }

  NumpyArray::NumpyArray(const Index64 index, const std::string& format)
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(int64_t) }),
                 0,
                 sizeof(int64_t),
                 format,
                 index.ptr_lib()) { }

  const std::shared_ptr<void>
  NumpyArray::ptr() const {
    return ptr_;
  }

  const std::vector<ssize_t>
  NumpyArray::shape() const {
    return shape_;
  }

  const std::vector<ssize_t>
  NumpyArray::strides() const {
    return strides_;
  }

  ssize_t
  NumpyArray::byteoffset() const {
    return byteoffset_;
  }

  ssize_t
  NumpyArray::itemsize() const {
    return itemsize_;
  }

  const std::string
  NumpyArray::format() const {
    return format_;
  }

  ssize_t
  NumpyArray::ndim() const {
    return (ssize_t)shape_.size();
  }

  kernel::Lib
  NumpyArray::ptr_lib() const {
    return ptr_lib_;
  }

  bool
  NumpyArray::isempty() const {
    for (auto x : shape_) {
      if (x == 0) {
        return true;
      }
    }
    return false;  // false for isscalar(), too
  }

  void*
  NumpyArray::byteptr() const {
    return reinterpret_cast<void*>(
      reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_);
  }

  void*
  NumpyArray::byteptr(ssize_t at) const {
    return reinterpret_cast<void*>(
      reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_ + at);
  }

  ssize_t
  NumpyArray::bytelength() const {
    if (isscalar()) {
      return itemsize_;
    }
    else {
      return shape_[0]*strides_[0];
    }
  }

  uint8_t
  NumpyArray::getbyte(ssize_t at) const {
    return *reinterpret_cast<uint8_t*>(byteptr(at));
  }

  int8_t
  NumpyArray::getint8(ssize_t at) const  {
    return *reinterpret_cast<int8_t*>(byteptr(at));
  }

  uint8_t
  NumpyArray::getuint8(ssize_t at) const {
    return *reinterpret_cast<uint8_t*>(byteptr(at));
  }

  int16_t
  NumpyArray::getint16(ssize_t at) const {
    return *reinterpret_cast<int16_t*>(byteptr(at));
  }

  uint16_t
  NumpyArray::getuint16(ssize_t at) const {
    return *reinterpret_cast<uint16_t*>(byteptr(at));
  }

  int32_t
  NumpyArray::getint32(ssize_t at) const {
    return *reinterpret_cast<int32_t*>(byteptr(at));
  }

  uint32_t
  NumpyArray::getuint32(ssize_t at) const {
    return *reinterpret_cast<uint32_t*>(byteptr(at));
  }

  int64_t
  NumpyArray::getint64(ssize_t at) const {
    return *reinterpret_cast<int64_t*>(byteptr(at));
  }

  uint64_t
  NumpyArray::getuint64(ssize_t at) const {
    return *reinterpret_cast<uint64_t*>(byteptr(at));
  }

  float_t
  NumpyArray::getfloat(ssize_t at) const {
    return *reinterpret_cast<float*>(byteptr(at));
  }

  double_t
  NumpyArray::getdouble(ssize_t at) const {
    return *reinterpret_cast<double*>(byteptr(at));
  }

  const ContentPtr
  NumpyArray::toRegularArray() const {
    if (isscalar()) {
      return shallow_copy();
    }
    NumpyArray contiguous_self = contiguous();
    std::vector<ssize_t> flatshape({ 1 });
    for (auto x : shape_) {
      flatshape[0] = flatshape[0] * x;
    }
    std::vector<ssize_t> flatstrides({ itemsize_ });
    ContentPtr out = std::make_shared<NumpyArray>(
      identities_,
      parameters_,
      contiguous_self.ptr(),
      flatshape,
      flatstrides,
      contiguous_self.byteoffset(),
      contiguous_self.itemsize(),
      contiguous_self.format());
    for (int64_t i = (int64_t)shape_.size() - 1;  i > 0;  i--) {
      out = std::make_shared<RegularArray>(Identities::none(),
                                           util::Parameters(),
                                           out,
                                           shape_[(size_t)i]);
    }
    return out;
  }

  bool
  NumpyArray::isscalar() const {
    return ndim() == 0;
  }

  const std::string
  NumpyArray::classname() const {
    return "NumpyArray";
  }

  void
  NumpyArray::setidentities(const IdentitiesPtr& identities) {
    if (identities.get() != nullptr  &&
        length() != identities.get()->length()) {
      util::handle_error(
        failure("content and its identities must have the same length",
                kSliceNone,
                kSliceNone),
        classname(),
        identities_.get());
    }
    identities_ = identities;
  }

  void
  NumpyArray::setidentities() {
    if (length() <= kMaxInt32) {
      IdentitiesPtr newidentities =
        std::make_shared<Identities32>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1,
                                       length());
      Identities32* rawidentities =
        reinterpret_cast<Identities32*>(newidentities.get());
      struct Error err = kernel::new_Identities<int32_t>(rawidentities->ptr().get(),
                                                         length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
    else {
      IdentitiesPtr newidentities =
        std::make_shared<Identities64>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1,
                                       length());
      Identities64* rawidentities =
        reinterpret_cast<Identities64*>(newidentities.get());
      struct Error err = kernel::new_Identities<int64_t>(rawidentities->ptr().get(),
                                                         length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  template <typename T>
  void tostring_as(kernel::Lib ptr_lib, std::stringstream& out, T* ptr, int64_t length) {
    if (length <= 10) {
      for (int64_t i = 0;  i < length;  i++) {
        if (i != 0) {
          out << " ";
        }
        if (std::is_same<T, bool>::value) {
          out << (kernel::NumpyArray_getitem_at(ptr_lib, ptr, i) ? "true" : "false");
        }
        else {
          out << kernel::NumpyArray_getitem_at(ptr_lib, ptr, i);
        }
      }
    }
    else {
      for (int64_t i = 0;  i < 5;  i++) {
        if (i != 0) {
          out << " ";
        }
        if (std::is_same<T, bool>::value) {
          out << (kernel::NumpyArray_getitem_at(ptr_lib, ptr, i) ? "true" : "false");
        }
        else {
          out << kernel::NumpyArray_getitem_at(ptr_lib, ptr, i);
        }
      }
      out << " ... ";
      for (int64_t i = length - 5;  i < length;  i++) {
        if (i != length - 5) {
          out << " ";
        }
        if (std::is_same<T, bool>::value) {
          out << (kernel::NumpyArray_getitem_at(ptr_lib, ptr, i) ? "true" : "false");
        }
        else {
          out << kernel::NumpyArray_getitem_at(ptr_lib, ptr, i);
        }
      }
    }
  }

  const TypePtr
  NumpyArray::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  const FormPtr
  NumpyArray::form(bool materialize) const {
    std::vector<int64_t> inner_shape;
    for (size_t i = 1;  i < shape_.size();  i++) {
      inner_shape.push_back((int64_t)shape_[i]);
    }
    return std::make_shared<NumpyForm>(identities_.get() != nullptr,
                                       parameters_,
                                       inner_shape,
                                       (int64_t)itemsize_,
                                       format_);
  }

  bool
  NumpyArray::has_virtual_form() const {
    return false;
  }

  bool
  NumpyArray::has_virtual_length() const {
    return false;
  }

  const std::string
  NumpyArray::tostring_part(const std::string& indent,
                            const std::string& pre,
                            const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " format="
        << util::quote(format_, true) << " shape=\"";
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
#if defined _MSC_VER || defined __i386__
    if (ndim() == 1  &&  format_.compare("l") == 0) {
#else
    if (ndim() == 1  &&  format_.compare("i") == 0) {
#endif
      tostring_as<int32_t>(ptr_lib(),
                           out,
                           reinterpret_cast<int32_t*>(byteptr()),
                           length());
    }
#if defined _MSC_VER || defined __i386__
    else if (ndim() == 1  &&  format_.compare("q") == 0) {
#else
    else if (ndim() == 1  &&  format_.compare("l") == 0) {
#endif
      tostring_as<int64_t>(ptr_lib(),
                           out,
                           reinterpret_cast<int64_t*>(byteptr()),
                           length());
    }
    else if (ndim() == 1  &&  format_.compare("f") == 0) {
      tostring_as<float>(ptr_lib(),
                         out,
                         reinterpret_cast<float*>(byteptr()),
                         length());
    }
    else if (ndim() == 1  &&  format_.compare("d") == 0) {
      tostring_as<double>(ptr_lib(),
                          out,
                          reinterpret_cast<double*>(byteptr()),
                          length());
    }
    else if (ndim() == 1  &&  format_.compare("?") == 0) {
      tostring_as<bool>(ptr_lib(),
                        out,
                        reinterpret_cast<bool*>(byteptr()),
                        length());
    }
    else {
      out << "0x ";
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
    }
    out << "\" at=\"0x";
    out << std::hex << std::setw(12) << std::setfill('0')
        << reinterpret_cast<ssize_t>(ptr_.get());
    if(ptr_lib() == kernel::Lib::cuda_kernels) {
      out << "\">\n";
      out << kernellib_asstring(indent + std::string("    "), "", "\n");

      if (identities_.get() != nullptr) {
        out << identities_.get()->tostring_part(
          indent + std::string("    "), "", "\n");
      }
      if (!parameters_.empty()) {
        out << parameters_tostring(indent + std::string("    "), "", "\n");
      }
        out << indent << "</" << classname() << ">" << post;
    }
    else {
      if (identities_.get() == nullptr  &&  parameters_.empty()) {
        out << "\"/>" << post;
      }
      else {
        out << "\">\n";
        if (identities_.get() != nullptr) {
          out << identities_.get()->tostring_part(
            indent + std::string("    "), "", "\n");
        }
        if (!parameters_.empty()) {
          out << parameters_tostring(indent + std::string("    "), "", "\n");
        }
        out << indent << "</" << classname() << ">" << post;
      }
    }
    return out.str();
  }

  const std::string
  NumpyArray::kernellib_asstring(const std::string &indent,
                                 const std::string &pre,
                                 const std::string &post) const {
    if(ptr_lib_ == kernel::Lib::cpu_kernels) {
      return "";
    }
    else {
      std::stringstream out;
      out << indent << pre << "<Lib name=\"";
      if(ptr_lib_ == kernel::Lib::cuda_kernels) {
        out << "cuda\" " << "device_number=\"" << kernel::get_ptr_device_num(ptr_lib(), ptr_.get())
        << "\" device_name=\"" << kernel::get_ptr_device_name(ptr_lib(), ptr_.get()) << "\"";
      }
      out << "/>" << post;
      return out.str();
    }
  }

  void
  NumpyArray::tojson_part(ToJson& builder,
                          bool include_beginendlist) const {
    check_for_iteration();
    if (parameter_equals("__array__", "\"byte\"")) {
      tojson_string(builder, include_beginendlist);
    }
    else if (parameter_equals("__array__", "\"char\"")) {
      tojson_string(builder, include_beginendlist);
    }
    else if (format_.compare("d") == 0) {
      tojson_real<double>(builder, include_beginendlist);
    }
    else if (format_.compare("f") == 0) {
      tojson_real<float>(builder, include_beginendlist);
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("q") == 0) {
#else
    else if (format_.compare("l") == 0) {
#endif
      tojson_integer<int64_t>(builder, include_beginendlist);
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("Q") == 0) {
#else
    else if (format_.compare("L") == 0) {
#endif
      tojson_integer<uint64_t>(builder, include_beginendlist);
    }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("l") == 0) {
#else
      else if (format_.compare("i") == 0) {
#endif
      tojson_integer<int32_t>(builder, include_beginendlist);
    }
#if defined _MSC_VER || defined __i386__
    else if (format_.compare("L") == 0) {
#else
    else if (format_.compare("I") == 0) {
#endif
      tojson_integer<uint32_t>(builder, include_beginendlist);
    }
    else if (format_.compare("h") == 0) {
      tojson_integer<int16_t>(builder, include_beginendlist);
    }
    else if (format_.compare("H") == 0) {
      tojson_integer<uint16_t>(builder, include_beginendlist);
    }
    else if (format_.compare("b") == 0) {
      tojson_integer<int8_t>(builder, include_beginendlist);
    }
    else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
      tojson_integer<uint8_t>(builder, include_beginendlist);
    }
    else if (format_.compare("?") == 0) {
      tojson_boolean(builder, include_beginendlist);
    }
    else {
      throw std::invalid_argument(
        std::string("cannot convert Numpy format \"") + format_
        + std::string("\" into JSON"));
    }
  }

  void
  NumpyArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
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

  int64_t
  NumpyArray::length() const {
    if (isscalar()) {
      return -1;   // just like Record, which is also a scalar
    }
    else {
      return (int64_t)shape_[0];
    }
  }

  const ContentPtr
  NumpyArray::shallow_copy() const {
    return std::make_shared<NumpyArray>(identities_,
                                        parameters_,
                                        ptr_,
                                        shape_,
                                        strides_,
                                        byteoffset_,
                                        itemsize_,
                                        format_);
  }

  const ContentPtr
  NumpyArray::deep_copy(bool copyarrays,
                        bool copyindexes,
                        bool copyidentities) const {
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
    IdentitiesPtr identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<NumpyArray>(identities,
                                        parameters_,
                                        ptr,
                                        shape,
                                        strides,
                                        byteoffset,
                                        itemsize_,
                                        format_);
  }

  void
  NumpyArray::check_for_iteration() const {
    if (identities_.get() != nullptr  &&
        identities_.get()->length() < shape_[0]) {
      util::handle_error(
        failure("len(identities) < len(array)", kSliceNone, kSliceNone),
        identities_.get()->classname(),
        nullptr);
    }
  }

  const ContentPtr
  NumpyArray::getitem_nothing() const {
    const std::vector<ssize_t> shape({ 0 });
    const std::vector<ssize_t> strides({ itemsize_ });
    IdentitiesPtr identities;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(0, 0);
    }
    return std::make_shared<NumpyArray>(identities,
                                        parameters_,
                                        ptr_,
                                        shape,
                                        strides,
                                        byteoffset_,
                                        itemsize_,
                                        format_);
  }

  const ContentPtr
  NumpyArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += shape_[0];
    }
    if (regular_at < 0  ||  regular_at >= shape_[0]) {
      util::handle_error(
        failure("index out of range", kSliceNone, at),
        classname(),
        identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const ContentPtr
  NumpyArray::getitem_at_nowrap(int64_t at) const {
    ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)at);
    const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
    const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
    IdentitiesPtr identities;
    if (identities_.get() != nullptr) {
      if (at >= identities_.get()->length()) {
        util::handle_error(
          failure("index out of range", kSliceNone, at),
          identities_.get()->classname(),
          nullptr);
      }
      identities = identities_.get()->getitem_range_nowrap(at, at + 1);
    }
    return std::make_shared<NumpyArray>(identities,
                                        parameters_,
                                        ptr_,
                                        shape,
                                        strides,
                                        byteoffset,
                                        itemsize_,
                                        format_);
  }

  const ContentPtr
  NumpyArray::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), shape_[0]);
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  const ContentPtr
  NumpyArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)start);
    std::vector<ssize_t> shape;
    shape.push_back((ssize_t)(stop - start));
    shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
    IdentitiesPtr identities;
    if (identities_.get() != nullptr) {
      if (stop > identities_.get()->length()) {
        util::handle_error(
          failure("index out of range", kSliceNone, stop),
          identities_.get()->classname(),
          nullptr);
      }
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<NumpyArray>(identities,
                                        parameters_,
                                        ptr_,
                                        shape,
                                        strides_,
                                        byteoffset,
                                        itemsize_,
                                        format_);
  }

  const ContentPtr
  NumpyArray::getitem_field(const std::string& key) const {
    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by field name"));
  }

  const ContentPtr
  NumpyArray::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by field names"));
  }

  bool getitem_too_general(const SliceItemPtr& head, const Slice& tail) {
    if (head.get() == nullptr) {
      return false;
    }
    else if (dynamic_cast<SliceMissing64*>(head.get())  ||
             dynamic_cast<SliceJagged64*>(head.get())) {
      return true;
    }
    else {
      return getitem_too_general(tail.head(), tail.tail());
    }
  }

  const ContentPtr
  NumpyArray::getitem(const Slice& where) const {
    if (isscalar()) {
      throw std::runtime_error("cannot get-item on a scalar");
    }

    if (getitem_too_general(where.head(), where.tail())) {
      if (ndim() == 1) {
        return Content::getitem(where);
      }
      else {
        return toRegularArray().get()->getitem(where);
      }
    }

    else if (!where.isadvanced()  &&  identities_.get() == nullptr) {
      std::vector<ssize_t> nextshape = { 1 };
      nextshape.insert(nextshape.end(), shape_.begin(), shape_.end());
      std::vector<ssize_t> nextstrides = { shape_[0]*strides_[0] };
      nextstrides.insert(nextstrides.end(), strides_.begin(), strides_.end());
      NumpyArray next(identities_,
                      parameters_,
                      ptr_,
                      nextshape,
                      nextstrides,
                      byteoffset_,
                      itemsize_,
                      format_);

      SliceItemPtr nexthead = where.head();
      Slice nexttail = where.tail();
      NumpyArray out = next.getitem_bystrides(nexthead, nexttail, 1);

      std::vector<ssize_t> outshape(out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides(out.strides_.begin() + 1,
                                      out.strides_.end());
      return std::make_shared<NumpyArray>(out.identities_,
                                          out.parameters_,
                                          out.ptr_,
                                          outshape,
                                          outstrides,
                                          out.byteoffset_,
                                          itemsize_,
                                          format_);
    }

    else {
      NumpyArray safe = contiguous();

      std::vector<ssize_t> nextshape = { 1 };
      nextshape.insert(nextshape.end(),
                       safe.shape_.begin(),
                       safe.shape_.end());
      std::vector<ssize_t> nextstrides = { safe.shape_[0]*safe.strides_[0] };
      nextstrides.insert(nextstrides.end(),
                         safe.strides_.begin(),
                         safe.strides_.end());
      NumpyArray next(safe.identities_,
                      safe.parameters_,
                      safe.ptr_,
                      nextshape,
                      nextstrides,
                      safe.byteoffset_,
                      itemsize_,
                      format_);

      SliceItemPtr nexthead = where.head();
      Slice nexttail = where.tail();
      Index64 nextcarry(1);
      nextcarry.setitem_at_nowrap(0, 0);
      Index64 nextadvanced(0);
      NumpyArray out = next.getitem_next(nexthead,
                                         nexttail,
                                         nextcarry,
                                         nextadvanced,
                                         1,
                                         next.strides_[0],
                                         true);

      std::vector<ssize_t> outshape(out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides(out.strides_.begin() + 1,
                                      out.strides_.end());
      return std::make_shared<NumpyArray>(out.identities_,
                                          out.parameters_,
                                          out.ptr_,
                                          outshape,
                                          outstrides,
                                          out.byteoffset_,
                                          itemsize_,
                                          format_);
    }
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceItemPtr& head,
                           const Slice& tail,
                           const Index64& advanced) const {
    Index64 carry(shape_[0]);
    struct Error err = kernel::carry_arange<int64_t>(carry.ptr().get(), shape_[0]);
    util::handle_error(err, classname(), identities_.get());
    return getitem_next(head,
                        tail,
                        carry,
                        advanced,
                        shape_[0],
                        strides_[0],
                        false).shallow_copy();
  }

  const ContentPtr
  NumpyArray::carry(const Index64& carry, bool allow_lazy) const {
    std::shared_ptr<void> ptr(
      kernel::ptr_alloc<uint8_t>(ptr_lib_, (size_t)(carry.length()*strides_[0])));
    struct Error err = kernel::NumpyArray_getitem_next_null_64(
      reinterpret_cast<uint8_t*>(ptr.get()),
      reinterpret_cast<uint8_t*>(ptr_.get()),
      carry.length(),
      strides_[0],
      byteoffset_,
      carry.ptr().get());
    util::handle_error(err, classname(), identities_.get());

    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }

    std::vector<ssize_t> shape = { (ssize_t)carry.length() };
    shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
    return std::make_shared<NumpyArray>(identities,
                                        parameters_,
                                        ptr,
                                        shape,
                                        strides_,
                                        0,
                                        itemsize_,
                                        format_);
  }

  int64_t
  NumpyArray::numfields() const {
    return -1;
  }

  int64_t
  NumpyArray::fieldindex(const std::string& key) const {
    throw std::invalid_argument(
      std::string("key ") + util::quote(key, true)
      + std::string(" does not exist (data are not records)"));
  }

  const std::string
  NumpyArray::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("fieldindex \"") + std::to_string(fieldindex)
      + std::string("\" does not exist (data are not records)"));
  }

  bool
  NumpyArray::haskey(const std::string& key) const {
    return false;
  }

  const std::vector<std::string>
  NumpyArray::keys() const {
    return std::vector<std::string>();
  }

  const std::string
  NumpyArray::validityerror(const std::string& path) const {
    if (shape_.empty()) {
      return (std::string("at ") + path + std::string(" (") + classname()
              + std::string("): shape is zero-dimensional"));
    }
    for (size_t i = 0;  i < shape_.size();  i++) {
      if (shape_[i] < 0) {
        return (std::string("at ") + path + std::string(" (") + classname()
                + std::string("): shape[") + std::to_string(i) + ("] < 0"));
      }
    }
    for (size_t i = 0;  i < strides_.size();  i++) {
      if (strides_[i] % itemsize_ != 0) {
        return (std::string("at ") + path + std::string(" (") + classname()
                + std::string("): shape[") + std::to_string(i)
                + ("] % itemsize != 0"));
      }
    }
    return std::string();
  }

  const ContentPtr
  NumpyArray::shallow_simplify() const {
    return shallow_copy();
  }

  const ContentPtr
  NumpyArray::num(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      Index64 out(1);
      out.setitem_at_nowrap(0, length());
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    std::vector<ssize_t> shape;
    int64_t reps = 1;
    int64_t size = length();
    int64_t i = 0;
    while (i < ndim() - 1  &&  depth < toaxis) {
      shape.push_back(shape_[(size_t)i]);
      reps *= shape_[(size_t)i];
      size = shape_[(size_t)i + 1];
      i++;
      depth++;
    }
    if (toaxis > depth) {
      throw std::invalid_argument("'axis' out of range for 'num'");
    }

    ssize_t x = sizeof(int64_t);
    std::vector<ssize_t> strides;
    for (int64_t j = (int64_t)shape.size();  j > 0;  j--) {
      strides.insert(strides.begin(), x);
      x *= shape[(size_t)(j - 1)];
    }

    Index64 tonum(reps);
    struct Error err = kernel::RegularArray_num_64(
      tonum.ptr().get(),
      size,
      reps);
    util::handle_error(err, classname(), identities_.get());

    return std::make_shared<NumpyArray>(
      Identities::none(),
      util::Parameters(),
      tonum.ptr(),
      shape,
      strides,
      0,
      sizeof(int64_t),
      format_map.at(std::type_index(typeid(int64_t))));
  }

  const std::vector<ssize_t>
  flatten_shape(const std::vector<ssize_t> shape) {
    if (shape.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      std::vector<ssize_t> out = { shape[0]*shape[1] };
      out.insert(out.end(), shape.begin() + 2, shape.end());
      return out;
    }
  }

  const std::vector<ssize_t>
  flatten_strides(const std::vector<ssize_t> strides) {
    if (strides.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      return std::vector<ssize_t>(strides.begin() + 1, strides.end());
    }
  }

  const std::vector<ssize_t>
  flatten_shape(const std::vector<ssize_t>& shape, int64_t axis) {
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

  const std::vector<ssize_t>
  flatten_strides(const std::vector<ssize_t>& strides, int64_t axis) {
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

  const std::pair<Index64, ContentPtr>
  NumpyArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      throw std::invalid_argument("axis=0 not allowed for flatten");
    }
    else if (shape_.size() != 1  ||  !iscontiguous()) {
      return toRegularArray().get()->offsets_and_flattened(axis, depth);
    }
    else {
      throw std::invalid_argument("axis out of range for flatten");
    }
  }

  bool
  NumpyArray::mergeable(const ContentPtr& other, bool mergebool) const {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
      return mergeable(raw->array(), mergebool);
    }

    if (!parameters_equal(other.get()->parameters())) {
      return false;
    }

    if (dynamic_cast<EmptyArray*>(other.get())  ||
        dynamic_cast<UnionArray8_32*>(other.get())  ||
        dynamic_cast<UnionArray8_U32*>(other.get())  ||
        dynamic_cast<UnionArray8_64*>(other.get())) {
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

    if (ndim() == 0) {
      return false;
    }

    if (NumpyArray* rawother = dynamic_cast<NumpyArray*>(other.get())) {
      if (ndim() != rawother->ndim()) {
        return false;
      }

      std::string other_format = rawother->format();

      if (!mergebool  &&
          ((format_.compare("?") == 0  &&  other_format.compare("?") != 0)  ||
           (format_.compare("?") != 0  &&  other_format.compare("?") == 0))) {
        return false;
      }

      if (!(format_.compare("d") == 0  ||
            format_.compare("f") == 0  ||
            format_.compare("q") == 0  ||
            format_.compare("Q") == 0  ||
            format_.compare("l") == 0  ||
            format_.compare("L") == 0  ||
            format_.compare("i") == 0  ||
            format_.compare("I") == 0  ||
            format_.compare("h") == 0  ||
            format_.compare("H") == 0  ||
            format_.compare("b") == 0  ||
            format_.compare("B") == 0  ||
            format_.compare("c") == 0  ||
            format_.compare("?") == 0  ||
            other_format.compare("d") == 0  ||
            other_format.compare("f") == 0  ||
            other_format.compare("q") == 0  ||
            other_format.compare("Q") == 0  ||
            other_format.compare("l") == 0  ||
            other_format.compare("L") == 0  ||
            other_format.compare("i") == 0  ||
            other_format.compare("I") == 0  ||
            other_format.compare("h") == 0  ||
            other_format.compare("H") == 0  ||
            other_format.compare("b") == 0  ||
            other_format.compare("B") == 0  ||
            other_format.compare("c") == 0  ||
            other_format.compare("?") == 0)) {
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

  const ContentPtr
  NumpyArray::merge(const ContentPtr& other) const {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
      return merge(raw->array());
    }

    if (!parameters_equal(other.get()->parameters())) {
      return merge_as_union(other);
    }

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
    else if (UnionArray8_32* rawother =
             dynamic_cast<UnionArray8_32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnionArray8_U32* rawother =
             dynamic_cast<UnionArray8_U32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnionArray8_64* rawother =
             dynamic_cast<UnionArray8_64*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }

    if (ndim() == 0) {
      throw std::invalid_argument("cannot merge Numpy scalars");
    }

    if ((parameter_equals("__array__", "\"byte\"")  ||
         parameter_equals("__array__", "\"char\""))  &&
        (other.get()->parameter_equals("__array__", "\"byte\"")  ||
         other.get()->parameter_equals("__array__", "\"char\""))) {
      if (std::shared_ptr<NumpyArray> othernumpy =
            std::dynamic_pointer_cast<NumpyArray>(other)) {
        if (ndim() == 1  &&  othernumpy.get()->ndim() == 1  &&
            itemsize() == 1  &&  othernumpy.get()->itemsize() == 1) {
          return merge_bytes(othernumpy);
        }
      }
    }

    NumpyArray contiguous_self = contiguous();
    if (NumpyArray* rawother = dynamic_cast<NumpyArray*>(other.get())) {
      if (ndim() != rawother->ndim()) {
        throw std::invalid_argument(
          "cannot merge arrays with different shapes");
      }

      std::string other_format = rawother->format();

      ssize_t itemsize;
      std::string format;
      if (format_.compare("d") == 0  ||
          format_.compare("f") == 0  ||
          other_format.compare("d") == 0  ||
          other_format.compare("f") == 0) {
        itemsize = 8;
        format = "d";
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("Q") == 0  &&  other_format.compare("Q") == 0) {
        itemsize = 8;
        format = "Q";
#else
      else if (format_.compare("L") == 0  &&  other_format.compare("L") == 0) {
        itemsize = 8;
        format = "L";
#endif
      }
      else if (format_.compare("q") == 0  ||
               format_.compare("Q") == 0  ||
               format_.compare("l") == 0  ||
               format_.compare("L") == 0  ||
               format_.compare("i") == 0  ||
               format_.compare("I") == 0  ||
               format_.compare("h") == 0  ||
               format_.compare("H") == 0  ||
               format_.compare("b") == 0  ||
               format_.compare("B") == 0  ||
               format_.compare("c") == 0  ||
               other_format.compare("q") == 0  ||
               other_format.compare("Q") == 0  ||
               other_format.compare("l") == 0  ||
               other_format.compare("L") == 0  ||
               other_format.compare("i") == 0  ||
               other_format.compare("I") == 0  ||
               other_format.compare("h") == 0  ||
               other_format.compare("H") == 0  ||
               other_format.compare("b") == 0  ||
               other_format.compare("B") == 0  ||
               other_format.compare("c") == 0) {
        itemsize = 8;
#if defined _MSC_VER || defined __i386__
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
        throw std::invalid_argument(
          std::string("cannot merge Numpy format \"") + format_
          + std::string("\" with \"") + other_format + std::string("\""));
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
          throw std::invalid_argument(
            "cannot merge arrays with different shapes");
        }
        shape.insert(shape.begin() + 1, shape_[(size_t)i]);
        strides.insert(strides.begin(), strides[0]*shape_[(size_t)i]);
        self_flatlength *= (int64_t)shape_[(size_t)i];
        other_flatlength *= (int64_t)shape_[(size_t)i];
      }

      std::shared_ptr<void> ptr(
        kernel::ptr_alloc<uint8_t>(ptr_lib_, (size_t)(itemsize*(self_flatlength + other_flatlength))));

      NumpyArray contiguous_other = rawother->contiguous();

      int64_t self_offset = (contiguous_self.byteoffset() /
                             contiguous_self.itemsize());
      int64_t other_offset = (contiguous_other.byteoffset() /
                              contiguous_other.itemsize());

      struct Error err;
      if (format.compare("d") == 0) {
        if (format_.compare("d") == 0) {
          err = kernel::NumpyArray_fill<double, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<double*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("f") == 0) {
          err = kernel::NumpyArray_fill<float, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<float*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
#if defined _MSC_VER || defined __i386__
        else if (format_.compare("q") == 0) {
#else
        else if (format_.compare("l") == 0) {
#endif
          err = kernel::NumpyArray_fill<int64_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<int64_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (format_.compare("Q") == 0) {
#else
          else if (format_.compare("L") == 0) {
#endif
          err = kernel::NumpyArray_fill<uint64_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<uint64_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (format_.compare("l") == 0) {
#else
          else if (format_.compare("i") == 0) {
#endif
          err = kernel::NumpyArray_fill<int32_t , double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<int32_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (format_.compare("L") == 0) {
#else
          else if (format_.compare("I") == 0) {
#endif
          err = kernel::NumpyArray_fill<uint32_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<uint32_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("h") == 0) {
          err = kernel::NumpyArray_fill<int16_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<int16_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("H") == 0) {
          err = kernel::NumpyArray_fill<uint16_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<uint16_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("b") == 0) {
          err = kernel::NumpyArray_fill<int8_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<int8_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
          err = kernel::NumpyArray_fill<uint8_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<uint8_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("?") == 0) {
          err = kernel::NumpyArray_fill_frombool<double>(
                  reinterpret_cast<double*>(ptr.get()),
                  0,
                  reinterpret_cast<bool*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else {
          throw std::invalid_argument(
            std::string("cannot merge Numpy format \"") + format_
            + std::string("\" with \"") + other_format + std::string("\""));
        }
        util::handle_error(err, classname(), nullptr);

        if (other_format.compare("d") == 0) {
          err = kernel::NumpyArray_fill<double, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<double*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("f") == 0) {
          err = kernel::NumpyArray_fill<float, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<float*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
#if defined _MSC_VER || defined __i386__
        else if (other_format.compare("q") == 0) {
#else
        else if (other_format.compare("l") == 0) {
#endif
          err = kernel::NumpyArray_fill<int64_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<int64_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (other_format.compare("Q") == 0) {
#else
          else if (other_format.compare("L") == 0) {
#endif
          err = kernel::NumpyArray_fill<uint64_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<uint64_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (other_format.compare("l") == 0) {
#else
          else if (other_format.compare("i") == 0) {
#endif
          err = kernel::NumpyArray_fill<int32_t , double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<int32_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (other_format.compare("L") == 0) {
#else
          else if (other_format.compare("I") == 0) {
#endif
          err = kernel::NumpyArray_fill<uint32_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<uint32_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("h") == 0) {
          err = kernel::NumpyArray_fill<int16_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<int16_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("H") == 0) {
          err = kernel::NumpyArray_fill<uint16_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<uint16_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("b") == 0) {
          err = kernel::NumpyArray_fill<int8_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<int8_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("B") == 0  ||
                 format_.compare("c") == 0) {
          err = kernel::NumpyArray_fill<uint8_t, double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<uint8_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("?") == 0) {
          err = kernel::NumpyArray_fill_frombool<double>(
                  reinterpret_cast<double*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<bool*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else {
          throw std::invalid_argument(
            std::string("cannot merge Numpy format \"") + format_
            + std::string("\" with \"") + other_format + std::string("\""));
        }
        util::handle_error(err, classname(), nullptr);
      }

      else if (format.compare("Q") == 0  ||  format.compare("L") == 0) {
        err = kernel::NumpyArray_fill<uint64_t, uint64_t>(
                reinterpret_cast<uint64_t*>(ptr.get()),
                0,
                reinterpret_cast<uint64_t*>(contiguous_self.ptr().get()),
                self_offset,
                self_flatlength);
        util::handle_error(err, classname(), nullptr);
        err = kernel::NumpyArray_fill<uint64_t, uint64_t>(
                reinterpret_cast<uint64_t*>(ptr.get()),
                self_flatlength,
                reinterpret_cast<uint64_t*>(contiguous_other.ptr().get()),
                other_offset,
                other_flatlength);
        util::handle_error(err, classname(), nullptr);
      }

      else if (itemsize == 8) {
#if defined _MSC_VER || defined __i386__
        if (format_.compare("q") == 0) {
#else
        if (format_.compare("l") == 0) {
#endif
          err = kernel::NumpyArray_fill<int64_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  0,
                  reinterpret_cast<int64_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (format_.compare("Q") == 0) {
#else
          else if (format_.compare("L") == 0) {
#endif
          err = kernel::NumpyArray_fill_to64_fromU64(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  0,
                  reinterpret_cast<uint64_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (format_.compare("l") == 0) {
#else
          else if (format_.compare("i") == 0) {
#endif
          err = kernel::NumpyArray_fill<int32_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  0,
                  reinterpret_cast<int32_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (format_.compare("L") == 0) {
#else
          else if (format_.compare("I") == 0) {
#endif
          err = kernel::NumpyArray_fill<uint32_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  0,
                  reinterpret_cast<uint32_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("h") == 0) {
          err = kernel::NumpyArray_fill<int16_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  0,
                  reinterpret_cast<int16_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("H") == 0) {
          err = kernel::NumpyArray_fill<uint16_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  0,
                  reinterpret_cast<uint16_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("b") == 0) {
          err = kernel::NumpyArray_fill<int8_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  0,
                  reinterpret_cast<int8_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
          err = kernel::NumpyArray_fill<uint8_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  0,
                  reinterpret_cast<uint8_t*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else if (format_.compare("?") == 0) {
          err = kernel::NumpyArray_fill_frombool<int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  0,
                  reinterpret_cast<bool*>(contiguous_self.ptr().get()),
                  self_offset,
                  self_flatlength);
        }
        else {
          throw std::invalid_argument(
            std::string("cannot merge Numpy format \"") + format_
            + std::string("\" with \"") + other_format + std::string("\""));
        }
        util::handle_error(err, classname(), nullptr);

#if defined _MSC_VER || defined __i386__
        if (other_format.compare("q") == 0) {
#else
        if (other_format.compare("l") == 0) {
#endif
          err = kernel::NumpyArray_fill<int64_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<int64_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (other_format.compare("Q") == 0) {
#else
          else if (other_format.compare("L") == 0) {
#endif
          err = kernel::NumpyArray_fill_to64_fromU64(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<uint64_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (other_format.compare("l") == 0) {
#else
          else if (other_format.compare("i") == 0) {
#endif
          err = kernel::NumpyArray_fill<int32_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<int32_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
#if defined _MSC_VER || defined __i386__
          else if (other_format.compare("L") == 0) {
#else
          else if (other_format.compare("I") == 0) {
#endif
          err = kernel::NumpyArray_fill<uint32_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<uint32_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("h") == 0) {
          err = kernel::NumpyArray_fill<int16_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<int16_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("H") == 0) {
          err = kernel::NumpyArray_fill<uint16_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<uint16_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("b") == 0) {
          err = kernel::NumpyArray_fill<int8_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<int8_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("B") == 0  ||
                 format_.compare("c") == 0) {
          err = kernel::NumpyArray_fill<uint8_t, int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<uint8_t*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else if (other_format.compare("?") == 0) {
          err = kernel::NumpyArray_fill_frombool<int64_t>(
                  reinterpret_cast<int64_t*>(ptr.get()),
                  self_flatlength,
                  reinterpret_cast<bool*>(contiguous_other.ptr().get()),
                  other_offset,
                  other_flatlength);
        }
        else {
          throw std::invalid_argument(
            std::string("cannot merge Numpy format \"") + format_
            + std::string("\" with \"") + other_format + std::string("\""));
        }
        util::handle_error(err, classname(), nullptr);
      }

      else {
        err = kernel::NumpyArray_fill_frombool<bool>(
                reinterpret_cast<bool*>(ptr.get()),
                0,
                reinterpret_cast<bool*>(contiguous_self.ptr().get()),
                self_offset,
                self_flatlength);
        util::handle_error(err, classname(), nullptr);
        err = kernel::NumpyArray_fill_frombool<bool>(
                reinterpret_cast<bool*>(ptr.get()),
                self_flatlength,
                reinterpret_cast<bool*>(contiguous_other.ptr().get()),
                other_offset,
                other_flatlength);
        util::handle_error(err, classname(), nullptr);
      }

      return std::make_shared<NumpyArray>(Identities::none(),
                                          parameters_,
                                          ptr,
                                          shape,
                                          strides,
                                          0,
                                          itemsize,
                                          format);
    }

    else {
      throw std::invalid_argument(
        std::string("cannot merge ") + classname() + std::string(" with ")
        + other.get()->classname());
    }
  }

  const ContentPtr
  NumpyArray::merge_bytes(const std::shared_ptr<NumpyArray>& other) const {
    NumpyArray contiguous_self = contiguous();
    NumpyArray contiguous_other = other.get()->contiguous();

    std::shared_ptr<void> ptr(
      kernel::ptr_alloc<uint8_t>(ptr_lib_, (size_t)(length() + other.get()->length())));

    struct Error err;

    err = kernel::NumpyArray_fill<int8_t, int8_t>(
            reinterpret_cast<int8_t*>(ptr.get()),
            0,
            reinterpret_cast<int8_t*>(contiguous_self.ptr().get()),
            (int64_t)contiguous_self.byteoffset(),
            contiguous_self.length());
    util::handle_error(err, classname(), nullptr);

    err = kernel::NumpyArray_fill<int8_t, int8_t>(
            reinterpret_cast<int8_t*>(ptr.get()),
            length(),
            reinterpret_cast<int8_t*>(contiguous_other.ptr().get()),
            (int64_t)contiguous_other.byteoffset(),
            contiguous_other.length());
    util::handle_error(err, classname(), nullptr);

    std::vector<ssize_t> shape({ (ssize_t)(length() + other.get()->length()) });
    std::vector<ssize_t> strides({ 1 });
    return std::make_shared<NumpyArray>(Identities::none(),
                                        parameters_,
                                        ptr,
                                        shape,
                                        strides,
                                        0,
                                        1,
                                        format_);
  }

  const SliceItemPtr
  NumpyArray::asslice() const {
    if (ndim() != 1) {
      throw std::invalid_argument(
        "slice items can have all fixed-size dimensions (to follow NumPy's "
        "slice rules) or they can have all var-sized dimensions (for jagged "
        "indexing), but not both in the same slice item");
    }
#if defined _MSC_VER || defined __i386__
    if (format_.compare("q") == 0) {
#else
    if (format_.compare("l") == 0) {
#endif
      int64_t* raw = reinterpret_cast<int64_t*>(ptr_.get());
      std::shared_ptr<int64_t> ptr(ptr_, raw);
      std::vector<int64_t> shape({ (int64_t)shape_[0] });
      std::vector<int64_t> strides({ (int64_t)strides_[0] /
                                     (int64_t)itemsize_ });
      return std::make_shared<SliceArray64>(
        Index64(ptr, (int64_t)byteoffset_ / (int64_t)itemsize_, length()),
        shape,
        strides,
        false);
    }
    else if (format_.compare("q") == 0  ||
             format_.compare("Q") == 0  ||
             format_.compare("l") == 0  ||
             format_.compare("L") == 0  ||
             format_.compare("i") == 0  ||
             format_.compare("I") == 0  ||
             format_.compare("h") == 0  ||
             format_.compare("H") == 0  ||
             format_.compare("b") == 0  ||
             format_.compare("B") == 0  ||
             format_.compare("c") == 0) {
      NumpyArray contiguous_self = contiguous();
      int64_t offset = ((int64_t)contiguous_self.byteoffset() /
                        (int64_t)itemsize_);
      Index64 index(length());
      struct Error err;
#if defined _MSC_VER || defined __i386__
      if (format_.compare("Q") == 0) {
#else
      if (format_.compare("L") == 0) {
#endif
        err = kernel::NumpyArray_fill_to64_fromU64(
                index.ptr().get(),
                0,
                reinterpret_cast<uint64_t*>(contiguous_self.ptr().get()),
                offset,
                length());
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("l") == 0) {
#else
      else if (format_.compare("i") == 0) {
#endif
        err = kernel::NumpyArray_fill<int32_t, int64_t>(
                index.ptr().get(),
                0,
                reinterpret_cast<int32_t*>(contiguous_self.ptr().get()),
                offset,
                length());
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("L") == 0) {
#else
      else if (format_.compare("I") == 0) {
#endif
        err = kernel::NumpyArray_fill<uint32_t, int64_t>(
                index.ptr().get(),
                0,
                reinterpret_cast<uint32_t*>(contiguous_self.ptr().get()),
                offset,
                length());
      }
      else if (format_.compare("h") == 0) {
        err = kernel::NumpyArray_fill<int16_t, int64_t>(
                index.ptr().get(),
                0,
                reinterpret_cast<int16_t*>(contiguous_self.ptr().get()),
                offset,
                length());
      }
      else if (format_.compare("H") == 0) {
        err = kernel::NumpyArray_fill<uint16_t, int64_t>(
                index.ptr().get(),
                0,
                reinterpret_cast<uint16_t*>(contiguous_self.ptr().get()),
                offset,
                length());
      }
      else if (format_.compare("b") == 0) {
        err = kernel::NumpyArray_fill<int8_t, int64_t>(
                index.ptr().get(),
                0,
                reinterpret_cast<int8_t*>(contiguous_self.ptr().get()),
                offset,
                length());
      }
      else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
        err = kernel::NumpyArray_fill<uint8_t, int64_t>(
                index.ptr().get(),
                0,
                reinterpret_cast<uint8_t*>(contiguous_self.ptr().get()),
                offset,
                length());
      }
      else {
        throw std::runtime_error("oops: check format_.compare cases above");
      }
      util::handle_error(err, classname(), identities_.get());

      std::vector<int64_t> shape({ (int64_t)shape_[0] });
      std::vector<int64_t> strides({ 1 });
      return std::make_shared<SliceArray64>(index, shape, strides, false);
    }
    else if (format_.compare("?") == 0) {
      int64_t numtrue;
      struct Error err1 = kernel::NumpyArray_getitem_boolean_numtrue(
        &numtrue,
        reinterpret_cast<int8_t*>(ptr_.get()),
        (int64_t)byteoffset_,
        (int64_t)shape_[0],
        (int64_t)strides_[0]);
      util::handle_error(err1, classname(), identities_.get());

      Index64 index(numtrue);
      struct Error err2 = kernel::NumpyArray_getitem_boolean_nonzero_64(
        index.ptr().get(),
        reinterpret_cast<int8_t*>(ptr_.get()),
        (int64_t)byteoffset_,
        (int64_t)shape_[0],
        (int64_t)strides_[0]);
      util::handle_error(err2, classname(), identities_.get());

      std::vector<int64_t> shape({ numtrue });
      std::vector<int64_t> strides({ 1 });
      return std::make_shared<SliceArray64>(index, shape, strides, true);
    }
    else {
      throw std::invalid_argument(
        "only arrays of integers or booleans may be used as a slice");
    }
  }

  const ContentPtr
  NumpyArray::fillna(const ContentPtr& value) const {
    return shallow_copy();
  }

  const ContentPtr
  NumpyArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    if (ndim() == 0) {
      throw std::runtime_error("cannot rpad a scalar");
    }
    else if (ndim() > 1  ||  !iscontiguous()) {
      return toRegularArray().get()->rpad(target, axis, depth);
    }
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
  NumpyArray::rpad_and_clip(int64_t target,
                            int64_t axis,
                            int64_t depth) const {
    if (ndim() == 0) {
      throw std::runtime_error("cannot rpad a scalar");
    }
    else if (ndim() > 1  ||  !iscontiguous()) {
      return toRegularArray().get()->rpad_and_clip(target, axis, depth);
    }
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis != depth) {
      throw std::invalid_argument("axis exceeds the depth of this array");
    }
    return rpad_axis0(target, true);
  }

  const ContentPtr
  NumpyArray::reduce_next(const Reducer& reducer,
                          int64_t negaxis,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength,
                          bool mask,
                          bool keepdims) const {
    if (shape_.empty()) {
      throw std::runtime_error("attempting to reduce a scalar");
    }
    else if (shape_.size() != 1  ||  !iscontiguous()) {
      return toRegularArray().get()->reduce_next(reducer,
                                                 negaxis,
                                                 starts,
                                                 parents,
                                                 outlength,
                                                 mask,
                                                 keepdims);
    }
    else {
      std::shared_ptr<void> ptr;
      if (format_.compare("?") == 0) {
        ptr = reducer.apply_bool(reinterpret_cast<bool*>(ptr_.get()),
                                 byteoffset_ / itemsize_,
                                 starts,
                                 parents,
                                 outlength);
      }
      else if (format_.compare("b") == 0) {
        ptr = reducer.apply_int8(reinterpret_cast<int8_t*>(ptr_.get()),
                                 byteoffset_ / itemsize_,
                                 starts,
                                 parents,
                                 outlength);
      }
      else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
        ptr = reducer.apply_uint8(reinterpret_cast<uint8_t*>(ptr_.get()),
                                  byteoffset_ / itemsize_,
                                  starts,
                                  parents,
                                  outlength);
      }
      else if (format_.compare("h") == 0) {
        ptr = reducer.apply_int16(reinterpret_cast<int16_t*>(ptr_.get()),
                                  byteoffset_ / itemsize_,
                                  starts,
                                  parents,
                                  outlength);
      }
      else if (format_.compare("H") == 0) {
        ptr = reducer.apply_uint16(reinterpret_cast<uint16_t*>(ptr_.get()),
                                   byteoffset_ / itemsize_,
                                   starts,
                                   parents,
                                   outlength);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("l") == 0) {
#else
      else if (format_.compare("i") == 0) {
#endif
        ptr = reducer.apply_int32(reinterpret_cast<int32_t*>(ptr_.get()),
                                  byteoffset_ / itemsize_,
                                  starts,
                                  parents,
                                  outlength);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("L") == 0) {
#else
      else if (format_.compare("I") == 0) {
#endif
        ptr = reducer.apply_uint32(reinterpret_cast<uint32_t*>(ptr_.get()),
                                   byteoffset_ / itemsize_,
                                   starts,
                                   parents,
                                   outlength);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("q") == 0) {
#else
      else if (format_.compare("l") == 0) {
#endif
        ptr = reducer.apply_int64(reinterpret_cast<int64_t*>(ptr_.get()),
                                  byteoffset_ / itemsize_,
                                  starts,
                                  parents,
                                  outlength);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("Q") == 0) {
#else
      else if (format_.compare("L") == 0) {
#endif
        ptr = reducer.apply_uint64(reinterpret_cast<uint64_t*>(ptr_.get()),
                                   byteoffset_ / itemsize_,
                                   starts,
                                   parents,
                                   outlength);
      }
      else if (format_.compare("f") == 0) {
        ptr = reducer.apply_float32(reinterpret_cast<float*>(ptr_.get()),
                                    byteoffset_ / itemsize_,
                                    starts,
                                    parents,
                                    outlength);
      }
      else if (format_.compare("d") == 0) {
        ptr = reducer.apply_float64(reinterpret_cast<double*>(ptr_.get()),
                                    byteoffset_ / itemsize_,
                                    starts,
                                    parents,
                                    outlength);
      }
      else {
        throw std::invalid_argument(
          std::string("cannot apply reducers to NumpyArray with format \"")
          + format_ + std::string("\""));
      }

      std::string format = reducer.return_type(format_);
      ssize_t itemsize = reducer.return_typesize(format_);
      std::vector<ssize_t> shape({ (ssize_t)outlength });
      std::vector<ssize_t> strides({ itemsize });
      ContentPtr out = std::make_shared<NumpyArray>(Identities::none(),
                                                    util::Parameters(),
                                                    ptr,
                                                    shape,
                                                    strides,
                                                    0,
                                                    itemsize,
                                                    format);

      if (mask) {
        Index8 mask(outlength);
        struct Error err = kernel::NumpyArray_reduce_mask_ByteMaskedArray_64(
          mask.ptr().get(),
          parents.ptr().get(),
          parents.offset(),
          parents.length(),
          outlength);
        util::handle_error(err, classname(), nullptr);
        out = std::make_shared<ByteMaskedArray>(Identities::none(),
                                                util::Parameters(),
                                                mask,
                                                out,
                                                false);
      }

      if (keepdims) {
        out = std::make_shared<RegularArray>(Identities::none(),
                                             util::Parameters(),
                                             out,
                                             1);
      }

      return out;
    }
  }

  const ContentPtr
  NumpyArray::localindex(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (axis == depth) {
      return localindex_axis0();
    }
    else if (shape_.size() <= 1) {
      throw std::invalid_argument("'axis' out of range for localindex");
    }
    else {
      return toRegularArray().get()->localindex(axis, depth);
    }
  }

  const ContentPtr
  NumpyArray::combinations(int64_t n,
                           bool replacement,
                           const util::RecordLookupPtr& recordlookup,
                           const util::Parameters& parameters,
                           int64_t axis,
                           int64_t depth) const {
    if (n < 1) {
      throw std::invalid_argument("in combinations, 'n' must be at least 1");
    }

    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      return combinations_axis0(n, replacement, recordlookup, parameters);
    }

    else if (shape_.size() <= 1) {
      throw std::invalid_argument("'axis' out of range for combinations");
    }

    else {
      return toRegularArray().get()->combinations(n,
                                                  replacement,
                                                  recordlookup,
                                                  parameters,
                                                  axis,
                                                  depth);
    }
  }

  const ContentPtr
  NumpyArray::sort_next(int64_t negaxis,
                        const Index64& starts,
                        const Index64& parents,
                        int64_t outlength,
                        bool ascending,
                        bool stable,
                        bool keepdims) const {
    if (shape_.empty()) {
      throw std::runtime_error("attempting to sort a scalar");
    }
    else if (shape_.size() != 1  ||  !iscontiguous()) {
      return toRegularArray().get()->sort_next(negaxis,
                                               starts,
                                               parents,
                                               outlength,
                                               ascending,
                                               stable,
                                               keepdims);
    }
    else {
      std::shared_ptr<Content> out;
      int64_t offset = byteoffset_ / itemsize_;
      std::shared_ptr<void> ptr;
      if (format_.compare("?") == 0) {
        ptr = array_sort<bool>(reinterpret_cast<bool*>(ptr_.get()),
                               length(),
                               offset,
                               starts,
                               parents,
                               outlength,
                               ascending,
                               stable);
      }
      else if (format_.compare("b") == 0) {
        ptr = array_sort<int8_t>(reinterpret_cast<int8_t*>(ptr_.get()),
                                 length(),
                                 offset,
                                 starts,
                                 parents,
                                 outlength,
                                 ascending,
                                 stable);
      }
      else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
        ptr = array_sort<uint8_t>(reinterpret_cast<uint8_t*>(ptr_.get()),
                                  length(),
                                  offset,
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
      }
      else if (format_.compare("h") == 0) {
        ptr = array_sort<int16_t>(reinterpret_cast<int16_t*>(ptr_.get()),
                                  length(),
                                  offset,
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
      }
      else if (format_.compare("H") == 0) {
        ptr = array_sort<uint16_t>(reinterpret_cast<uint16_t*>(ptr_.get()),
                                   length(),
                                   offset,
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("l") == 0) {
#else
      else if (format_.compare("i") == 0) {
#endif
        ptr = array_sort<int32_t>(reinterpret_cast<int32_t*>(ptr_.get()),
                                  length(),
                                  offset,
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("L") == 0) {
#else
      else if (format_.compare("I") == 0) {
#endif
        ptr = array_sort<uint32_t>(reinterpret_cast<uint32_t*>(ptr_.get()),
                                   length(),
                                   offset,
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("q") == 0) {
#else
      else if (format_.compare("l") == 0) {
#endif
        ptr = array_sort<int64_t>(reinterpret_cast<int64_t*>(ptr_.get()),
                                  length(),
                                  offset,
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("Q") == 0) {
#else
      else if (format_.compare("L") == 0) {
#endif
        ptr = array_sort<uint64_t>(reinterpret_cast<uint64_t*>(ptr_.get()),
                                   length(),
                                   offset,
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
      }
      else if (format_.compare("f") == 0) {
        ptr = array_sort<float>(reinterpret_cast<float*>(ptr_.get()),
                                length(),
                                offset,
                                starts,
                                parents,
                                outlength,
                                ascending,
                                stable);
      }
      else if (format_.compare("d") == 0) {
        ptr = array_sort<double>(reinterpret_cast<double*>(ptr_.get()),
                                 length(),
                                 offset,
                                 starts,
                                 parents,
                                 outlength,
                                 ascending,
                                 stable);
      }
      else {
        throw std::invalid_argument(
          std::string("cannot sort NumpyArray with format \"")
          + format_ + std::string("\""));
      }

      out = std::make_shared<NumpyArray>(Identities::none(),
                                         parameters_,
                                         ptr,
                                         shape_,
                                         strides_,
                                         0,
                                         itemsize_,
                                         format_);

      if (keepdims) {
        out = std::make_shared<RegularArray>(
          Identities::none(),
          util::Parameters(),
          out,
          parents.length() / starts.length());
      }

      return out;
    }
  }

  const ContentPtr
  NumpyArray::argsort_next(int64_t negaxis,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength,
                           bool ascending,
                           bool stable,
                           bool keepdims) const {
    if (shape_.empty()) {
      throw std::runtime_error("attempting to argsort a scalar");
    }
    else if (shape_.size() != 1  ||  !iscontiguous()) {
      return toRegularArray().get()->argsort_next(negaxis,
                                                  starts,
                                                  parents,
                                                  outlength,
                                                  ascending,
                                                  stable,
                                                  keepdims);
    }
    else {
      std::shared_ptr<Content> out;
      int64_t offset = byteoffset_ / itemsize_;
      std::shared_ptr<void> ptr;
      if (format_.compare("?") == 0) {
        ptr = index_sort<bool>(reinterpret_cast<bool*>(ptr_.get()),
                               length(),
                               offset,
                               starts,
                               parents,
                               outlength,
                               ascending,
                               stable);
      }
      else if (format_.compare("b") == 0) {
        ptr = index_sort<int8_t>(reinterpret_cast<int8_t*>(ptr_.get()),
                                 length(),
                                 offset,
                                 starts,
                                 parents,
                                 outlength,
                                 ascending,
                                 stable);
      }
      else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
        ptr = index_sort<uint8_t>(reinterpret_cast<uint8_t*>(ptr_.get()),
                                  length(),
                                  offset,
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
      }
      else if (format_.compare("h") == 0) {
        ptr = index_sort<int16_t>(reinterpret_cast<int16_t*>(ptr_.get()),
                                  length(),
                                  offset,
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
      }
      else if (format_.compare("H") == 0) {
        ptr = index_sort<uint16_t>(reinterpret_cast<uint16_t*>(ptr_.get()),
                                   length(),
                                   offset,
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("l") == 0) {
#else
      else if (format_.compare("i") == 0) {
#endif
        ptr = index_sort<int32_t>(reinterpret_cast<int32_t*>(ptr_.get()),
                                  length(),
                                  offset,
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("L") == 0) {
#else
      else if (format_.compare("I") == 0) {
#endif
        ptr = index_sort<uint32_t>(reinterpret_cast<uint32_t*>(ptr_.get()),
                                   length(),
                                   offset,
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("q") == 0) {
#else
      else if (format_.compare("l") == 0) {
#endif
        ptr = index_sort<int64_t>(reinterpret_cast<int64_t*>(ptr_.get()),
                                  length(),
                                  offset,
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
      }
#if defined _MSC_VER || defined __i386__
      else if (format_.compare("Q") == 0) {
#else
      else if (format_.compare("L") == 0) {
#endif
        ptr = index_sort<uint64_t>(reinterpret_cast<uint64_t*>(ptr_.get()),
                                   length(),
                                   offset,
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
      }
      else if (format_.compare("f") == 0) {
        ptr = index_sort<float>(reinterpret_cast<float*>(ptr_.get()),
                                length(),
                                offset,
                                starts,
                                parents,
                                outlength,
                                ascending,
                                stable);
      }
      else if (format_.compare("d") == 0) {
        ptr = index_sort<double>(reinterpret_cast<double*>(ptr_.get()),
                                 length(),
                                 offset,
                                 starts,
                                 parents,
                                 outlength,
                                 ascending,
                                 stable);
      }
      else {
        throw std::invalid_argument(
          std::string("cannot sort NumpyArray with format \"")
          + format_ + std::string("\""));
      }

      std::string format;
#if defined _MSC_VER || defined __i386__
      format = "q";
#else
      format = "l";
#endif

      ssize_t itemsize = 8;
      std::vector<ssize_t> shape({ (ssize_t)shape_[0] });
      std::vector<ssize_t> strides({ itemsize });
      out = std::make_shared<NumpyArray>(Identities::none(),
                                         util::Parameters(),
                                         ptr,
                                         shape_,
                                         strides,
                                         0,
                                         itemsize,
                                         format);

      if (keepdims) {
        out = std::make_shared<RegularArray>(
          Identities::none(),
          util::Parameters(),
          out,
          parents.length() / starts.length());
      }
      return out;
    }
  }

  const ContentPtr
  NumpyArray::sort_asstrings(const Index64& offsets,
                             bool ascending,
                             bool stable) const {
    std::shared_ptr<Content> out;
    int64_t offset = byteoffset_ / itemsize_;
    std::shared_ptr<void> ptr;

    Index64 outoffsets(offsets.length());

    if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
      ptr = string_sort<uint8_t>(reinterpret_cast<uint8_t*>(ptr_.get()),
                                 length(),
                                 offsets,
                                 outoffsets,
                                 ascending,
                                 stable);
    } else {
      throw std::invalid_argument(
        std::string("cannot sort NumpyArray as strings with format \"")
        + format_ + std::string("\""));
    }

    out = std::make_shared<NumpyArray>(identities_,
                                       parameters_,
                                       ptr,
                                       shape_,
                                       strides_,
                                       0,
                                       itemsize_,
                                       format_);

   out = std::make_shared<ListOffsetArray64>(Identities::none(),
                                             util::Parameters(),
                                             outoffsets,
                                             out);

   return out;
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceAt& at,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: NumpyArray::getitem_next(at) "
      "(without 'length', 'stride', and 'first')");
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceRange& range,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: NumpyArray::getitem_next(range) "
      "(without 'length', 'stride', and 'first')");
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceArray64& array,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: NumpyArray::getitem_next(array) "
      "(without 'length','stride', and 'first')");
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceField& field,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: NumpyArray::getitem_next(field) "
      "(without 'length', 'stride', and 'first')");
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceFields& fields,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: NumpyArray::getitem_next(fields) "
      "(without 'length', 'stride', and 'first')");
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceJagged64& jagged,
                           const Slice& tail, const Index64& advanced) const {
    if (shape_.size() != 1) {
      throw std::runtime_error(
        "undefined operation: NumpyArray::getitem_next(jagged) with "
        "ndim != 1");
    }

    if (advanced.length() != 0) {
      throw std::invalid_argument(
        "cannot mix jagged slice with NumPy-style advanced indexing");
    }

    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by a jagged array because it is one-dimensional"));
  }

  const ContentPtr
  NumpyArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceArray64& slicecontent,
                                  const Slice& tail) const {
    if (ndim() == 1) {
      throw std::invalid_argument(
        "too many jagged slice dimensions for array");
    }
    else {
      throw std::runtime_error(
        std::string("undefined operation: NumpyArray::getitem_next_jagged("
                    "array) for ndim == ") + std::to_string(ndim()));
    }
  }

  const ContentPtr
  NumpyArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceMissing64& slicecontent,
                                  const Slice& tail) const {
    if (ndim() == 1) {
      throw std::invalid_argument(
        "too many jagged slice dimensions for array");
    }
    else {
      throw std::runtime_error(
        std::string("undefined operation: NumpyArray::getitem_next_jagged("
                    "missing) for ndim == ") + std::to_string(ndim()));
    }
  }

  const ContentPtr
  NumpyArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceJagged64& slicecontent,
                                  const Slice& tail) const {
    if (ndim() == 1) {
      throw std::invalid_argument(
        "too many jagged slice dimensions for array");
    }
    else {
      throw std::runtime_error(
        std::string("undefined operation: NumpyArray::getitem_next_jagged("
                    "jagged) for ndim == ") + std::to_string(ndim()));
    }
  }

  bool
  NumpyArray::iscontiguous() const {
    ssize_t x = itemsize_;
    for (ssize_t i = ndim() - 1;  i >= 0;  i--) {
      if (x != strides_[i]) return false;
      x *= shape_[i];
    }
    return true;  // true for isscalar(), too
  }

  const NumpyArray
  NumpyArray::contiguous() const {
    if (iscontiguous()) {
      return NumpyArray(identities_,
                        parameters_,
                        ptr_,
                        shape_,
                        strides_,
                        byteoffset_,
                        itemsize_,
                        format_);
    }
    else {
      Index64 bytepos(shape_[0]);
      struct Error err =
        kernel::NumpyArray_contiguous_init_64(bytepos.ptr().get(),
                                              shape_[0],
                                              strides_[0]);
      util::handle_error(err, classname(), identities_.get());
      return contiguous_next(bytepos);
    }
  }

  const NumpyArray
  NumpyArray::contiguous_next(const Index64& bytepos) const {
    if (iscontiguous()) {
      std::shared_ptr<void> ptr(
        kernel::ptr_alloc<uint8_t>(ptr_lib_, (size_t)(bytepos.length()*strides_[0])));

      struct Error err = kernel::NumpyArray_contiguous_copy_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        bytepos.length(),
        strides_[0],
        byteoffset_,
        bytepos.ptr().get());
      util::handle_error(err, classname(), identities_.get());
      return NumpyArray(identities_,
                        parameters_,
                        ptr,
                        shape_,
                        strides_,
                        0,
                        itemsize_,
                        format_);
    }

    else if (shape_.size() == 1) {
      std::shared_ptr<void> ptr(
        kernel::ptr_alloc<uint8_t>(ptr_lib_, (size_t)(bytepos.length()*itemsize_)));
      struct Error err = kernel::NumpyArray_contiguous_copy_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        bytepos.length(),
        itemsize_,
        byteoffset_,
        bytepos.ptr().get());
      util::handle_error(err, classname(), identities_.get());
      std::vector<ssize_t> strides = { itemsize_ };
      return NumpyArray(identities_,
                        parameters_,
                        ptr,
                        shape_,
                        strides,
                        0,
                        itemsize_,
                        format_);
    }

    else {
      NumpyArray next(identities_,
                      parameters_,
                      ptr_,
                      flatten_shape(shape_),
                      flatten_strides(strides_),
                      byteoffset_,
                      itemsize_,
                      format_);

      Index64 nextbytepos(bytepos.length()*shape_[1]);
      struct Error err = kernel::NumpyArray_contiguous_next_64(
        nextbytepos.ptr().get(),
        bytepos.ptr().get(),
        bytepos.length(),
        (int64_t)shape_[1],
        (int64_t)strides_[1]);
      util::handle_error(err, classname(), identities_.get());

      NumpyArray out = next.contiguous_next(nextbytepos);
      std::vector<ssize_t> outstrides = { shape_[1]*out.strides_[0] };
      outstrides.insert(outstrides.end(),
                        out.strides_.begin(),
                        out.strides_.end());
      return NumpyArray(out.identities_,
                        out.parameters_,
                        out.ptr_,
                        shape_,
                        outstrides,
                        out.byteoffset_,
                        itemsize_,
                        format_);
    }
  }

  const NumpyArray
  NumpyArray::getitem_bystrides(const SliceItemPtr& head,
                                const Slice& tail,
                                int64_t length) const {
    if (head.get() == nullptr) {
      return NumpyArray(identities_,
                        parameters_,
                        ptr_,
                        shape_,
                        strides_,
                        byteoffset_,
                        itemsize_,
                        format_);
    }
    else if (SliceAt* at =
             dynamic_cast<SliceAt*>(head.get())) {
      return getitem_bystrides(*at, tail, length);
    }
    else if (SliceRange* range =
             dynamic_cast<SliceRange*>(head.get())) {
      return getitem_bystrides(*range, tail, length);
    }
    else if (SliceEllipsis* ellipsis =
             dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_bystrides(*ellipsis, tail, length);
    }
    else if (SliceNewAxis* newaxis =
             dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_bystrides(*newaxis, tail, length);
    }
    else {
      throw std::runtime_error(
        "unrecognized slice item type for NumpyArray::getitem_bystrides");
    }
  }

  const NumpyArray
  NumpyArray::getitem_bystrides(const SliceAt& at,
                                const Slice& tail,
                                int64_t length) const {
    if (ndim() < 2) {
      util::handle_error(
        failure("too many dimensions in slice", kSliceNone, kSliceNone),
        classname(),
        identities_.get());
    }

    int64_t i = at.at();
    if (i < 0) i += shape_[1];
    if (i < 0  ||  i >= shape_[1]) {
      util::handle_error(
        failure("index out of range", kSliceNone, at.at()),
        classname(),
        identities_.get());
    }

    ssize_t nextbyteoffset = byteoffset_ + ((ssize_t)i)*strides_[1];
    NumpyArray next(identities_,
                    parameters_,
                    ptr_,
                    flatten_shape(shape_),
                    flatten_strides(strides_),
                    nextbyteoffset,
                    itemsize_,
                    format_);

    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = next.getitem_bystrides(nexthead, nexttail, length);

    std::vector<ssize_t> outshape = { (ssize_t)length };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    return NumpyArray(out.identities_,
                      out.parameters_,
                      out.ptr_,
                      outshape,
                      out.strides_,
                      out.byteoffset_,
                      itemsize_,
                      format_);
  }

  const NumpyArray
  NumpyArray::getitem_bystrides(const SliceRange& range,
                                const Slice& tail,
                                int64_t length) const {
    if (ndim() < 2) {
      util::handle_error(
        failure("too many dimensions in slice", kSliceNone, kSliceNone),
        classname(),
        identities_.get());
    }

    int64_t start = range.start();
    int64_t stop = range.stop();
    int64_t step = range.step();
    if (step == Slice::none()) {
      step = 1;
    }
    kernel::regularize_rangeslice(&start, &stop, step > 0,
      range.hasstart(), range.hasstop(), (int64_t)shape_[1]);

    int64_t numer = std::abs(start - stop);
    int64_t denom = std::abs(step);
    int64_t d = numer / denom;
    int64_t m = numer % denom;
    int64_t lenhead = d + (m != 0 ? 1 : 0);

    ssize_t nextbyteoffset = byteoffset_ + ((ssize_t)start)*strides_[1];
    NumpyArray next(identities_,
                    parameters_,
                    ptr_,
                    flatten_shape(shape_),
                    flatten_strides(strides_),
                    nextbyteoffset,
                    itemsize_,
                    format_);

    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = next.getitem_bystrides(nexthead,
                                            nexttail,
                                            length*lenhead);

    std::vector<ssize_t> outshape = { (ssize_t)length,
                                      (ssize_t)lenhead };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    std::vector<ssize_t> outstrides = { strides_[0],
                                        strides_[1]*((ssize_t)step) };
    outstrides.insert(outstrides.end(),
                      out.strides_.begin() + 1,
                      out.strides_.end());
    return NumpyArray(out.identities_,
                      out.parameters_,
                      out.ptr_,
                      outshape,
                      outstrides,
                      out.byteoffset_,
                      itemsize_,
                      format_);
  }

  const NumpyArray
  NumpyArray::getitem_bystrides(const SliceEllipsis& ellipsis,
                                const Slice& tail,
                                int64_t length) const {
    std::pair<int64_t, int64_t> minmax = minmax_depth();
    int64_t mindepth = minmax.first;

    if (tail.length() == 0  ||  mindepth - 1 == tail.dimlength()) {
      SliceItemPtr nexthead = tail.head();
      Slice nexttail = tail.tail();
      return getitem_bystrides(nexthead, nexttail, length);
    }
    else {
      std::vector<SliceItemPtr> tailitems = tail.items();
      std::vector<SliceItemPtr> items = { std::make_shared<SliceEllipsis>() };
      items.insert(items.end(), tailitems.begin(), tailitems.end());

      SliceItemPtr nexthead = std::make_shared<SliceRange>(Slice::none(),
                                                           Slice::none(),
                                                           1);
      Slice nexttail(items);
      return getitem_bystrides(nexthead, nexttail, length);
    }
  }

  const NumpyArray
  NumpyArray::getitem_bystrides(const SliceNewAxis& newaxis,
                                const Slice& tail,
                                int64_t length) const {
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = getitem_bystrides(nexthead, nexttail, length);

    std::vector<ssize_t> outshape = { (ssize_t)length, 1 };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    std::vector<ssize_t> outstrides = { out.strides_[0] };
    outstrides.insert(outstrides.end(),
                      out.strides_.begin(),
                      out.strides_.end());
    return NumpyArray(out.identities_,
                      out.parameters_,
                      out.ptr_,
                      outshape,
                      outstrides,
                      out.byteoffset_,
                      itemsize_,
                      format_);
  }

  const NumpyArray
  NumpyArray::getitem_next(const SliceItemPtr& head,
                           const Slice& tail,
                           const Index64& carry,
                           const Index64& advanced,
                           int64_t length,
                           int64_t stride,
                           bool first) const {
    if (head.get() == nullptr) {
      std::shared_ptr<void> ptr(kernel::ptr_alloc<uint8_t>(ptr_lib_,
                                                               (size_t)(carry.length()*stride)));
      struct Error err = kernel::NumpyArray_getitem_next_null_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        carry.length(),
        stride,
        byteoffset_,
        carry.ptr().get());
      util::handle_error(err, classname(), identities_.get());

      IdentitiesPtr identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_carry_64(carry);
      }

      std::vector<ssize_t> shape = { (ssize_t)carry.length() };
      shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
      std::vector<ssize_t> strides = { (ssize_t)stride };
      strides.insert(strides.end(), strides_.begin() + 1, strides_.end());
      return NumpyArray(identities,
                        parameters_,
                        ptr,
                        shape,
                        strides,
                        0,
                        itemsize_,
                        format_);
    }

    else if (SliceAt* at =
             dynamic_cast<SliceAt*>(head.get())) {
      return getitem_next(*at,
                          tail,
                          carry,
                          advanced,
                          length,
                          stride,
                          first);
    }
    else if (SliceRange* range =
             dynamic_cast<SliceRange*>(head.get())) {
      return getitem_next(*range,
                          tail,
                          carry,
                          advanced,
                          length,
                          stride,
                          first);
    }
    else if (SliceEllipsis* ellipsis =
             dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_next(*ellipsis,
                          tail,
                          carry,
                          advanced,
                          length,
                          stride,
                          first);
    }
    else if (SliceNewAxis* newaxis =
             dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_next(*newaxis,
                          tail,
                          carry,
                          advanced,
                          length,
                          stride,
                          first);
    }
    else if (SliceArray64* array =
             dynamic_cast<SliceArray64*>(head.get())) {
      return getitem_next(*array,
                          tail,
                          carry,
                          advanced,
                          length,
                          stride,
                          first);
    }
    else if (SliceField* field =
             dynamic_cast<SliceField*>(head.get())) {
      throw std::invalid_argument(
        std::string("cannot slice ") + classname()
        + std::string(" by a field name because it has no fields"));
    }
    else if (SliceFields* fields =
             dynamic_cast<SliceFields*>(head.get())) {
      throw std::invalid_argument(
        std::string("cannot slice ") + classname()
        + std::string(" by field names because it has no fields"));
    }
    else if (SliceMissing64* missing =
             dynamic_cast<SliceMissing64*>(head.get())) {
      throw std::runtime_error(
        "undefined operation: NumpyArray::getitem_next(missing) "
        "(defer to Content::getitem_next(missing))");
    }
    else if (SliceJagged64* jagged =
             dynamic_cast<SliceJagged64*>(head.get())) {
      throw std::runtime_error("FIXME: NumpyArray::getitem_next(jagged)");
    }
    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }

  const NumpyArray
  NumpyArray::getitem_next(const SliceAt& at,
                           const Slice& tail,
                           const Index64& carry,
                           const Index64& advanced,
                           int64_t length,
                           int64_t stride,
                           bool first) const {
    if (ndim() < 2) {
      util::handle_error(
        failure("too many dimensions in slice", kSliceNone, kSliceNone),
        classname(),
        identities_.get());
    }

    NumpyArray next(first ? identities_ : Identities::none(),
                    parameters_,
                    ptr_,
                    flatten_shape(shape_),
                    flatten_strides(strides_),
                    byteoffset_,
                    itemsize_,
                    format_);
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();

    int64_t regular_at = at.at();
    if (regular_at < 0) {
      regular_at += shape_[1];
    }
    if (!(0 <= regular_at  &&  regular_at < shape_[1])) {
      util::handle_error(
        failure("index out of range", kSliceNone, at.at()),
        classname(),
        identities_.get());
    }

    Index64 nextcarry(carry.length());
    struct Error err = kernel::NumpyArray_getitem_next_at_64(
      nextcarry.ptr().get(),
      carry.ptr().get(),
      carry.length(),
      shape_[1],   // because this is contiguous
      regular_at);
    util::handle_error(err, classname(), identities_.get());

    NumpyArray out = next.getitem_next(nexthead,
                                       nexttail,
                                       nextcarry,
                                       advanced,
                                       length,
                                       next.strides_[0],
                                       false);

    std::vector<ssize_t> outshape = { (ssize_t)length };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    return NumpyArray(out.identities_,
                      out.parameters_,
                      out.ptr_,
                      outshape,
                      out.strides_,
                      out.byteoffset_,
                      itemsize_,
                      format_);
  }

  const NumpyArray
  NumpyArray::getitem_next(const SliceRange& range,
                           const Slice& tail,
                           const Index64& carry,
                           const Index64& advanced,
                           int64_t length,
                           int64_t stride,
                           bool first) const {
    if (ndim() < 2) {
      util::handle_error(
        failure("too many dimensions in slice", kSliceNone, kSliceNone),
        classname(),
        identities_.get());
    }

    int64_t start = range.start();
    int64_t stop = range.stop();
    int64_t step = range.step();
    if (step == Slice::none()) {
      step = 1;
    }
    kernel::regularize_rangeslice(&start,
                                  &stop,
                                  step > 0,
                                  range.hasstart(),
                                  range.hasstop(),
                                  (int64_t)shape_[1]);

    int64_t numer = std::abs(start - stop);
    int64_t denom = std::abs(step);
    int64_t d = numer / denom;
    int64_t m = numer % denom;
    int64_t lenhead = d + (m != 0 ? 1 : 0);

    NumpyArray next(first ? identities_ : Identities::none(),
                    parameters_,
                    ptr_,
                    flatten_shape(shape_),
                    flatten_strides(strides_),
                    byteoffset_,
                    itemsize_,
                    format_);
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();

    if (advanced.length() == 0) {
      Index64 nextcarry(carry.length()*lenhead);
      struct Error err = kernel::NumpyArray_getitem_next_range_64(
        nextcarry.ptr().get(),
        carry.ptr().get(),
        carry.length(),
        lenhead,
        shape_[1],   // because this is contiguous
        start,
        step);
      util::handle_error(err, classname(), identities_.get());

      NumpyArray out = next.getitem_next(nexthead,
                                         nexttail,
                                         nextcarry,
                                         advanced,
                                         length*lenhead,
                                         next.strides_[0],
                                         false);
      std::vector<ssize_t> outshape = { (ssize_t)length, (ssize_t)lenhead };
      outshape.insert(outshape.end(),
                      out.shape_.begin() + 1,
                      out.shape_.end());
      std::vector<ssize_t> outstrides = { (ssize_t)lenhead*out.strides_[0] };
      outstrides.insert(outstrides.end(),
                        out.strides_.begin(),
                        out.strides_.end());
      return NumpyArray(out.identities_,
                        out.parameters_,
                        out.ptr_,
                        outshape,
                        outstrides,
                        out.byteoffset_,
                        itemsize_,
                        format_);
    }

    else {
      Index64 nextcarry(carry.length()*lenhead);
      Index64 nextadvanced(carry.length()*lenhead);
      struct Error err = kernel::NumpyArray_getitem_next_range_advanced_64(
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

      NumpyArray out = next.getitem_next(nexthead,
                                         nexttail,
                                         nextcarry,
                                         nextadvanced,
                                         length*lenhead,
                                         next.strides_[0],
                                         false);
      std::vector<ssize_t> outshape = { (ssize_t)length, (ssize_t)lenhead };
      outshape.insert(outshape.end(),
                      out.shape_.begin() + 1,
                      out.shape_.end());
      std::vector<ssize_t> outstrides = { (ssize_t)lenhead*out.strides_[0] };
      outstrides.insert(outstrides.end(),
                        out.strides_.begin(),
                        out.strides_.end());
      return NumpyArray(out.identities_,
                        out.parameters_,
                        out.ptr_,
                        outshape,
                        outstrides,
                        out.byteoffset_,
                        itemsize_,
                        format_);
    }
  }

  const NumpyArray
  NumpyArray::getitem_next(const SliceEllipsis& ellipsis,
                           const Slice& tail,
                           const Index64& carry,
                           const Index64& advanced,
                           int64_t length,
                           int64_t stride,
                           bool first) const {
    std::pair<int64_t, int64_t> minmax = minmax_depth();
    int64_t mindepth = minmax.first;

    if (tail.length() == 0  ||  mindepth - 1 == tail.dimlength()) {
      SliceItemPtr nexthead = tail.head();
      Slice nexttail = tail.tail();
      return getitem_next(nexthead,
                          nexttail,
                          carry,
                          advanced,
                          length,
                          stride,
                          false);
    }
    else {
      std::vector<SliceItemPtr> tailitems = tail.items();
      std::vector<SliceItemPtr> items = { std::make_shared<SliceEllipsis>() };
      items.insert(items.end(), tailitems.begin(), tailitems.end());
      SliceItemPtr nexthead = std::make_shared<SliceRange>(Slice::none(),
                                                           Slice::none(),
                                                           1);
      Slice nexttail(items);
      return getitem_next(nexthead,
                          nexttail,
                          carry,
                          advanced,
                          length,
                          stride,
                          false);
    }
  }

  const NumpyArray
  NumpyArray::getitem_next(const SliceNewAxis& newaxis,
                           const Slice& tail,
                           const Index64& carry,
                           const Index64& advanced,
                           int64_t length,
                           int64_t stride,
                           bool first) const {
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = getitem_next(nexthead,
                                  nexttail,
                                  carry,
                                  advanced,
                                  length,
                                  stride,
                                  false);

    std::vector<ssize_t> outshape = { (ssize_t)length, 1 };
    outshape.insert(outshape.end(),
                    out.shape_.begin() + 1,
                    out.shape_.end());
    std::vector<ssize_t> outstrides = { out.strides_[0] };
    outstrides.insert(outstrides.end(),
                      out.strides_.begin(),
                      out.strides_.end());
    return NumpyArray(out.identities_,
                      out.parameters_,
                      out.ptr_,
                      outshape,
                      outstrides,
                      out.byteoffset_,
                      itemsize_,
                      format_);
  }

  const NumpyArray
  NumpyArray::getitem_next(const SliceArray64& array,
                           const Slice& tail,
                           const Index64& carry,
                           const Index64& advanced,
                           int64_t length,
                           int64_t stride,
                           bool first) const {
    if (ndim() < 2) {
      util::handle_error(
        failure("too many dimensions in slice", kSliceNone, kSliceNone),
        classname(),
        identities_.get());
    }

    NumpyArray next(first ? identities_ : Identities::none(),
                    parameters_,
                    ptr_,
                    flatten_shape(shape_),
                    flatten_strides(strides_),
                    byteoffset_,
                    itemsize_,
                    format_);
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();

    Index64 flathead = array.ravel();
    struct Error err = kernel::regularize_arrayslice_64(
      flathead.ptr().get(),
      flathead.length(),
      shape_[1]);
    util::handle_error(err, classname(), identities_.get());

    if (advanced.length() == 0) {
      Index64 nextcarry(carry.length()*flathead.length());
      Index64 nextadvanced(carry.length()*flathead.length());
      struct Error err = kernel::NumpyArray_getitem_next_array_64(
        nextcarry.ptr().get(),
        nextadvanced.ptr().get(),
        carry.ptr().get(),
        flathead.ptr().get(),
        carry.length(),
        flathead.length(),
        shape_[1]);   // because this is contiguous
      util::handle_error(err, classname(), identities_.get());

      NumpyArray out = next.getitem_next(nexthead,
                                         nexttail,
                                         nextcarry,
                                         nextadvanced,
                                         length*flathead.length(),
                                         next.strides_[0],
                                         false);

      std::vector<ssize_t> outshape = { (ssize_t)length };
      std::vector<int64_t> arrayshape = array.shape();
      for (auto x = arrayshape.begin();  x != arrayshape.end();  ++x) {
        outshape.push_back((ssize_t)(*x));
      }
      outshape.insert(outshape.end(),
                      out.shape_.begin() + 1,
                      out.shape_.end());

      std::vector<ssize_t> outstrides(out.strides_.begin(),
                                      out.strides_.end());
      for (auto x = arrayshape.rbegin();  x != arrayshape.rend();  ++x) {
        outstrides.insert(outstrides.begin(), ((ssize_t)(*x))*outstrides[0]);
      }
      return NumpyArray(arrayshape.size() == 1 ? out.identities_
                                               : Identities::none(),
                        out.parameters_,
                        out.ptr_,
                        outshape,
                        outstrides,
                        out.byteoffset_,
                        itemsize_,
                        format_);
    }

    else {
      Index64 nextcarry(carry.length());
      struct Error err = kernel::NumpyArray_getitem_next_array_advanced_64(
        nextcarry.ptr().get(),
        carry.ptr().get(),
        advanced.ptr().get(),
        flathead.ptr().get(),
        carry.length(),
        shape_[1]);   // because this is contiguous
      util::handle_error(err, classname(), identities_.get());

      NumpyArray out = next.getitem_next(nexthead,
                                         nexttail,
                                         nextcarry,
                                         advanced,
                                         length*array.length(),
                                         next.strides_[0],
                                         false);

      std::vector<ssize_t> outshape = { (ssize_t)length };
      outshape.insert(outshape.end(),
                      out.shape_.begin() + 1,
                      out.shape_.end());
      return NumpyArray(out.identities_,
                        out.parameters_,
                        out.ptr_,
                        outshape,
                        out.strides_,
                        out.byteoffset_,
                        itemsize_,
                        format_);
    }
  }

  void
  NumpyArray::tojson_boolean(ToJson& builder,
                             bool include_beginendlist) const {
    if (ndim() == 0) {
      bool* array = reinterpret_cast<bool*>(byteptr());
      builder.boolean(array[0]);
    }
    else if (ndim() == 1) {
      bool* array = reinterpret_cast<bool*>(byteptr());
      int64_t stride = (int64_t)(strides_[0]);
      if (include_beginendlist) {
        builder.beginlist();
      }
      for (int64_t i = 0;  i < length();  i++) {
        builder.boolean(array[i*stride]);
      }
      if (include_beginendlist) {
        builder.endlist();
      }
    }
    else {
      const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
      const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)i);
        NumpyArray numpy(Identities::none(),
                         util::Parameters(),
                         ptr_,
                         shape,
                         strides,
                         byteoffset,
                         itemsize_,
                         format_);
        numpy.tojson_boolean(builder, true);
      }
      builder.endlist();
    }
  }

  template <typename T>
  void
  NumpyArray::tojson_integer(ToJson& builder,
                             bool include_beginendlist) const {
    if (ndim() == 0) {
      T* array = reinterpret_cast<T*>(byteptr());
      builder.integer(array[0]);
    }
    else if (ndim() == 1) {
      T* array = reinterpret_cast<T*>(byteptr());
      int64_t stride = (int64_t)(strides_[0] / sizeof(T));
      if (include_beginendlist) {
        builder.beginlist();
      }
      for (int64_t i = 0;  i < length();  i++) {
        builder.integer(array[i*stride]);
      }
      if (include_beginendlist) {
        builder.endlist();
      }
    }
    else {
      const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
      const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)i);
        NumpyArray numpy(Identities::none(),
                         util::Parameters(),
                         ptr_,
                         shape,
                         strides,
                         byteoffset,
                         itemsize_,
                         format_);
        numpy.tojson_integer<T>(builder, true);
      }
      builder.endlist();
    }
  }

  template <typename T>
  void
  NumpyArray::tojson_real(ToJson& builder,
                          bool include_beginendlist) const {
    if (ndim() == 0) {
      T* array = reinterpret_cast<T*>(byteptr());
      builder.real(array[0]);
    }
    else if (ndim() == 1) {
      T* array = reinterpret_cast<T*>(byteptr());
      int64_t stride = (int64_t)(strides_[0] / sizeof(T));
      if (include_beginendlist) {
        builder.beginlist();
      }
      for (int64_t i = 0;  i < length();  i++) {
        builder.real(array[i*stride]);
      }
      if (include_beginendlist) {
        builder.endlist();
      }
    }
    else {
      const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
      const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
      builder.beginlist();
      for (int64_t i = 0;  i < length();  i++) {
        ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)i);
        NumpyArray numpy(Identities::none(),
                         util::Parameters(),
                         ptr_,
                         shape,
                         strides,
                         byteoffset,
                         itemsize_,
                         format_);
        numpy.tojson_real<T>(builder, true);
      }
      builder.endlist();
    }
  }

  void
  NumpyArray::tojson_string(ToJson& builder,
                            bool include_beginendlist) const {
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
        NumpyArray numpy(Identities::none(),
                         util::Parameters(),
                         ptr_,
                         shape,
                         strides,
                         byteoffset,
                         itemsize_,
                         format_);
        numpy.tojson_string(builder, true);
      }
      builder.endlist();
    }
  }

  const ContentPtr
  NumpyArray::copy_to(kernel::Lib ptr_lib) const {
    if(ptr_lib_ == ptr_lib) {
      return std::make_shared<NumpyArray>(identities(),
                                          parameters(),
                                          ptr_,
                                          shape(),
                                          strides(),
                                          byteoffset(),
                                          itemsize(),
                                          format(),
                                          ptr_lib_);
    }

    ssize_t length = 1;
    for (auto i : shape_) {
      length = length * i;
    }

    std::shared_ptr<void> ptr;
    if (format_.compare("?") == 0) {
      ptr = kernel::ptr_alloc<bool>(ptr_lib, length);

      Error err = kernel::copy_to<bool>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<bool *>(ptr.get()),
        reinterpret_cast<bool *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("b") == 0) {
      ptr = kernel::ptr_alloc<int8_t>(ptr_lib, length);

      Error err = kernel::copy_to<int8_t>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<int8_t *>(ptr.get()),
        reinterpret_cast<int8_t *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("B") == 0) {
      ptr = kernel::ptr_alloc<uint8_t>(ptr_lib, length);

      Error err = kernel::copy_to<uint8_t>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<uint8_t *>(ptr.get()),
        reinterpret_cast<uint8_t *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("h") == 0) {
      ptr = kernel::ptr_alloc<int16_t>(ptr_lib, length);

      Error err = kernel::copy_to<int16_t>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<int16_t *>(ptr.get()),
        reinterpret_cast<int16_t *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("H") == 0) {
      ptr = kernel::ptr_alloc<uint16_t>(ptr_lib, length);

      Error err = kernel::copy_to<uint16_t>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<uint16_t *>(ptr.get()),
        reinterpret_cast<uint16_t *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("i") == 0) {
      ptr = kernel::ptr_alloc<int32_t>(ptr_lib, length);

      Error err = kernel::copy_to<int32_t>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<int32_t *>(ptr.get()),
        reinterpret_cast<int32_t *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("I") == 0) {
      ptr = kernel::ptr_alloc<uint32_t>(ptr_lib, length);

      Error err = kernel::copy_to<uint32_t>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<uint32_t *>(ptr.get()),
        reinterpret_cast<uint32_t *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("l") == 0) {
      ptr = kernel::ptr_alloc<int64_t>(ptr_lib, length);

      Error err = kernel::copy_to<int64_t>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<int64_t *>(ptr.get()),
        reinterpret_cast<int64_t *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("L") == 0) {
      ptr = kernel::ptr_alloc<uint64_t>(ptr_lib, length);

      Error err = kernel::copy_to<uint64_t>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<uint64_t *>(ptr.get()),
        reinterpret_cast<uint64_t *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("f") == 0) {
      ptr = kernel::ptr_alloc<float>(ptr_lib, length);

      Error err = kernel::copy_to<float>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<float *>(ptr.get()),
        reinterpret_cast<float *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else if (format_.compare("d") == 0) {
      ptr = kernel::ptr_alloc<double>(ptr_lib, length);

      Error err = kernel::copy_to<double>(
        ptr_lib,
        ptr_lib_,
        reinterpret_cast<double *>(ptr.get()),
        reinterpret_cast<double *>(ptr_.get()),
        length);
      util::handle_error(err);
    }
    else {
      Error err = failure("Unknown Numpy dtype for transfer",
                          0,
                          kSliceNone,
                          true);
      util::handle_error(err);
    }

    return std::make_shared<NumpyArray>(identities(),
                                        parameters(),
                                        ptr,
                                        shape(),
                                        strides(),
                                        byteoffset(),
                                        itemsize(),
                                        format(),
                                        ptr_lib);
  }

  template<typename T>
  const std::shared_ptr<void>
  NumpyArray::index_sort(const T* data,
                         int64_t length,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength,
                         bool ascending,
                         bool stable) const {
    std::shared_ptr<int64_t> ptr(
      new int64_t[length], kernel::array_deleter<int64_t>());

    if (length == 0) {
      return ptr;
    }

    int64_t ranges_length = 0;
    struct Error err1 = kernel::sorting_ranges_length(
      &ranges_length,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err1, classname(), nullptr);

    Index64 outranges(ranges_length);
    struct Error err2 = kernel::sorting_ranges(
      outranges.ptr().get(),
      ranges_length,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err2, classname(), nullptr);

    struct Error err3 = kernel::NumpyArray_argsort<T>(
      ptr.get(),
      data,
      length,
      outranges.ptr().get(),
      ranges_length,
      ascending,
      stable);
    util::handle_error(err3, classname(), nullptr);

    return ptr;
  }

  template<typename T>
  const std::shared_ptr<void>
  NumpyArray::array_sort(const T* data,
                         int64_t length,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength,
                         bool ascending,
                         bool stable) const {
    std::shared_ptr<T> ptr(
      new T[length], kernel::array_deleter<T>());

    if (length == 0) {
      return ptr;
    }

    int64_t ranges_length = 0;
    struct Error err1 = kernel::sorting_ranges_length(
      &ranges_length,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err1, classname(), nullptr);

    Index64 outranges(ranges_length);
    struct Error err2 = kernel::sorting_ranges(
      outranges.ptr().get(),
      ranges_length,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err2, classname(), nullptr);

    struct Error err3 = kernel::NumpyArray_sort<T>(
      ptr.get(),
      data,
      length,
      outranges.ptr().get(),
      ranges_length,
      parents.length(),
      ascending,
      stable);
    util::handle_error(err3, classname(), nullptr);

    return ptr;
  }

  template<typename T>
  const std::shared_ptr<void>
  NumpyArray::string_sort(const T* data,
                          int64_t length,
                          const Index64& offsets,
                          Index64& outoffsets,
                          bool ascending,
                          bool stable) const {
    std::shared_ptr<T> ptr(
      new T[length], kernel::array_deleter<T>());

    if (length == 0) {
      return ptr;
    }

    struct Error err = kernel::NumpyArray_sort_asstrings(
      ptr.get(),
      data,
      length,
      offsets.ptr().get(),
      offsets.length(),
      outoffsets.ptr().get(),
      ascending,
      stable);
    util::handle_error(err, classname(), nullptr);

    return ptr;
  }
}
