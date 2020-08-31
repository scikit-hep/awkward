// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/NumpyArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/NumpyArray.cpp", line)

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "awkward/kernels/identities.h"
#include "awkward/kernels/getitem.h"
#include "awkward/kernels/operations.h"
#include "awkward/kernels/reducers.h"
#include "awkward/kernels/sorting.h"
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
                       const FormKey& form_key,
                       const std::vector<int64_t>& inner_shape,
                       int64_t itemsize,
                       const std::string& format,
                       util::dtype dtype)
      : Form(has_identities, parameters, form_key)
      , inner_shape_(inner_shape)
      , itemsize_(itemsize)
      , format_(format)
      , dtype_(dtype) { }

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

  util::dtype
  NumpyForm::dtype() const {
    return dtype_;
  }

  const std::string
  NumpyForm::primitive() const {
    return util::dtype_to_name(dtype_);
  }

  const TypePtr
  NumpyForm::type(const util::TypeStrs& typestrs) const {
    TypePtr out;
    if (dtype_ == util::dtype::NOT_PRIMITIVE) {
      throw std::invalid_argument(
        std::string("Numpy format \"") + format_
        + std::string("\" cannot be expressed as a PrimitiveType")
        + FILENAME(__LINE__));
    }
    else {
      out = std::make_shared<PrimitiveType>(
                 parameters_,
                 util::gettypestr(parameters_, typestrs),
                 dtype_);
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
    if (verbose  ||
        toplevel  ||
        p.empty()  ||
        !inner_shape_.empty() ||
        has_identities_  ||
        !parameters_.empty()  ||
        form_key_.get() != nullptr) {
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
      form_key_tojson(builder, verbose);
      builder.endrecord();
    }
    else {
      builder.string(p.c_str(), (int64_t)p.length());
    }
  }

  const FormPtr
  NumpyForm::shallow_copy() const {
    return std::make_shared<NumpyForm>(has_identities_,
                                       parameters_,
                                       form_key_,
                                       inner_shape_,
                                       itemsize_,
                                       format_,
                                       dtype_);
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
      std::string("key ") + util::quote(key)
      + std::string(" does not exist (data are not records)")
      + FILENAME(__LINE__));
  }

  const std::string
  NumpyForm::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("fieldindex \"") + std::to_string(fieldindex)
      + std::string("\" does not exist (data are not records)")
      + FILENAME(__LINE__));
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
                   bool check_form_key,
                   bool compatibility_check) const {
    if (check_identities  &&
        has_identities_ != other.get()->has_identities()) {
      return false;
    }
    if (check_parameters  &&
        !util::parameters_equal(parameters_, other.get()->parameters())) {
      return false;
    }
    if (check_form_key  &&
        !form_key_equals(other.get()->form_key())) {
      return false;
    }
    if (NumpyForm* t = dynamic_cast<NumpyForm*>(other.get())) {
      return (inner_shape_ == t->inner_shape()  &&  format_ == t->format());
    }
    else {
      return false;
    }
  }

  const FormPtr
  NumpyForm::getitem_field(const std::string& key) const {
    throw std::invalid_argument(
      std::string("key ") + util::quote(key)
      + std::string(" does not exist (data are not records)"));
  }

  ////////// NumpyArray

  NumpyArray::NumpyArray(const IdentitiesPtr& identities,
                         const util::Parameters& parameters,
                         const std::shared_ptr<void>& ptr,
                         const std::vector<ssize_t>& shape,
                         const std::vector<ssize_t>& strides,
                         ssize_t byteoffset,
                         ssize_t itemsize,
                         const std::string format,
                         util::dtype dtype,
                         const kernel::lib ptr_lib)
      : Content(identities, parameters)
      , ptr_(ptr)
      , ptr_lib_(ptr_lib)
      , shape_(shape)
      , strides_(strides)
      , byteoffset_(byteoffset)
      , itemsize_(itemsize)
      , format_(format)
      , dtype_(dtype) {
    if (shape.size() != strides.size()) {
      throw std::invalid_argument(
        std::string("len(shape), which is ") + std::to_string(shape.size())
        + std::string(", must be equal to len(strides), which is ")
        + std::to_string(strides.size()) + FILENAME(__LINE__));
    }
  }

  NumpyArray::NumpyArray(const Index8 index)
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(int8_t) }),
                 index.offset() * (ssize_t)sizeof(int8_t),
                 (ssize_t)sizeof(int8_t),
                 util::dtype_to_format(util::dtype::int8),
                 util::dtype::int8,
                 index.ptr_lib()) { }

  NumpyArray::NumpyArray(const IndexU8 index)
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(uint8_t) }),
                 index.offset() * (ssize_t)sizeof(uint8_t),
                 (ssize_t)sizeof(uint8_t),
                 util::dtype_to_format(util::dtype::uint8),
                 util::dtype::uint8,
                 index.ptr_lib()) { }

  NumpyArray::NumpyArray(const Index32 index)
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(int32_t) }),
                 index.offset() * (ssize_t)sizeof(int32_t),
                 (ssize_t)sizeof(int32_t),
                 util::dtype_to_format(util::dtype::int32),
                 util::dtype::int32,
                 index.ptr_lib()) { }

  NumpyArray::NumpyArray(const IndexU32 index)
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(uint32_t) }),
                 index.offset() * (ssize_t)sizeof(uint32_t),
                 (ssize_t)sizeof(uint32_t),
                 util::dtype_to_format(util::dtype::uint32),
                 util::dtype::uint32,
                 index.ptr_lib()) { }

  NumpyArray::NumpyArray(const Index64 index)
    : NumpyArray(Identities::none(),
                 util::Parameters(),
                 index.ptr(),
                 std::vector<ssize_t>({ (ssize_t)index.length() }),
                 std::vector<ssize_t>({ (ssize_t)sizeof(int64_t) }),
                 index.offset() * (ssize_t)sizeof(int64_t),
                 (ssize_t)sizeof(int64_t),
                 util::dtype_to_format(util::dtype::int64),
                 util::dtype::int64,
                 index.ptr_lib()) { }

  const std::shared_ptr<void>
  NumpyArray::ptr() const {
    return ptr_;
  }

  void*
  NumpyArray::data() const {
    return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr_.get()) +
                                   byteoffset_);
  }

  kernel::lib
  NumpyArray::ptr_lib() const {
    return ptr_lib_;
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

  util::dtype
  NumpyArray::dtype() const {
    return dtype_;
  }

  ssize_t
  NumpyArray::ndim() const {
    return (ssize_t)shape_.size();
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

  ssize_t
  NumpyArray::bytelength() const {
    if (isscalar()) {
      return itemsize_;
    }
    else {
      ssize_t out = itemsize_;
      for (size_t i = 0;  i < shape_.size();  i++) {
        out += (shape_[i] - 1)*strides_[i];
      }
      return out;
    }
  }

  uint8_t
  NumpyArray::getbyte(ssize_t at) const {
    return kernel::NumpyArray_getitem_at0(ptr_lib(),
                                          reinterpret_cast<uint8_t*>(ptr_.get()) + byteoffset_ + at*strides_[0]);
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
      contiguous_self.format(),
      contiguous_self.dtype(),
      ptr_lib_);
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
                kSliceNone,
                FILENAME_C(__LINE__)),
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
      struct Error err = kernel::new_Identities<int32_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->ptr().get(),
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
      struct Error err = kernel::new_Identities<int64_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->ptr().get(),
        length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  template <typename T>
  void tostring_as(kernel::lib ptr_lib,
                   std::stringstream& out,
                   T* ptr,
                   ssize_t stride,
                   int64_t length,
                   util::dtype dtype) {
    if (length <= 10) {
      for (int64_t i = 0;  i < length;  i++) {
        T* ptr2 = reinterpret_cast<T*>(
            reinterpret_cast<ssize_t>(ptr) + stride*((ssize_t)i));
        if (i != 0) {
          out << " ";
        }
        if (dtype == util::dtype::boolean) {
          out << (kernel::NumpyArray_getitem_at0(ptr_lib, ptr2) != 0 ? "true" : "false");
        }
        else if (dtype == util::dtype::int8) {
          out << (int)kernel::NumpyArray_getitem_at0(ptr_lib, ptr2);
        }
        else if (dtype == util::dtype::uint8) {
          out << (unsigned int)kernel::NumpyArray_getitem_at0(ptr_lib, ptr2);
        }
        else {
          out << kernel::NumpyArray_getitem_at0(ptr_lib, ptr2);
        }
      }
    }
    else {
      for (int64_t i = 0;  i < 5;  i++) {
        T* ptr2 = reinterpret_cast<T*>(
            reinterpret_cast<ssize_t>(ptr) + stride*((ssize_t)i));
        if (i != 0) {
          out << " ";
        }
        if (dtype == util::dtype::boolean) {
          out << (kernel::NumpyArray_getitem_at0(ptr_lib, ptr2) != 0 ? "true" : "false");
        }
        else if (dtype == util::dtype::int8) {
          out << (int)kernel::NumpyArray_getitem_at0(ptr_lib, ptr2);
        }
        else if (dtype == util::dtype::uint8) {
          out << (unsigned int)kernel::NumpyArray_getitem_at0(ptr_lib, ptr2);
        }
        else {
          out << kernel::NumpyArray_getitem_at0(ptr_lib, ptr2);
        }
      }
      out << " ... ";
      for (int64_t i = length - 5;  i < length;  i++) {
        T* ptr2 = reinterpret_cast<T*>(
            reinterpret_cast<ssize_t>(ptr) + stride*((ssize_t)i));
        if (i != length - 5) {
          out << " ";
        }
        if (dtype == util::dtype::boolean) {
          out << (kernel::NumpyArray_getitem_at0(ptr_lib, ptr2) != 0 ? "true" : "false");
        }
        else if (dtype == util::dtype::int8) {
          out << (int)kernel::NumpyArray_getitem_at0(ptr_lib, ptr2);
        }
        else if (dtype == util::dtype::uint8) {
          out << (unsigned int)kernel::NumpyArray_getitem_at0(ptr_lib, ptr2);
        }
        else {
          out << kernel::NumpyArray_getitem_at0(ptr_lib, ptr2);
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
    std::vector<int64_t> inner_shape(std::next(shape_.begin()), shape_.end());
    return std::make_shared<NumpyForm>(identities_.get() != nullptr,
                                       parameters_,
                                       FormKey(nullptr),
                                       inner_shape,
                                       (int64_t)itemsize_,
                                       format_,
                                       dtype_);
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
        << util::quote(format_) << " shape=\"";
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
          out << " ";
        }
        out << strides_[i];
      }
      out << "\" ";
    }
    out << "data=\"";
    if (ndim() == 1  &&  dtype_ == util::dtype::boolean) {
      tostring_as<bool>(ptr_lib(),
                        out,
                        reinterpret_cast<bool*>(data()),
                        strides_[0],
                        length(),
                        dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::int8) {
      tostring_as<int8_t>(ptr_lib(),
                          out,
                          reinterpret_cast<int8_t*>(data()),
                          strides_[0],
                          length(),
                          dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::int16) {
      tostring_as<int16_t>(ptr_lib(),
                           out,
                           reinterpret_cast<int16_t*>(data()),
                           strides_[0],
                           length(),
                           dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::int32) {
      tostring_as<int32_t>(ptr_lib(),
                           out,
                           reinterpret_cast<int32_t*>(data()),
                           strides_[0],
                           length(),
                           dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::int64) {
      tostring_as<int64_t>(ptr_lib(),
                           out,
                           reinterpret_cast<int64_t*>(data()),
                           strides_[0],
                           length(),
                           dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::uint8) {
      tostring_as<uint8_t>(ptr_lib(),
                           out,
                           reinterpret_cast<uint8_t*>(data()),
                           strides_[0],
                           length(),
                           dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::uint16) {
      tostring_as<uint16_t>(ptr_lib(),
                            out,
                            reinterpret_cast<uint16_t*>(data()),
                            strides_[0],
                            length(),
                            dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::uint32) {
      tostring_as<uint32_t>(ptr_lib(),
                            out,
                            reinterpret_cast<uint32_t*>(data()),
                            strides_[0],
                            length(),
                            dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::uint64) {
      tostring_as<uint64_t>(ptr_lib(),
                            out,
                            reinterpret_cast<uint64_t*>(data()),
                            strides_[0],
                            length(),
                            dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::float32) {
      tostring_as<float>(ptr_lib(),
                         out,
                         reinterpret_cast<float*>(data()),
                         strides_[0],
                         length(),
                         dtype_);
    }
    else if (ndim() == 1  &&  dtype_ == util::dtype::float64) {
      tostring_as<double>(ptr_lib(),
                          out,
                          reinterpret_cast<double*>(data()),
                          strides_[0],
                          length(),
                          dtype_);
    }
    else {
      out << "0x ";
      ssize_t len = bytelength();
      if (len <= 32) {
        for (ssize_t i = 0;  i < len;  i++) {
          if (i != 0  &&  i % 4 == 0) {
            out << " ";
          }
          unsigned char* ptr2 = reinterpret_cast<unsigned char*>(
              reinterpret_cast<ssize_t>(data()) + (ssize_t)i);
          out << std::hex << std::setw(2) << std::setfill('0')
              << int((unsigned char)kernel::NumpyArray_getitem_at0(ptr_lib_, ptr2));
        }
      }
      else {
        for (ssize_t i = 0;  i < 16;  i++) {
          if (i != 0  &&  i % 4 == 0) {
            out << " ";
          }
          unsigned char* ptr2 = reinterpret_cast<unsigned char*>(
              reinterpret_cast<ssize_t>(data()) + (ssize_t)i);
          out << std::hex << std::setw(2) << std::setfill('0')
              << int((unsigned char)kernel::NumpyArray_getitem_at0(ptr_lib_, ptr2));
        }
        out << " ... ";
        for (ssize_t i = len - 16;  i < len;  i++) {
          if (i != len - 16  &&  i % 4 == 0) {
            out << " ";
          }
          unsigned char* ptr2 = reinterpret_cast<unsigned char*>(
              reinterpret_cast<ssize_t>(data()) + (ssize_t)i);
          out << std::hex << std::setw(2) << std::setfill('0')
              << int((unsigned char)kernel::NumpyArray_getitem_at0(ptr_lib_, ptr2));
        }
      }
    }
    out << "\" at=\"0x";
    out << std::hex << std::setw(12) << std::setfill('0')
        << reinterpret_cast<ssize_t>(ptr_.get());
    if (ptr_lib() == kernel::lib::cuda) {
      out << "\">\n";
      out << kernel::lib_tostring(ptr_lib_,
                                  ptr_.get(),
                                  indent + std::string("    "),
                                  "",
                                  "\n");

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
    else {
      switch (dtype_) {
        case util::dtype::boolean:
          tojson_boolean(builder, include_beginendlist);
          break;
        case util::dtype::int8:
          tojson_integer<int8_t>(builder, include_beginendlist);
          break;
        case util::dtype::int16:
          tojson_integer<int16_t>(builder, include_beginendlist);
          break;
        case util::dtype::int32:
          tojson_integer<int32_t>(builder, include_beginendlist);
          break;
        case util::dtype::int64:
          tojson_integer<int64_t>(builder, include_beginendlist);
          break;
        case util::dtype::uint8:
          tojson_integer<uint8_t>(builder, include_beginendlist);
          break;
        case util::dtype::uint16:
          tojson_integer<uint16_t>(builder, include_beginendlist);
          break;
        case util::dtype::uint32:
          tojson_integer<uint32_t>(builder, include_beginendlist);
          break;
        case util::dtype::uint64:
          tojson_integer<uint64_t>(builder, include_beginendlist);
          break;
        case util::dtype::float16:
          throw std::runtime_error(
            std::string("FIXME: float16 to JSON") + FILENAME(__LINE__));
        case util::dtype::float32:
          tojson_real<float>(builder, include_beginendlist);
          break;
        case util::dtype::float64:
          tojson_real<double>(builder, include_beginendlist);
          break;
        case util::dtype::float128:
          throw std::runtime_error(
            std::string("FIXME: float128 to JSON") + FILENAME(__LINE__));
        case util::dtype::complex64:
          throw std::runtime_error(
            std::string("FIXME: complex64 to JSON") + FILENAME(__LINE__));
        case util::dtype::complex128:
          throw std::runtime_error(
            std::string("FIXME: complex128 to JSON") + FILENAME(__LINE__));
        case util::dtype::complex256:
          throw std::runtime_error(
            std::string("FIXME: complex256 to JSON") + FILENAME(__LINE__));
        default:
          throw std::invalid_argument(
            std::string("cannot convert Numpy format \"") + format_
            + std::string("\" into JSON") + FILENAME(__LINE__));
      }
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
                                        format_,
                                        dtype_,
                                        ptr_lib_);
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
                                        format_,
                                        dtype_,
                                        ptr_lib_);
  }

  void
  NumpyArray::check_for_iteration() const {
    if (identities_.get() != nullptr  &&
        identities_.get()->length() < shape_[0]) {
      util::handle_error(
        failure("len(identities) < len(array)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
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
                                        format_,
                                        dtype_,
                                        ptr_lib_);
  }

  const ContentPtr
  NumpyArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += shape_[0];
    }
    if (regular_at < 0  ||  regular_at >= shape_[0]) {
      util::handle_error(
        failure("index out of range", kSliceNone, at, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const ContentPtr
  NumpyArray::getitem_at_nowrap(int64_t at) const {
    ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)at);
    const std::vector<ssize_t> shape(std::next(shape_.begin()), shape_.end());
    const std::vector<ssize_t> strides(std::next(strides_.begin()), strides_.end());
    IdentitiesPtr identities;
    if (identities_.get() != nullptr) {
      if (at >= identities_.get()->length()) {
        util::handle_error(
          failure("index out of range", kSliceNone, at, FILENAME_C(__LINE__)),
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
                                        format_,
                                        dtype_,
                                        ptr_lib_);
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
    shape.emplace_back((ssize_t)(stop - start));
    shape.insert(shape.end(), std::next(shape_.begin()), shape_.end());
    IdentitiesPtr identities;
    if (identities_.get() != nullptr) {
      if (stop > identities_.get()->length()) {
        util::handle_error(
          failure("index out of range", kSliceNone, stop, FILENAME_C(__LINE__)),
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
                                        format_,
                                        dtype_,
                                        ptr_lib_);
  }

  const ContentPtr
  NumpyArray::getitem_field(const std::string& key) const {
    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by field name") + FILENAME(__LINE__));
  }

  const ContentPtr
  NumpyArray::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by field names") + FILENAME(__LINE__));
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
      throw std::runtime_error(
        std::string("cannot get-item on a scalar") + FILENAME(__LINE__));
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
                      format_,
                      dtype_,
                      ptr_lib_);

      SliceItemPtr nexthead = where.head();
      Slice nexttail = where.tail();
      NumpyArray out = next.getitem_bystrides(nexthead, nexttail, 1);

      std::vector<ssize_t> outshape(std::next(out.shape_.begin()), out.shape_.end());
      std::vector<ssize_t> outstrides(std::next(out.strides_.begin()),
                                      out.strides_.end());
      return std::make_shared<NumpyArray>(out.identities_,
                                          out.parameters_,
                                          out.ptr_,
                                          outshape,
                                          outstrides,
                                          out.byteoffset_,
                                          itemsize_,
                                          format_,
                                          dtype_,
                                          ptr_lib_);
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
                      format_,
                      dtype_,
                      ptr_lib_);

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

      std::vector<ssize_t> outshape(std::next(out.shape_.begin()), out.shape_.end());
      std::vector<ssize_t> outstrides(std::next(out.strides_.begin()),
                                      out.strides_.end());
      return std::make_shared<NumpyArray>(out.identities_,
                                          out.parameters_,
                                          out.ptr_,
                                          outshape,
                                          outstrides,
                                          out.byteoffset_,
                                          itemsize_,
                                          format_,
                                          dtype_,
                                          ptr_lib_);
    }
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceItemPtr& head,
                           const Slice& tail,
                           const Index64& advanced) const {
    Index64 carry(shape_[0]);
    struct Error err = kernel::carry_arange<int64_t>(
      kernel::lib::cpu,   // DERIVE
      carry.data(),
      shape_[0]);
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
    if (!iscontiguous()) {
      return contiguous().carry(carry, allow_lazy);
    }

    std::shared_ptr<void> ptr(
      kernel::malloc<void>(ptr_lib_, carry.length()*((int64_t)strides_[0])));
    struct Error err = kernel::NumpyArray_getitem_next_null_64(
      kernel::lib::cpu,   // DERIVE
      reinterpret_cast<uint8_t*>(ptr.get()),
      reinterpret_cast<uint8_t*>(data()),
      carry.length(),
      strides_[0],
      carry.ptr().get());
    util::handle_error(err, classname(), identities_.get());

    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }

    std::vector<ssize_t> shape = { (ssize_t)carry.length() };
    shape.insert(shape.end(), std::next(shape_.begin()), shape_.end());
    return std::make_shared<NumpyArray>(identities,
                                        parameters_,
                                        ptr,
                                        shape,
                                        strides_,
                                        0,
                                        itemsize_,
                                        format_,
                                        dtype_,
                                        ptr_lib_);
  }

  int64_t
  NumpyArray::numfields() const {
    return -1;
  }

  int64_t
  NumpyArray::fieldindex(const std::string& key) const {
    throw std::invalid_argument(
      std::string("key ") + util::quote(key)
      + std::string(" does not exist (data are not records)")
      + FILENAME(__LINE__));
  }

  const std::string
  NumpyArray::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("fieldindex \"") + std::to_string(fieldindex)
      + std::string("\" does not exist (data are not records)")
      + FILENAME(__LINE__));
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
              + std::string("): shape is zero-dimensional")
              + FILENAME(__LINE__));
    }
    for (size_t i = 0;  i < shape_.size();  i++) {
      if (shape_[i] < 0) {
        return (std::string("at ") + path + std::string(" (") + classname()
                + std::string("): shape[") + std::to_string(i) + ("] < 0")
                + FILENAME(__LINE__));
      }
    }
    for (size_t i = 0;  i < strides_.size();  i++) {
      if (strides_[i] % itemsize_ != 0) {
        return (std::string("at ") + path + std::string(" (") + classname()
                + std::string("): shape[") + std::to_string(i)
                + ("] % itemsize != 0")
                + FILENAME(__LINE__));
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
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      Index64 out(1);
      out.setitem_at_nowrap(0, length());
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    std::vector<ssize_t> shape;
    int64_t reps = 1;
    int64_t size = length();
    int64_t i = 0;
    while (i < ndim() - 1  &&  depth < posaxis) {
      shape.emplace_back(shape_[(size_t)i]);
      reps *= shape_[(size_t)i];
      size = shape_[(size_t)i + 1];
      i++;
      depth++;
    }
    if (posaxis > depth) {
      throw std::invalid_argument(
        std::string("'axis' out of range for 'num'") + FILENAME(__LINE__));
    }

    ssize_t x = sizeof(int64_t);
    std::vector<ssize_t> strides;
    for (int64_t j = (int64_t)shape.size();  j > 0;  j--) {
      strides.insert(strides.begin(), x);
      x *= shape[(size_t)(j - 1)];
    }

    Index64 tonum(reps, ptr_lib());

    struct Error err = kernel::RegularArray_num_64(
      ptr_lib(),
      tonum.data(),
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
      util::dtype_to_format(util::dtype::int64),
      util::dtype::int64,
      ptr_lib());
  }

  const std::vector<ssize_t>
  flatten_shape(const std::vector<ssize_t> shape) {
    if (shape.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      std::vector<ssize_t> out = { shape[0]*shape[1] };
      out.insert(out.end(), std::next(shape.begin(), 2), shape.end());
      return out;
    }
  }

  const std::vector<ssize_t>
  flatten_strides(const std::vector<ssize_t> strides) {
    if (strides.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      return std::vector<ssize_t>(std::next(strides.begin()), strides.end());
    }
  }

  const std::pair<Index64, ContentPtr>
  NumpyArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      throw std::invalid_argument(
        std::string("axis=0 not allowed for flatten") + FILENAME(__LINE__));
    }
    else if (shape_.size() != 1  ||  !iscontiguous()) {
      return toRegularArray().get()->offsets_and_flattened(posaxis, depth);
    }
    else {
      throw std::invalid_argument(
        std::string("axis out of range for flatten") + FILENAME(__LINE__));
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

      if (!mergebool  &&
          dtype_ != rawother->dtype()  &&
          (dtype_ == util::dtype::boolean  ||  rawother->dtype() == util::dtype::boolean)) {
        return false;
      }

      // if (dtype_ != rawother->dtype()  &&
      //     (dtype_ == util::dtype::datetime64  ||  rawother->dtype() == util::dtype::datetime64)) {
      //   return false;
      // }

      // if (dtype_ != rawother->dtype()  &&
      //     (dtype_ == util::dtype::timediff64  ||  rawother->dtype() == util::dtype::timediff64)) {
      //   return false;
      // }

      if (!(dtype_ == util::dtype::boolean  ||
            dtype_ == util::dtype::int8  ||
            dtype_ == util::dtype::int16  ||
            dtype_ == util::dtype::int32  ||
            dtype_ == util::dtype::int64  ||
            dtype_ == util::dtype::uint8  ||
            dtype_ == util::dtype::uint16  ||
            dtype_ == util::dtype::uint32  ||
            dtype_ == util::dtype::uint64  ||
            dtype_ == util::dtype::float16  ||
            dtype_ == util::dtype::float32  ||
            dtype_ == util::dtype::float64  ||
            dtype_ == util::dtype::float128  ||
            dtype_ == util::dtype::complex64  ||
            dtype_ == util::dtype::complex128  ||
            dtype_ == util::dtype::complex256  ||
            dtype_ == util::dtype::boolean  ||
            rawother->dtype() == util::dtype::int8  ||
            rawother->dtype() == util::dtype::int16  ||
            rawother->dtype() == util::dtype::int32  ||
            rawother->dtype() == util::dtype::int64  ||
            rawother->dtype() == util::dtype::uint8  ||
            rawother->dtype() == util::dtype::uint16  ||
            rawother->dtype() == util::dtype::uint32  ||
            rawother->dtype() == util::dtype::uint64  ||
            rawother->dtype() == util::dtype::float16  ||
            rawother->dtype() == util::dtype::float32  ||
            rawother->dtype() == util::dtype::float64  ||
            rawother->dtype() == util::dtype::float128  ||
            rawother->dtype() == util::dtype::complex64  ||
            rawother->dtype() == util::dtype::complex128  ||
            rawother->dtype() == util::dtype::complex256)) {
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
      throw std::invalid_argument(
        std::string("cannot merge Numpy scalars") + FILENAME(__LINE__));
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
      util::dtype dtype = util::dtype::NOT_PRIMITIVE;

      if (ndim() != rawother->ndim()) {
        throw std::invalid_argument(
          std::string("cannot merge arrays with different shapes")
          + FILENAME(__LINE__));
      }

      kernel::lib ptr_lib = ptr_lib_;   // DERIVE

      if (dtype_ == util::dtype::complex256  ||
          rawother->dtype() == util::dtype::complex256) {
        dtype = util::dtype::complex256;
      }
      else if ((dtype_ == util::dtype::float128  &&
                util::is_complex(rawother->dtype()))  ||
               (rawother->dtype() == util::dtype::float128  &&
                util::is_complex(dtype_))) {
        dtype = util::dtype::complex256;
      }
      else if (dtype_ == util::dtype::complex128  ||
               rawother->dtype() == util::dtype::complex128) {
        dtype = util::dtype::complex128;
      }
      else if (((dtype_ == util::dtype::float64  ||
                 dtype_ == util::dtype::uint64  ||
                 dtype_ == util::dtype::int64  ||
                 dtype_ == util::dtype::uint32  ||
                 dtype_ == util::dtype::int32)  &&
                util::is_complex(rawother->dtype()))  ||
               ((rawother->dtype() == util::dtype::float64  ||
                 rawother->dtype() == util::dtype::uint64  ||
                 rawother->dtype() == util::dtype::int64  ||
                 rawother->dtype() == util::dtype::uint32  ||
                 rawother->dtype() == util::dtype::int32)  &&
                util::is_complex(dtype_))) {
        dtype = util::dtype::complex128;
      }
      else if (dtype_ == util::dtype::complex64  ||
               rawother->dtype() == util::dtype::complex64) {
        dtype = util::dtype::complex64;
      }
      else if (dtype_ == util::dtype::float128  ||
               rawother->dtype() == util::dtype::float128) {
        dtype = util::dtype::float128;
      }
      else if (dtype_ == util::dtype::float64  ||
               rawother->dtype() == util::dtype::float64) {
        dtype = util::dtype::float64;
      }
      else if ((dtype_ == util::dtype::float32  &&
                (rawother->dtype() == util::dtype::uint64  ||
                 rawother->dtype() == util::dtype::int64  ||
                 rawother->dtype() == util::dtype::uint32  ||
                 rawother->dtype() == util::dtype::int32))  ||
               (rawother->dtype() == util::dtype::float32  &&
                (dtype_ == util::dtype::uint64  ||
                 dtype_ == util::dtype::int64  ||
                 dtype_ == util::dtype::uint32  ||
                 dtype_ == util::dtype::int32))) {
        dtype = util::dtype::float64;
      }
      else if (dtype_ == util::dtype::float32  ||
               rawother->dtype() == util::dtype::float32) {
        dtype = util::dtype::float32;
      }
      else if ((dtype_ == util::dtype::float16  &&
                (rawother->dtype() == util::dtype::uint64  ||
                 rawother->dtype() == util::dtype::int64  ||
                 rawother->dtype() == util::dtype::uint32  ||
                 rawother->dtype() == util::dtype::int32))  ||
               (rawother->dtype() == util::dtype::float16  &&
                (dtype_ == util::dtype::uint64  ||
                 dtype_ == util::dtype::int64  ||
                 dtype_ == util::dtype::uint32  ||
                 dtype_ == util::dtype::int32))) {
        dtype = util::dtype::float64;
      }
      else if ((dtype_ == util::dtype::float16  &&
                (rawother->dtype() == util::dtype::uint16  ||
                 rawother->dtype() == util::dtype::int16))  ||
               (rawother->dtype() == util::dtype::float16  &&
                (dtype_ == util::dtype::uint16  ||
                 dtype_ == util::dtype::int16))) {
        dtype = util::dtype::float32;
      }
      else if (dtype_ == util::dtype::float16  ||
               rawother->dtype() == util::dtype::float16) {
        dtype = util::dtype::float16;
      }
      else if ((dtype_ == util::dtype::uint64  &&
                util::is_signed(rawother->dtype()))  ||
               (rawother->dtype() == util::dtype::uint64  &&
                util::is_signed(dtype_))) {
        dtype = util::dtype::float64;
      }
      else if (dtype_ == util::dtype::uint64  ||
               rawother->dtype() == util::dtype::uint64) {
        dtype = util::dtype::uint64;
      }
      else if (dtype_ == util::dtype::int64  ||
               rawother->dtype() == util::dtype::int64) {
        dtype = util::dtype::int64;
      }
      else if ((dtype_ == util::dtype::uint32  &&
                util::is_signed(rawother->dtype()))  ||
               (rawother->dtype() == util::dtype::uint32  &&
                util::is_signed(dtype_))) {
        dtype = util::dtype::int64;
      }
      else if (dtype_ == util::dtype::uint32  ||
               rawother->dtype() == util::dtype::uint32) {
        dtype = util::dtype::uint32;
      }
      else if (dtype_ == util::dtype::int32  ||
               rawother->dtype() == util::dtype::int32) {
        dtype = util::dtype::int32;
      }
      else if ((dtype_ == util::dtype::uint16  &&
                util::is_signed(rawother->dtype()))  ||
               (rawother->dtype() == util::dtype::uint16  &&
                util::is_signed(dtype_))) {
        dtype = util::dtype::int32;
      }
      else if (dtype_ == util::dtype::uint16  ||
               rawother->dtype() == util::dtype::uint16) {
        dtype = util::dtype::uint16;
      }
      else if (dtype_ == util::dtype::int16  ||
               rawother->dtype() == util::dtype::int16) {
        dtype = util::dtype::int16;
      }
      else if ((dtype_ == util::dtype::uint8  &&
                util::is_signed(rawother->dtype()))  ||
               (rawother->dtype() == util::dtype::uint8  &&
                util::is_signed(dtype_))) {
        dtype = util::dtype::int16;
      }
      else if (dtype_ == util::dtype::uint8  ||
               rawother->dtype() == util::dtype::uint8) {
        dtype = util::dtype::uint8;
      }
      else if (dtype_ == util::dtype::int8  ||
               rawother->dtype() == util::dtype::int8) {
        dtype = util::dtype::int8;
      }
      else if (dtype_ == util::dtype::boolean  &&
               rawother->dtype() == util::dtype::boolean) {
        dtype = util::dtype::boolean;
      }
      // else if (dtype_ == util::dtype::datetime64  &&
      //          rawother->dtype() == util::dtype::datetime64) {
      //   dtype = util::dtype::datetime64;
      // }
      // else if (dtype_ == util::dtype::timedelta64  &&
      //          rawother->dtype() == util::dtype::timedelta64) {
      //   dtype = util::dtype::timedelta64;
      // }
      else {
        throw std::invalid_argument(
          std::string("cannot merge Numpy format \"") + format_
          + std::string("\" with \"") + rawother->format() + std::string("\"")
          + FILENAME(__LINE__));
      }

      int64_t itemsize = util::dtype_to_itemsize(dtype);

      std::vector<ssize_t> other_shape = rawother->shape();
      std::vector<ssize_t> shape;
      std::vector<ssize_t> strides;
      shape.emplace_back(shape_[0] + other_shape[0]);
      strides.emplace_back((ssize_t)itemsize);
      int64_t self_flatlength = shape_[0];
      int64_t other_flatlength = other_shape[0];
      for (int64_t i = ((int64_t)shape_.size()) - 1;  i > 0;  i--) {
        if (shape_[(size_t)i] != other_shape[(size_t)i]) {
          throw std::invalid_argument(
            std::string("cannot merge arrays with different shapes")
            + FILENAME(__LINE__));
        }
        shape.insert(std::next(shape.begin()), shape_[(size_t)i]);
        strides.insert(strides.begin(), strides[0]*shape_[(size_t)i]);
        self_flatlength *= (int64_t)shape_[(size_t)i];
        other_flatlength *= (int64_t)shape_[(size_t)i];
      }

      std::shared_ptr<void> ptr(
        kernel::malloc<void>(ptr_lib_, itemsize*(self_flatlength + other_flatlength)));

      NumpyArray contiguous_other = rawother->contiguous();

      struct Error err;

      switch (dtype) {
      // to boolean
      case util::dtype::boolean:
        err = kernel::NumpyArray_fill_frombool<bool>(
          kernel::lib::cpu,   // DERIVE
          reinterpret_cast<bool*>(ptr.get()),
          0,
          reinterpret_cast<bool*>(contiguous_self.data()),
          self_flatlength);
        util::handle_error(err, classname(), nullptr);
        err = kernel::NumpyArray_fill_frombool<bool>(
          kernel::lib::cpu,   // DERIVE
          reinterpret_cast<bool*>(ptr.get()),
          self_flatlength,
          reinterpret_cast<bool*>(contiguous_other.data()),
          other_flatlength);
        util::handle_error(err, classname(), nullptr);
        break;

      // // to datetime64
      // case util::dtype::datetime64:
      //   throw std::runtime_error(
      //     std::string("FIXME: merge to datetime64 not implemented")
      //     + FILENAME(__LINE__));
      //   break;

      // // to timedelta64
      // case util::dtype::timedelta64:
      //   throw std::runtime_error(
      //     std::string("FIXME: merge to timedelta64 not implemented")
      //     + FILENAME(__LINE__));
      //   break;

      // to int
      case util::dtype::int8:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<int8_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int8_t*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, int8_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int8_t*>(ptr.get()),
              0,
              reinterpret_cast<int8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8} (1)") + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<int8_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int8_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, int8_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int8_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8} (2)") + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to int
      case util::dtype::int16:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<int16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int16_t*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, int16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int16_t*>(ptr.get()),
              0,
              reinterpret_cast<int8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, int16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int16_t*>(ptr.get()),
              0,
              reinterpret_cast<int16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, int16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int16_t*>(ptr.get()),
              0,
              reinterpret_cast<uint8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, uint8} (1)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<int16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int16_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, int16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int16_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, int16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int16_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, int16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int16_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, uint8} (2)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to int32
      case util::dtype::int32:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              0,
              reinterpret_cast<int8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              0,
              reinterpret_cast<int16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int32:
            err = kernel::NumpyArray_fill<int32_t, int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              0,
              reinterpret_cast<int32_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              0,
              reinterpret_cast<uint8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              0,
              reinterpret_cast<uint16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, int32, uint8, uint16} (1)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int32:
            err = kernel::NumpyArray_fill<int32_t , int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int32_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, int32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, int32, uint8, uint16} (2)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to int64
      case util::dtype::int64:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              0,
              reinterpret_cast<int8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              0,
              reinterpret_cast<int16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int32:
            err = kernel::NumpyArray_fill<int32_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              0,
              reinterpret_cast<int32_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int64:
            err = kernel::NumpyArray_fill<int64_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              0,
              reinterpret_cast<int64_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              0,
              reinterpret_cast<uint8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              0,
              reinterpret_cast<uint16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint32:
            err = kernel::NumpyArray_fill<uint32_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              0,
              reinterpret_cast<uint32_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, int32, int64, "
                          "uint8, uint16, uint32} (1)") + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int32:
            err = kernel::NumpyArray_fill<int32_t , int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int32_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int64:
            err = kernel::NumpyArray_fill<int64_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int64_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint32:
            err = kernel::NumpyArray_fill<uint32_t, int64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<int64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint32_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, int32, int64, "
                          "uint8, uint16, uint32} (2)") + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to uint8
      case util::dtype::uint8:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<uint8_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint8_t*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, uint8_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint8_t*>(ptr.get()),
              0,
              reinterpret_cast<uint8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, uint8} (1)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<uint8_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint8_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, uint8_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint8_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, uint8} (2)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to uint16
      case util::dtype::uint16:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<uint16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint16_t*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, uint16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint16_t*>(ptr.get()),
              0,
              reinterpret_cast<uint8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, uint16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint16_t*>(ptr.get()),
              0,
              reinterpret_cast<uint16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, uint8, uint16} (1)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<uint16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint16_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, uint16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint16_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, uint16_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint16_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, uint8, uint16} (2)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to uint32
      case util::dtype::uint32:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<uint32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint32_t*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, uint32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint32_t*>(ptr.get()),
              0,
              reinterpret_cast<uint8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, uint32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint32_t*>(ptr.get()),
              0,
              reinterpret_cast<uint16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint32:
            err = kernel::NumpyArray_fill<uint32_t, uint32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint32_t*>(ptr.get()),
              0,
              reinterpret_cast<uint32_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, uint8, uint16, uint32} (1)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<uint32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, uint32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, uint32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint32:
            err = kernel::NumpyArray_fill<uint32_t, uint32_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint32_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint32_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, uint8, uint16, uint32} (2)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to uint64
      case util::dtype::uint64:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              0,
              reinterpret_cast<uint8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              0,
              reinterpret_cast<uint16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint32:
            err = kernel::NumpyArray_fill<uint32_t, uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              0,
              reinterpret_cast<uint32_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint64:
            err = kernel::NumpyArray_fill<uint64_t, uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              0,
              reinterpret_cast<uint64_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, uint8, uint16, uint32, uint64} (1)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint32:
            err = kernel::NumpyArray_fill<uint32_t, uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint32_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint64:
            err = kernel::NumpyArray_fill<uint64_t, uint64_t>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<uint64_t*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint64_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, uint8, uint16, uint32, uint64} (2)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to float16
      case util::dtype::float16:
        throw std::runtime_error(
          std::string("FIXME: merge to float16 not implemented") + FILENAME(__LINE__));
        break;

      // to float32
      case util::dtype::float32:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              0,
              reinterpret_cast<int8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              0,
              reinterpret_cast<int16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              0,
              reinterpret_cast<uint8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              0,
              reinterpret_cast<uint16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::float16:
            throw std::runtime_error(
              std::string("FIXME: merge from float16 not implemented")
              + FILENAME(__LINE__));
          case util::dtype::float32:
            err = kernel::NumpyArray_fill<float, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              0,
              reinterpret_cast<float*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, uint8, uint16, "
                          "float16, float32} (1)") + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::float16:
            throw std::runtime_error(
              std::string("FIXME: merge from float16 not implemented")
              + FILENAME(__LINE__));
          case util::dtype::float32:
            err = kernel::NumpyArray_fill<float, float>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<float*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<float*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, uint8, uint16, "
                          "float16, float32} (2)") + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to float64
      case util::dtype::float64:
        switch (dtype_) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<bool*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<int8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<int16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int32:
            err = kernel::NumpyArray_fill<int32_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<int32_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::int64:
            err = kernel::NumpyArray_fill<int64_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<int64_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<uint8_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<uint16_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint32:
            err = kernel::NumpyArray_fill<uint32_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<uint32_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::uint64:
            err = kernel::NumpyArray_fill<uint64_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<uint64_t*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::float16:
            throw std::runtime_error(
              std::string("FIXME: merge from float16 not implemented")
              + FILENAME(__LINE__));
          case util::dtype::float32:
            err = kernel::NumpyArray_fill<float, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<float*>(contiguous_self.data()),
              self_flatlength);
            break;
          case util::dtype::float64:
            err = kernel::NumpyArray_fill<double, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              0,
              reinterpret_cast<double*>(contiguous_self.data()),
              self_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, int32, int64, "
                          "uint8, uint16, uint32, uint64 float16, float32, float64} (1)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        switch (rawother->dtype()) {
          case util::dtype::boolean:
            err = kernel::NumpyArray_fill_frombool<double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<bool*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int8:
            err = kernel::NumpyArray_fill<int8_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int16:
            err = kernel::NumpyArray_fill<int16_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int32:
            err = kernel::NumpyArray_fill<int32_t , double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int32_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::int64:
            err = kernel::NumpyArray_fill<int64_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<int64_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint8:
            err = kernel::NumpyArray_fill<uint8_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint8_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint16:
            err = kernel::NumpyArray_fill<uint16_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint16_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint32:
            err = kernel::NumpyArray_fill<uint32_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint32_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::uint64:
            err = kernel::NumpyArray_fill<uint64_t, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<uint64_t*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::float16:
            throw std::runtime_error(
              std::string("FIXME: merge from float16 not implemented")
              + FILENAME(__LINE__));
          case util::dtype::float32:
            err = kernel::NumpyArray_fill<float, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<float*>(contiguous_other.data()),
              other_flatlength);
            break;
          case util::dtype::float64:
            err = kernel::NumpyArray_fill<double, double>(
              kernel::lib::cpu,   // DERIVE
              reinterpret_cast<double*>(ptr.get()),
              self_flatlength,
              reinterpret_cast<double*>(contiguous_other.data()),
              other_flatlength);
            break;
          default:
            throw std::runtime_error(
              std::string("dtype_ not in {boolean, int8, int16, int32, int64, "
                          "uint8, uint16, uint32, uint64 float16, float32, float64} (2)")
              + FILENAME(__LINE__));
        }
        util::handle_error(err, classname(), nullptr);
        break;

      // to float128
      case util::dtype::float128:
        throw std::runtime_error(
          std::string("FIXME: merge to float128 not implemented")
          + FILENAME(__LINE__));
        break;

      // to complex64
      case util::dtype::complex64:
        throw std::runtime_error(
          std::string("FIXME: merge to complex64 not implemented")
          + FILENAME(__LINE__));
        break;

      // to complex128
      case util::dtype::complex128:
        throw std::runtime_error(
          std::string("FIXME: merge to complex128 not implemented")
          + FILENAME(__LINE__));
        break;

      // to complex256
      case util::dtype::complex256:
        throw std::runtime_error(
          std::string("FIXME: merge to complex256 not implemented")
          + FILENAME(__LINE__));
        break;

      // something's wrong
      default:
        throw std::runtime_error(
          std::string("unhandled merge case: to ") + util::dtype_to_name(dtype)
          + FILENAME(__LINE__));
      }

      return std::make_shared<NumpyArray>(Identities::none(),
                                          parameters_,
                                          ptr,
                                          shape,
                                          strides,
                                          0,
                                          (ssize_t)itemsize,
                                          util::dtype_to_format(dtype),
                                          dtype,
                                          ptr_lib);
    }

    else {
      throw std::invalid_argument(
        std::string("cannot merge ") + classname() + std::string(" with ")
        + other.get()->classname() + FILENAME(__LINE__));
    }
  }

  const ContentPtr
  NumpyArray::merge_bytes(const std::shared_ptr<NumpyArray>& other) const {
    NumpyArray contiguous_self = contiguous();
    NumpyArray contiguous_other = other.get()->contiguous();

    std::shared_ptr<void> ptr(
      kernel::malloc<void>(ptr_lib_, length() + other.get()->length()));

    kernel::lib ptr_lib = ptr_lib_;   // DERIVE

    struct Error err;

    err = kernel::NumpyArray_fill<uint8_t, uint8_t>(
      kernel::lib::cpu,   // DERIVE
      reinterpret_cast<uint8_t*>(ptr.get()),
      0,
      reinterpret_cast<uint8_t*>(contiguous_self.data()),
      contiguous_self.length());
    util::handle_error(err, classname(), nullptr);

    err = kernel::NumpyArray_fill<uint8_t, uint8_t>(
      kernel::lib::cpu,   // DERIVE
      reinterpret_cast<uint8_t*>(ptr.get()),
      length(),
      reinterpret_cast<uint8_t*>(contiguous_other.data()),
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
                                        format_,
                                        dtype_,
                                        ptr_lib);
  }

  const SliceItemPtr
  NumpyArray::asslice() const {
    if (ndim() != 1) {
      throw std::invalid_argument(
        std::string("slice items can have all fixed-size dimensions (to follow "
                    "NumPy's slice rules) or they can have all var-sized "
                    "dimensions (for jagged indexing), but not both in the "
                    "same slice item") + FILENAME(__LINE__));
    }
    if (dtype_ == util::dtype::int64) {
        int64_t* raw = reinterpret_cast<int64_t*>(ptr_.get());
        std::shared_ptr<int64_t> ptr(ptr_, raw);
        std::vector<int64_t> shape({ (int64_t)shape_[0] });
        std::vector<int64_t> strides({ (int64_t)strides_[0] /
                                       (int64_t)itemsize_ });
        return std::make_shared<SliceArray64>(
          Index64(ptr, (int64_t)byteoffset_ / (int64_t)itemsize_, length(), ptr_lib_),
          shape,
          strides,
          false);
    }

    else if (util::is_integer(dtype_)) {
      NumpyArray contiguous_self = contiguous();
      Index64 index(length());

      struct Error err;
      switch (dtype_) {
      case util::dtype::int8:
        err = kernel::NumpyArray_fill<int8_t, int64_t>(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          0,
          reinterpret_cast<int8_t*>(contiguous_self.data()),
          length());
        break;
      case util::dtype::int16:
        err = kernel::NumpyArray_fill<int16_t, int64_t>(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          0,
          reinterpret_cast<int16_t*>(contiguous_self.data()),
          length());
        break;
      case util::dtype::int32:
        err = kernel::NumpyArray_fill<int32_t, int64_t>(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          0,
          reinterpret_cast<int32_t*>(contiguous_self.data()),
          length());
        break;
      case util::dtype::uint8:
        err = kernel::NumpyArray_fill<uint8_t, int64_t>(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          0,
          reinterpret_cast<uint8_t*>(contiguous_self.data()),
          length());
        break;
      case util::dtype::uint16:
        err = kernel::NumpyArray_fill<uint16_t, int64_t>(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          0,
          reinterpret_cast<uint16_t*>(contiguous_self.data()),
          length());
        break;
      case util::dtype::uint32:
        err = kernel::NumpyArray_fill<uint32_t, int64_t>(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          0,
          reinterpret_cast<uint32_t*>(contiguous_self.data()),
          length());
        break;
      case util::dtype::uint64:
        err = kernel::NumpyArray_fill<uint64_t, int64_t>(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          0,
          reinterpret_cast<uint64_t*>(contiguous_self.data()),
          length());
        break;
      default:
        throw std::runtime_error(
          std::string("unexpected integer type in NumpyArray::asslice: ") +
          util::dtype_to_name(dtype_) + FILENAME(__LINE__));
      }
      util::handle_error(err, classname(), identities_.get());

      std::vector<int64_t> shape( {(int64_t)shape_[0] });
      std::vector<int64_t> strides( {1} );
      return std::make_shared<SliceArray64>(index, shape, strides, false);
    }

    else if (dtype_ == util::dtype::boolean) {
      int64_t numtrue;
      struct Error err1 = kernel::NumpyArray_getitem_boolean_numtrue(
        kernel::lib::cpu,   // DERIVE
        &numtrue,
        reinterpret_cast<int8_t*>(data()),
        (int64_t)shape_[0],
        (int64_t)strides_[0]);
      util::handle_error(err1, classname(), identities_.get());

      Index64 index(numtrue);
      struct Error err2 = kernel::NumpyArray_getitem_boolean_nonzero_64(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        reinterpret_cast<int8_t*>(data()),
        (int64_t)shape_[0],
        (int64_t)strides_[0]);
      util::handle_error(err2, classname(), identities_.get());

      std::vector<int64_t> shape({ numtrue });
      std::vector<int64_t> strides({ 1 });
      return std::make_shared<SliceArray64>(index, shape, strides, true);
    }
    else {
      throw std::invalid_argument(
        std::string("only arrays of integers or booleans may be used as a slice")
        + FILENAME(__LINE__));
    }
  }

  const ContentPtr
  NumpyArray::fillna(const ContentPtr& value) const {
    return shallow_copy();
  }

  const ContentPtr
  NumpyArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    if (ndim() == 0) {
      throw std::runtime_error(
        std::string("cannot rpad a scalar") + FILENAME(__LINE__));
    }
    else if (ndim() > 1  ||  !iscontiguous()) {
      return toRegularArray().get()->rpad(target, axis, depth);
    }
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis != depth) {
      throw std::invalid_argument(
        std::string("axis exceeds the depth of this array") + FILENAME(__LINE__));
    }
    if (target < length()) {
      return shallow_copy();
    }
    else {
      return rpad_and_clip(target, posaxis, depth);
    }
  }

  const ContentPtr
  NumpyArray::rpad_and_clip(int64_t target,
                            int64_t axis,
                            int64_t depth) const {
    if (ndim() == 0) {
      throw std::runtime_error(
        std::string("cannot rpad a scalar") + FILENAME(__LINE__));
    }
    else if (ndim() > 1  ||  !iscontiguous()) {
      return toRegularArray().get()->rpad_and_clip(target, axis, depth);
    }
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis != depth) {
      throw std::invalid_argument(
        std::string("axis exceeds the depth of this array") + FILENAME(__LINE__));
    }
    return rpad_axis0(target, true);
  }

  const ContentPtr
  NumpyArray::reduce_next(const Reducer& reducer,
                          int64_t negaxis,
                          const Index64& starts,
                          const Index64& shifts,
                          const Index64& parents,
                          int64_t outlength,
                          bool mask,
                          bool keepdims) const {
    if (shape_.empty()) {
      throw std::runtime_error(
        std::string("attempting to reduce a scalar") + FILENAME(__LINE__));
    }
    else if (shape_.size() != 1  ||  !iscontiguous()) {
      return toRegularArray().get()->reduce_next(reducer,
                                                 negaxis,
                                                 starts,
                                                 shifts,
                                                 parents,
                                                 outlength,
                                                 mask,
                                                 keepdims);
    }
    else {
      std::shared_ptr<void> ptr;
      switch (dtype_) {
      case util::dtype::boolean:
        ptr = reducer.apply_bool(reinterpret_cast<bool*>(data()),
                                 parents,
                                 outlength);
        break;
      case util::dtype::int8:
        ptr = reducer.apply_int8(reinterpret_cast<int8_t*>(data()),
                                 parents,
                                 outlength);
        break;
      case util::dtype::int16:
        ptr = reducer.apply_int16(reinterpret_cast<int16_t*>(data()),
                                  parents,
                                  outlength);
        break;
      case util::dtype::int32:
        ptr = reducer.apply_int32(reinterpret_cast<int32_t*>(data()),
                                  parents,
                                  outlength);
        break;
      case util::dtype::int64:
        ptr = reducer.apply_int64(reinterpret_cast<int64_t*>(data()),
                                  parents,
                                  outlength);
        break;
      case util::dtype::uint8:
        ptr = reducer.apply_uint8(reinterpret_cast<uint8_t*>(data()),
                                  parents,
                                  outlength);
        break;
      case util::dtype::uint16:
        ptr = reducer.apply_uint16(reinterpret_cast<uint16_t*>(data()),
                                   parents,
                                   outlength);
        break;
      case util::dtype::uint32:
        ptr = reducer.apply_uint32(reinterpret_cast<uint32_t*>(data()),
                                   parents,
                                   outlength);
        break;
      case util::dtype::uint64:
        ptr = reducer.apply_uint64(reinterpret_cast<uint64_t*>(data()),
                                   parents,
                                   outlength);
        break;
      case util::dtype::float16:
        throw std::runtime_error(
          std::string("FIXME: reducers on float16") + FILENAME(__LINE__));
      case util::dtype::float32:
        ptr = reducer.apply_float32(reinterpret_cast<float*>(data()),
                                    parents,
                                    outlength);
        break;
      case util::dtype::float64:
        ptr = reducer.apply_float64(reinterpret_cast<double*>(data()),
                                    parents,
                                    outlength);
        break;
      case util::dtype::float128:
        throw std::runtime_error(
          std::string("FIXME: reducers on float128") + FILENAME(__LINE__));
      case util::dtype::complex64:
        throw std::runtime_error(
          std::string("FIXME: reducers on complex64") + FILENAME(__LINE__));
      case util::dtype::complex128:
        throw std::runtime_error(
          std::string("FIXME: reducers on complex128") + FILENAME(__LINE__));
      case util::dtype::complex256:
        throw std::runtime_error(
          std::string("FIXME: reducers on complex256") + FILENAME(__LINE__));
      // case util::dtype::datetime64:
      //   throw std::runtime_error(
      //     std::string("FIXME: reducers on datetime64") + FILENAME(__LINE__));
      // case util::dtype:::timedelta64:
      //   throw std::runtime_error(
      //     std:string("FIXME: reducers on timedelta64") + FILENAME(__LINE__));
      default:
        throw std::invalid_argument(
          std::string("cannot apply reducers to NumpyArray with format \"")
          + format_ + std::string("\"") + FILENAME(__LINE__));
      }

      if (reducer.returns_positions()) {
        struct Error err3;
        if (shifts.length() == 0) {
          err3 = kernel::NumpyArray_reduce_adjust_starts_64(
            kernel::lib::cpu,   // DERIVE
            reinterpret_cast<int64_t*>(ptr.get()),
            outlength,
            parents.data(),
            starts.data());
        }
        else {
          err3 = kernel::NumpyArray_reduce_adjust_starts_shifts_64(
            kernel::lib::cpu,   // DERIVE
            reinterpret_cast<int64_t*>(ptr.get()),
            outlength,
            parents.data(),
            starts.data(),
            shifts.data());
        }
        util::handle_error(err3, classname(), identities_.get());
      }

      util::dtype dtype = reducer.return_dtype(dtype_);
      std::string format = util::dtype_to_format(dtype);
      ssize_t itemsize = util::dtype_to_itemsize(dtype);

      std::vector<ssize_t> shape({ (ssize_t)outlength });
      std::vector<ssize_t> strides({ itemsize });
      ContentPtr out = std::make_shared<NumpyArray>(Identities::none(),
                                                    util::Parameters(),
                                                    ptr,
                                                    shape,
                                                    strides,
                                                    0,
                                                    itemsize,
                                                    format,
                                                    dtype,
                                                    ptr_lib_);

      if (mask) {
        Index8 mask(outlength);
        struct Error err = kernel::NumpyArray_reduce_mask_ByteMaskedArray_64(
          kernel::lib::cpu,   // DERIVE
          mask.data(),
          parents.data(),
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
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return localindex_axis0();
    }
    else if (shape_.size() <= 1) {
      throw std::invalid_argument(
        std::string("'axis' out of range for localindex") + FILENAME(__LINE__));
    }
    else {
      return toRegularArray().get()->localindex(posaxis, depth);
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
      throw std::invalid_argument(
        std::string("in combinations, 'n' must be at least 1") + FILENAME(__LINE__));
    }

    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return combinations_axis0(n, replacement, recordlookup, parameters);
    }

    else if (shape_.size() <= 1) {
      throw std::invalid_argument(
        std::string("'axis' out of range for combinations") + FILENAME(__LINE__));
    }

    else {
      return toRegularArray().get()->combinations(n,
                                                  replacement,
                                                  recordlookup,
                                                  parameters,
                                                  posaxis,
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
      throw std::runtime_error(
        std::string("attempting to sort a scalar") + FILENAME(__LINE__));
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
      std::shared_ptr<void> ptr;

      switch (dtype_) {
      case util::dtype::boolean:
        ptr = array_sort<bool>(reinterpret_cast<bool*>(data()),
                               length(),
                               starts,
                               parents,
                               outlength,
                               ascending,
                               stable);
        break;
      case util::dtype::int8:
        ptr = array_sort<int8_t>(reinterpret_cast<int8_t*>(data()),
                                 length(),
                                 starts,
                                 parents,
                                 outlength,
                                 ascending,
                                 stable);
        break;
      case util::dtype::int16:
        ptr = array_sort<int16_t>(reinterpret_cast<int16_t*>(data()),
                                  length(),
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
        break;
      case util::dtype::int32:
        ptr = array_sort<int32_t>(reinterpret_cast<int32_t*>(data()),
                                  length(),
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
        break;
      case util::dtype::int64:
        ptr = array_sort<int64_t>(reinterpret_cast<int64_t*>(data()),
                                  length(),
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
        break;
      case util::dtype::uint8:
        ptr = array_sort<uint8_t>(reinterpret_cast<uint8_t*>(data()),
                                  length(),
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
        break;
      case util::dtype::uint16:
        ptr = array_sort<uint16_t>(reinterpret_cast<uint16_t*>(data()),
                                   length(),
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
        break;
      case util::dtype::uint32:
        ptr = array_sort<uint32_t>(reinterpret_cast<uint32_t*>(data()),
                                   length(),
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
        break;
      case util::dtype::uint64:
        ptr = array_sort<uint64_t>(reinterpret_cast<uint64_t*>(data()),
                                   length(),
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
        break;
      case util::dtype::float16:
        throw std::runtime_error(
          std::string("FIXME: sort for float16 not implemented") + FILENAME(__LINE__));
      case util::dtype::float32:
        ptr = array_sort<float>(reinterpret_cast<float*>(data()),
                                length(),
                                starts,
                                parents,
                                outlength,
                                ascending,
                                stable);
        break;
      case util::dtype::float64:
        ptr = array_sort<double>(reinterpret_cast<double*>(data()),
                                 length(),
                                 starts,
                                 parents,
                                 outlength,
                                 ascending,
                                 stable);
        break;
      case util::dtype::float128:
        throw std::runtime_error(
          std::string("FIXME: sort for float128 not implemented") + FILENAME(__LINE__));
      case util::dtype::complex64:
        throw std::runtime_error(
          std::string("FIXME: sort for complex64 not implemented") + FILENAME(__LINE__));
      case util::dtype::complex128:
        throw std::runtime_error(
          std::string("FIXME: sort for complex128 not implemented") + FILENAME(__LINE__));
      case util::dtype::complex256:
        throw std::runtime_error(
          std::string("FIXME: sort for complex256 not implemented") + FILENAME(__LINE__));
      default:
        throw std::invalid_argument(
          std::string("cannot sort NumpyArray with format \"")
          + format_ + std::string("\"") + FILENAME(__LINE__));
      }

      out = std::make_shared<NumpyArray>(Identities::none(),
                                         parameters_,
                                         ptr,
                                         shape_,
                                         strides_,
                                         0,
                                         itemsize_,
                                         format_,
                                         dtype_,
                                         ptr_lib_);

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
      throw std::runtime_error(
        std::string("attempting to argsort a scalar") + FILENAME(__LINE__));
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
      std::shared_ptr<void> ptr;

      switch (dtype_) {
      case util::dtype::boolean:
        ptr = index_sort<bool>(reinterpret_cast<bool*>(data()),
                               length(),
                               starts,
                               parents,
                               outlength,
                               ascending,
                               stable);
        break;
      case util::dtype::int8:
        ptr = index_sort<int8_t>(reinterpret_cast<int8_t*>(data()),
                                 length(),
                                 starts,
                                 parents,
                                 outlength,
                                 ascending,
                                 stable);
        break;
      case util::dtype::int16:
        ptr = index_sort<int16_t>(reinterpret_cast<int16_t*>(data()),
                                  length(),
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
        break;
      case util::dtype::int32:
        ptr = index_sort<int32_t>(reinterpret_cast<int32_t*>(data()),
                                  length(),
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
        break;
      case util::dtype::int64:
        ptr = index_sort<int64_t>(reinterpret_cast<int64_t*>(data()),
                                  length(),
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
        break;
      case util::dtype::uint8:
        ptr = index_sort<uint8_t>(reinterpret_cast<uint8_t*>(data()),
                                  length(),
                                  starts,
                                  parents,
                                  outlength,
                                  ascending,
                                  stable);
        break;
      case util::dtype::uint16:
        ptr = index_sort<uint16_t>(reinterpret_cast<uint16_t*>(data()),
                                   length(),
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
        break;
      case util::dtype::uint32:
        ptr = index_sort<uint32_t>(reinterpret_cast<uint32_t*>(data()),
                                   length(),
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
        break;
      case util::dtype::uint64:
        ptr = index_sort<uint64_t>(reinterpret_cast<uint64_t*>(data()),
                                   length(),
                                   starts,
                                   parents,
                                   outlength,
                                   ascending,
                                   stable);
        break;
      case util::dtype::float16:
        throw std::runtime_error(
          std::string("FIXME: argsort for float16 not implemented")
          + FILENAME(__LINE__));
      case util::dtype::float32:
        ptr = index_sort<float>(reinterpret_cast<float*>(data()),
                                length(),
                                starts,
                                parents,
                                outlength,
                                ascending,
                                stable);
        break;
      case util::dtype::float64:
        ptr = index_sort<double>(reinterpret_cast<double*>(data()),
                                 length(),
                                 starts,
                                 parents,
                                 outlength,
                                 ascending,
                                 stable);
        break;
      case util::dtype::float128:
        throw std::runtime_error(
          std::string("FIXME: argsort for float128 not implemented")
          + FILENAME(__LINE__));
      case util::dtype::complex64:
        throw std::runtime_error(
          std::string("FIXME: argsort for complex64 not implemented")
          + FILENAME(__LINE__));
      case util::dtype::complex128:
        throw std::runtime_error(
          std::string("FIXME: argsort for complex128 not implemented")
          + FILENAME(__LINE__));
      case util::dtype::complex256:
        throw std::runtime_error(
          std::string("FIXME: argsort for complex256 not implemented")
          + FILENAME(__LINE__));
      default:
        throw std::invalid_argument(
          std::string("cannot sort NumpyArray with format \"")
          + format_ + std::string("\"") + FILENAME(__LINE__));
      }

      ssize_t itemsize = 8;
      util::dtype dtype = util::dtype::int64;
      std::vector<ssize_t> shape({ (ssize_t)shape_[0] });
      std::vector<ssize_t> strides({ itemsize });
      out = std::make_shared<NumpyArray>(Identities::none(),
                                         util::Parameters(),
                                         ptr,
                                         shape_,
                                         strides,
                                         0,
                                         itemsize,
                                         util::dtype_to_format(dtype),
                                         dtype,
                                         ptr_lib_);

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
    std::shared_ptr<void> ptr;

    Index64 outoffsets(offsets.length());

    if (dtype_ == util::dtype::uint8) {
      ptr = string_sort<uint8_t>(reinterpret_cast<uint8_t*>(data()),
                                 length(),
                                 offsets,
                                 outoffsets,
                                 ascending,
                                 stable);
    } else {
      throw std::invalid_argument(
        std::string("cannot sort NumpyArray as strings with format \"")
        + format_ + std::string("\"") + FILENAME(__LINE__));
    }

    out = std::make_shared<NumpyArray>(identities_,
                                       parameters_,
                                       ptr,
                                       shape_,
                                       strides_,
                                       0,
                                       itemsize_,
                                       format_,
                                       dtype_,
                                       ptr_lib_);

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
      std::string("undefined operation: NumpyArray::getitem_next(at) "
                  "(without 'length', 'stride', and 'first')")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceRange& range,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: NumpyArray::getitem_next(range) "
                  "(without 'length', 'stride', and 'first')")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceArray64& array,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: NumpyArray::getitem_next(array) "
                  "(without 'length','stride', and 'first')")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceField& field,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: NumpyArray::getitem_next(field) "
                  "(without 'length', 'stride', and 'first')")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceFields& fields,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: NumpyArray::getitem_next(fields) "
                  "(without 'length', 'stride', and 'first')")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  NumpyArray::getitem_next(const SliceJagged64& jagged,
                           const Slice& tail, const Index64& advanced) const {
    if (shape_.size() != 1) {
      throw std::runtime_error(
        std::string("undefined operation: NumpyArray::getitem_next(jagged) with "
                    "ndim != 1") + FILENAME(__LINE__));
    }

    if (advanced.length() != 0) {
      throw std::invalid_argument(
        std::string("cannot mix jagged slice with NumPy-style advanced indexing")
        + FILENAME(__LINE__));
    }

    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by a jagged array because it is one-dimensional")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  NumpyArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceArray64& slicecontent,
                                  const Slice& tail) const {
    if (ndim() == 1) {
      throw std::invalid_argument(
        std::string("too many jagged slice dimensions for array")
        + FILENAME(__LINE__));
    }
    else {
      throw std::runtime_error(
        std::string("undefined operation: NumpyArray::getitem_next_jagged("
                    "array) for ndim == ") + std::to_string(ndim())
        + FILENAME(__LINE__));
    }
  }

  const ContentPtr
  NumpyArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceMissing64& slicecontent,
                                  const Slice& tail) const {
    if (ndim() == 1) {
      throw std::invalid_argument(
        std::string("too many jagged slice dimensions for array")
        + FILENAME(__LINE__));
    }
    else {
      throw std::runtime_error(
        std::string("undefined operation: NumpyArray::getitem_next_jagged("
                    "missing) for ndim == ") + std::to_string(ndim())
        + FILENAME(__LINE__));
    }
  }

  const ContentPtr
  NumpyArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceJagged64& slicecontent,
                                  const Slice& tail) const {
    if (ndim() == 1) {
      throw std::invalid_argument(
        std::string("too many jagged slice dimensions for array")
        + FILENAME(__LINE__));
    }
    else {
      throw std::runtime_error(
        std::string("undefined operation: NumpyArray::getitem_next_jagged("
                    "jagged) for ndim == ") + std::to_string(ndim())
        + FILENAME(__LINE__));
    }
  }

  bool
  NumpyArray::iscontiguous() const {
    ssize_t x = itemsize_;
    for (ssize_t i = ndim() - 1;  i >= 0;  i--) {
      if (x != strides_[(size_t)i]) return false;
      x *= shape_[(size_t)i];
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
                        format_,
                        dtype_,
                        ptr_lib_);
    }
    else {
      Index64 bytepos(shape_[0]);
      struct Error err = kernel::NumpyArray_contiguous_init_64(
        kernel::lib::cpu,   // DERIVE
        bytepos.data(),
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
        kernel::malloc<void>(ptr_lib_, bytepos.length()*strides_[0]));

      struct Error err = kernel::NumpyArray_contiguous_copy_64(
        kernel::lib::cpu,   // DERIVE
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(data()),
        bytepos.length(),
        strides_[0],
        bytepos.data());
      util::handle_error(err, classname(), identities_.get());
      return NumpyArray(identities_,
                        parameters_,
                        ptr,
                        shape_,
                        strides_,
                        0,
                        itemsize_,
                        format_,
                        dtype_,
                        ptr_lib_);
    }

    else if (shape_.size() == 1) {
      std::shared_ptr<void> ptr(
        kernel::malloc<void>(ptr_lib_, bytepos.length()*((int64_t)itemsize_)));
      struct Error err = kernel::NumpyArray_contiguous_copy_64(
        kernel::lib::cpu,   // DERIVE
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(data()),
        bytepos.length(),
        itemsize_,
        bytepos.data());
      util::handle_error(err, classname(), identities_.get());
      std::vector<ssize_t> strides = { itemsize_ };
      return NumpyArray(identities_,
                        parameters_,
                        ptr,
                        shape_,
                        strides,
                        0,
                        itemsize_,
                        format_,
                        dtype_,
                        ptr_lib_);
    }

    else {
      NumpyArray next(identities_,
                      parameters_,
                      ptr_,
                      flatten_shape(shape_),
                      flatten_strides(strides_),
                      byteoffset_,
                      itemsize_,
                      format_,
                      dtype_,
                      ptr_lib_);

      Index64 nextbytepos(bytepos.length()*shape_[1]);
      struct Error err = kernel::NumpyArray_contiguous_next_64(
        kernel::lib::cpu,   // DERIVE
        nextbytepos.data(),
        bytepos.data(),
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
                        format_,
                        dtype_,
                        ptr_lib_);
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
                        format_,
                        dtype_,
                        ptr_lib_);
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
        std::string("unrecognized slice item type for NumpyArray::getitem_bystrides")
        + FILENAME(__LINE__));
    }
  }

  const NumpyArray
  NumpyArray::getitem_bystrides(const SliceAt& at,
                                const Slice& tail,
                                int64_t length) const {
    if (ndim() < 2) {
      util::handle_error(
        failure("too many dimensions in slice",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }

    int64_t i = at.at();
    if (i < 0) i += shape_[1];
    if (i < 0  ||  i >= shape_[1]) {
      util::handle_error(
        failure("index out of range", kSliceNone, at.at(), FILENAME_C(__LINE__)),
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
                    format_,
                    dtype_,
                    ptr_lib_);

    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = next.getitem_bystrides(nexthead, nexttail, length);

    std::vector<ssize_t> outshape = { (ssize_t)length };
    outshape.insert(outshape.end(), std::next(out.shape_.begin()), out.shape_.end());
    return NumpyArray(out.identities_,
                      out.parameters_,
                      out.ptr_,
                      outshape,
                      out.strides_,
                      out.byteoffset_,
                      itemsize_,
                      format_,
                      dtype_,
                      ptr_lib_);
  }

  const NumpyArray
  NumpyArray::getitem_bystrides(const SliceRange& range,
                                const Slice& tail,
                                int64_t length) const {
    if (ndim() < 2) {
      util::handle_error(
        failure("too many dimensions in slice",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
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
                    format_,
                    dtype_,
                    ptr_lib_);

    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = next.getitem_bystrides(nexthead,
                                            nexttail,
                                            length*lenhead);

    std::vector<ssize_t> outshape = { (ssize_t)length,
                                      (ssize_t)lenhead };
    outshape.insert(outshape.end(), std::next(out.shape_.begin()), out.shape_.end());
    std::vector<ssize_t> outstrides = { strides_[0],
                                        strides_[1]*((ssize_t)step) };
    outstrides.insert(outstrides.end(),
                      std::next(out.strides_.begin()),
                      out.strides_.end());
    return NumpyArray(out.identities_,
                      out.parameters_,
                      out.ptr_,
                      outshape,
                      outstrides,
                      out.byteoffset_,
                      itemsize_,
                      format_,
                      dtype_,
                      ptr_lib_);
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
    outshape.insert(outshape.end(), std::next(out.shape_.begin()), out.shape_.end());
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
                      format_,
                      dtype_,
                      ptr_lib_);
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
      std::shared_ptr<void> ptr(kernel::malloc<void>(ptr_lib_, carry.length()*stride));
      struct Error err = kernel::NumpyArray_getitem_next_null_64(
        kernel::lib::cpu,   // DERIVE
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(data()),
        carry.length(),
        stride,
        carry.ptr().get());
      util::handle_error(err, classname(), identities_.get());

      IdentitiesPtr identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_carry_64(carry);
      }

      std::vector<ssize_t> shape = { (ssize_t)carry.length() };
      shape.insert(shape.end(), std::next(shape_.begin()), shape_.end());
      std::vector<ssize_t> strides = { (ssize_t)stride };
      strides.insert(strides.end(), std::next(strides_.begin()), strides_.end());
      return NumpyArray(identities,
                        parameters_,
                        ptr,
                        shape,
                        strides,
                        0,
                        itemsize_,
                        format_,
                        dtype_,
                        ptr_lib_);
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
        + std::string(" by a field name because it has no fields")
        + FILENAME(__LINE__));
    }
    else if (SliceFields* fields =
             dynamic_cast<SliceFields*>(head.get())) {
      throw std::invalid_argument(
        std::string("cannot slice ") + classname()
        + std::string(" by field names because it has no fields")
        + FILENAME(__LINE__));
    }
    else if (SliceMissing64* missing =
             dynamic_cast<SliceMissing64*>(head.get())) {
      throw std::runtime_error(
        std::string("undefined operation: NumpyArray::getitem_next(missing) "
                    "(defer to Content::getitem_next(missing))")
        + FILENAME(__LINE__));
    }
    else if (SliceJagged64* jagged =
             dynamic_cast<SliceJagged64*>(head.get())) {
      throw std::runtime_error(
        std::string("FIXME: NumpyArray::getitem_next(jagged)") + FILENAME(__LINE__));
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized slice item type") + FILENAME(__LINE__));
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
        failure("too many dimensions in slice",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
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
                    format_,
                    dtype_,
                    ptr_lib_);
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();

    int64_t regular_at = at.at();
    if (regular_at < 0) {
      regular_at += shape_[1];
    }
    if (!(0 <= regular_at  &&  regular_at < shape_[1])) {
      util::handle_error(
        failure("index out of range", kSliceNone, at.at(), FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }

    Index64 nextcarry(carry.length());
    struct Error err = kernel::NumpyArray_getitem_next_at_64(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      carry.data(),
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
    outshape.insert(outshape.end(), std::next(out.shape_.begin()), out.shape_.end());
    return NumpyArray(out.identities_,
                      out.parameters_,
                      out.ptr_,
                      outshape,
                      out.strides_,
                      out.byteoffset_,
                      itemsize_,
                      format_,
                      dtype_,
                      ptr_lib_);
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
        failure("too many dimensions in slice",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
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
                    format_,
                    dtype_,
                    ptr_lib_);
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();

    if (advanced.length() == 0) {
      Index64 nextcarry(carry.length()*lenhead);
      struct Error err = kernel::NumpyArray_getitem_next_range_64(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        carry.data(),
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
                      std::next(out.shape_.begin()),
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
                        format_,
                        dtype_,
                        ptr_lib_);
    }

    else {
      Index64 nextcarry(carry.length()*lenhead);
      Index64 nextadvanced(carry.length()*lenhead);
      struct Error err = kernel::NumpyArray_getitem_next_range_advanced_64(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        nextadvanced.data(),
        carry.data(),
        advanced.data(),
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
                      std::next(out.shape_.begin()),
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
                        format_,
                        dtype_,
                        ptr_lib_);
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
                    std::next(out.shape_.begin()),
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
                      format_,
                      dtype_,
                      ptr_lib_);
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
        failure("too many dimensions in slice",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
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
                    format_,
                    dtype_,
                    ptr_lib_);
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();

    Index64 flathead = array.ravel();
    struct Error err = kernel::regularize_arrayslice_64(
      kernel::lib::cpu,   // DERIVE
      flathead.data(),
      flathead.length(),
      shape_[1]);
    util::handle_error(err, classname(), identities_.get());

    if (advanced.length() == 0) {
      Index64 nextcarry(carry.length()*flathead.length());
      Index64 nextadvanced(carry.length()*flathead.length());
      struct Error err = kernel::NumpyArray_getitem_next_array_64(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        nextadvanced.data(),
        carry.data(),
        flathead.data(),
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
      for (auto x : arrayshape) {
        outshape.emplace_back((ssize_t)x);
      }
      outshape.insert(outshape.end(),
                      std::next(out.shape_.begin()),
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
                        format_,
                        dtype_,
                        ptr_lib_);
    }

    else {
      Index64 nextcarry(carry.length());
      struct Error err = kernel::NumpyArray_getitem_next_array_advanced_64(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        carry.data(),
        advanced.data(),
        flathead.data(),
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
                      std::next(out.shape_.begin()),
                      out.shape_.end());
      return NumpyArray(out.identities_,
                        out.parameters_,
                        out.ptr_,
                        outshape,
                        out.strides_,
                        out.byteoffset_,
                        itemsize_,
                        format_,
                        dtype_,
                        ptr_lib_);
    }
  }

  void
  NumpyArray::tojson_boolean(ToJson& builder,
                             bool include_beginendlist) const {
    if (ndim() == 0) {
      bool* array = reinterpret_cast<bool*>(data());
      builder.boolean(array[0]);
    }
    else if (ndim() == 1) {
      bool* array = reinterpret_cast<bool*>(data());
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
      const std::vector<ssize_t> shape(std::next(shape_.begin()), shape_.end());
      const std::vector<ssize_t> strides(std::next(strides_.begin()), strides_.end());
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
                         format_,
                         dtype_,
                         ptr_lib_);
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
      T* array = reinterpret_cast<T*>(data());
      builder.integer((int64_t)array[0]);
    }
    else if (ndim() == 1) {
      T* array = reinterpret_cast<T*>(data());
      int64_t stride = strides_[0] / (int64_t)(sizeof(T));
      if (include_beginendlist) {
        builder.beginlist();
      }
      for (int64_t i = 0;  i < length();  i++) {
        builder.integer((int64_t)array[i*stride]);
      }
      if (include_beginendlist) {
        builder.endlist();
      }
    }
    else {
      const std::vector<ssize_t> shape(std::next(shape_.begin()), shape_.end());
      const std::vector<ssize_t> strides(std::next(strides_.begin()), strides_.end());
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
                         format_,
                         dtype_,
                         ptr_lib_);
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
      T* array = reinterpret_cast<T*>(data());
      builder.real(array[0]);
    }
    else if (ndim() == 1) {
      T* array = reinterpret_cast<T*>(data());
      int64_t stride = strides_[0] / (int64_t)(sizeof(T));
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
      const std::vector<ssize_t> shape(std::next(shape_.begin()), shape_.end());
      const std::vector<ssize_t> strides(std::next(strides_.begin()), strides_.end());
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
                         format_,
                         dtype_,
                         ptr_lib_);
        numpy.tojson_real<T>(builder, true);
      }
      builder.endlist();
    }
  }

  void
  NumpyArray::tojson_string(ToJson& builder,
                            bool include_beginendlist) const {
    if (ndim() == 0) {
      char* array = reinterpret_cast<char*>(data());
      builder.string(array, 1);
    }
    else if (ndim() == 1) {
      char* array = reinterpret_cast<char*>(data());
      builder.string(array, length());
    }
    else {
      const std::vector<ssize_t> shape(std::next(shape_.begin()), shape_.end());
      const std::vector<ssize_t> strides(std::next(strides_.begin()), strides_.end());
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
                         format_,
                         dtype_,
                         ptr_lib_);
        numpy.tojson_string(builder, true);
      }
      builder.endlist();
    }
  }

  const ContentPtr
  NumpyArray::copy_to(kernel::lib ptr_lib) const {
    if (ptr_lib == ptr_lib_) {
      return shallow_copy();
    }
    else {
      int64_t num_bytes = byteoffset_ + bytelength();
      std::shared_ptr<void> ptr = kernel::malloc<void>(ptr_lib, num_bytes);
      Error err = kernel::copy_to(ptr_lib,
                                  ptr_lib_,
                                  ptr.get(),
                                  ptr_.get(),
                                  num_bytes);
      util::handle_error(err);
      IdentitiesPtr identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->copy_to(ptr_lib);
      }
      return std::make_shared<NumpyArray>(identities,
                                          parameters_,
                                          ptr,
                                          shape_,
                                          strides_,
                                          byteoffset_,
                                          itemsize_,
                                          format_,
                                          dtype_,
                                          ptr_lib);
    }
  }

  const ContentPtr
  NumpyArray::numbers_to_type(const std::string& name) const {
    if (parameter_equals("__array__", "\"byte\"")) {
      return shallow_copy();
    }
    else if (parameter_equals("__array__", "\"char\"")) {
      return shallow_copy();
    }
    else {
      util::dtype dtype = util::name_to_dtype(name);
      NumpyArray contiguous_self = contiguous();

      ssize_t itemsize = util::dtype_to_itemsize(dtype);
      std::vector<ssize_t> shape = contiguous_self.shape();
      std::vector<ssize_t> strides;
      for (int64_t j = (int64_t)shape.size();  j > 0;  j--) {
        strides.insert(strides.begin(), itemsize);
        itemsize *= shape[(size_t)(j - 1)];
      }

      IdentitiesPtr identities = contiguous_self.identities();
      if (contiguous_self.identities().get() != nullptr) {
        identities = contiguous_self.identities().get()->deep_copy();
      }
      std::shared_ptr<void> ptr;
      switch (dtype_) {
      case util::dtype::boolean:
        ptr = as_type<bool>(reinterpret_cast<bool*>(contiguous_self.ptr().get()),
                            contiguous_self.length(),
                            dtype);
        break;
      case util::dtype::int8:
        ptr = as_type<int8_t>(reinterpret_cast<int8_t*>(contiguous_self.ptr().get()),
                              contiguous_self.length(),
                              dtype);
        break;
      case util::dtype::int16:
        ptr = as_type<int16_t>(reinterpret_cast<int16_t*>(contiguous_self.ptr().get()),
                               contiguous_self.length(),
                               dtype);
        break;
      case util::dtype::int32:
        ptr = as_type<int32_t>(reinterpret_cast<int32_t*>(contiguous_self.ptr().get()),
                               contiguous_self.length(),
                               dtype);
        break;
      case util::dtype::int64:
        ptr = as_type<int64_t>(reinterpret_cast<int64_t*>(contiguous_self.ptr().get()),
                               contiguous_self.length(),
                               dtype);
        break;
      case util::dtype::uint8:
        ptr = as_type<uint8_t>(reinterpret_cast<uint8_t*>(contiguous_self.ptr().get()),
                               contiguous_self.length(),
                               dtype);
        break;
      case util::dtype::uint16:
        ptr = as_type<uint16_t>(reinterpret_cast<uint16_t*>(contiguous_self.ptr().get()),
                                contiguous_self.length(),
                                dtype);
        break;
      case util::dtype::uint32:
        ptr = as_type<uint32_t>(reinterpret_cast<uint32_t*>(contiguous_self.ptr().get()),
                                contiguous_self.length(),
                                dtype);
        break;
      case util::dtype::uint64:
        ptr = as_type<uint64_t>(reinterpret_cast<uint64_t*>(contiguous_self.ptr().get()),
                                contiguous_self.length(),
                                dtype);
        break;
      case util::dtype::float16:
        throw std::runtime_error(
          std::string("FIXME: numbers_to_type for float16 not implemented")
          + FILENAME(__LINE__));
      case util::dtype::float32:
        ptr = as_type<float>(reinterpret_cast<float*>(contiguous_self.ptr().get()),
                             contiguous_self.length(),
                             dtype);
        break;
      case util::dtype::float64:
        ptr = as_type<double>(reinterpret_cast<double*>(contiguous_self.ptr().get()),
                              contiguous_self.length(),
                              dtype);
        break;
      case util::dtype::float128:
        throw std::runtime_error(
          std::string("FIXME: numbers_to_type for float128 not implemented")
          + FILENAME(__LINE__));
      case util::dtype::complex64:
        throw std::runtime_error(
          std::string("FIXME: values_astype for complex64 not implemented")
          + FILENAME(__LINE__));
      case util::dtype::complex128:
        throw std::runtime_error(
          std::string("FIXME: numbers_to_type for complex128 not implemented")
          + FILENAME(__LINE__));
      case util::dtype::complex256:
        throw std::runtime_error(
          std::string("FIXME: numbers_to_type for complex256 not implemented")
          + FILENAME(__LINE__));
      default:
        throw std::invalid_argument(
          std::string("cannot recast NumpyArray with format \"")
          + format_ + std::string("\"") + FILENAME(__LINE__));
      }

      return std::make_shared<NumpyArray>(identities,
                                          contiguous_self.parameters(),
                                          ptr,
                                          shape,
                                          strides,
                                          0,
                                          (ssize_t)util::dtype_to_itemsize(dtype),
                                          util::dtype_to_format(dtype),
                                          dtype,
                                          ptr_lib_);
    }
  }

  template<typename T>
  const std::shared_ptr<void>
  NumpyArray::index_sort(const T* data,
                         int64_t length,
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
      kernel::lib::cpu,   // DERIVE
      &ranges_length,
      parents.data(),
      parents.length());
    util::handle_error(err1, classname(), nullptr);

    Index64 outranges(ranges_length);
    struct Error err2 = kernel::sorting_ranges(
      kernel::lib::cpu,   // DERIVE
      outranges.data(),
      ranges_length,
      parents.data(),
      parents.length());
    util::handle_error(err2, classname(), nullptr);

    struct Error err3 = kernel::NumpyArray_argsort<T>(
      kernel::lib::cpu,   // DERIVE
      ptr.get(),
      data,
      length,
      outranges.data(),
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
      kernel::lib::cpu,   // DERIVE
      &ranges_length,
      parents.data(),
      parents.length());
    util::handle_error(err1, classname(), nullptr);

    Index64 outranges(ranges_length);
    struct Error err2 = kernel::sorting_ranges(
      kernel::lib::cpu,   // DERIVE
      outranges.data(),
      ranges_length,
      parents.data(),
      parents.length());
    util::handle_error(err2, classname(), nullptr);

    struct Error err3 = kernel::NumpyArray_sort<T>(
      kernel::lib::cpu,   // DERIVE
      ptr.get(),
      data,
      length,
      outranges.data(),
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
      kernel::lib::cpu,   // DERIVE
      ptr.get(),
      data,
      offsets.data(),
      offsets.length(),
      outoffsets.data(),
      ascending,
      stable);
    util::handle_error(err, classname(), nullptr);

    return ptr;
  }

  template<typename T>
  const std::shared_ptr<void>
  NumpyArray::as_type(const T* data,
                      int64_t length,
                      const util::dtype dtype) const {
    std::shared_ptr<void> ptr;
    switch (dtype) {
    case util::dtype::boolean:
      ptr = cast_to_type<bool>(data, length);
      break;
    case util::dtype::int8:
      ptr = cast_to_type<int8_t>(data, length);
      break;
    case util::dtype::int16:
      ptr = cast_to_type<int16_t>(data, length);
      break;
    case util::dtype::int32:
      ptr = cast_to_type<int32_t>(data, length);
      break;
    case util::dtype::int64:
      ptr = cast_to_type<int64_t>(data, length);
      break;
    case util::dtype::uint8:
      ptr = cast_to_type<uint8_t>(data, length);
      break;
    case util::dtype::uint16:
      ptr = cast_to_type<uint16_t>(data, length);
      break;
    case util::dtype::uint32:
      ptr = cast_to_type<uint32_t>(data, length);
      break;
    case util::dtype::uint64:
      ptr = cast_to_type<uint64_t>(data, length);
      break;
    case util::dtype::float16:
      throw std::runtime_error(
        std::string("FIXME: as_type for float16 not implemented")
        + FILENAME(__LINE__));
    case util::dtype::float32:
      ptr = cast_to_type<float>(data, length);
      break;
    case util::dtype::float64:
      ptr = cast_to_type<double>(data, length);
      break;
    case util::dtype::float128:
      throw std::runtime_error(
        std::string("FIXME: as_type for float128 not implemented")
        + FILENAME(__LINE__));
    case util::dtype::complex64:
      throw std::runtime_error(
        std::string("FIXME: as_type for complex64 not implemented")
        + FILENAME(__LINE__));
    case util::dtype::complex128:
      throw std::runtime_error(
        std::string("FIXME: as_type for complex128 not implemented")
        + FILENAME(__LINE__));
    case util::dtype::complex256:
      throw std::runtime_error(
        std::string("FIXME: as_type for complex256 not implemented")
        + FILENAME(__LINE__));
    default:
      throw std::invalid_argument(
        std::string("cannot recast NumpyArray with format \"")
        + format_ + std::string("\"") + FILENAME(__LINE__));
    }

    return ptr;
  }

  template<typename TO, typename FROM>
  const std::shared_ptr<void>
  NumpyArray::cast_to_type(const FROM* fromptr, int64_t length) const {
    std::shared_ptr<TO>toptr (
      new TO[length], kernel::array_deleter<TO>());

    struct Error err = kernel::NumpyArray_fill<FROM, TO>(
      kernel::lib::cpu,   // DERIVE
      toptr.get(),
      0,
      fromptr,
      length);
    util::handle_error(err, classname(), nullptr);

    return toptr;
  }
}
