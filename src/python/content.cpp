// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/content.cpp", line)

#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/chrono.h>

#include "awkward/layoutbuilder/BitMaskedArrayBuilder.h"
#include "awkward/layoutbuilder/ByteMaskedArrayBuilder.h"
#include "awkward/layoutbuilder/EmptyArrayBuilder.h"
#include "awkward/layoutbuilder/IndexedArrayBuilder.h"
#include "awkward/layoutbuilder/IndexedOptionArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"
#include "awkward/layoutbuilder/ListArrayBuilder.h"
#include "awkward/layoutbuilder/ListOffsetArrayBuilder.h"
#include "awkward/layoutbuilder/NumpyArrayBuilder.h"
#include "awkward/layoutbuilder/RecordArrayBuilder.h"
#include "awkward/layoutbuilder/RegularArrayBuilder.h"
#include "awkward/layoutbuilder/UnionArrayBuilder.h"
#include "awkward/layoutbuilder/UnmaskedArrayBuilder.h"

#include "awkward/python/util.h"
#include "awkward/datetime_util.h"


#include "awkward/python/dlpack_util.h"

////////// boxing

py::object
box(const std::shared_ptr<ak::Content>& content) {
  // scalars
  if (ak::None* raw =
      dynamic_cast<ak::None*>(content.get())) {
    return py::none();
  }
  else if (ak::Record* raw =
           dynamic_cast<ak::Record*>(content.get())) {
    return py::cast(*raw);
  }
  // scalar or array
  else if (ak::NumpyArray* raw =
           dynamic_cast<ak::NumpyArray*>(content.get())) {
    if (raw->isscalar()) {
      switch (raw->dtype()) {
        case ak::util::dtype::boolean:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<bool*>(raw->data())));
        case ak::util::dtype::int8:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<int8_t*>(raw->data())));
        case ak::util::dtype::int16:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<int16_t*>(raw->data())));
        case ak::util::dtype::int32:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<int32_t*>(raw->data())));
        case ak::util::dtype::int64:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<int64_t*>(raw->data())));
        case ak::util::dtype::uint8:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<uint8_t*>(raw->data())));
        case ak::util::dtype::uint16:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<uint16_t*>(raw->data())));
        case ak::util::dtype::uint32:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<uint32_t*>(raw->data())));
        case ak::util::dtype::uint64:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<uint64_t*>(raw->data())));
        case ak::util::dtype::float32:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<float*>(raw->data())));
        case ak::util::dtype::float64:
          return py::cast(ak::kernel::NumpyArray_getitem_at0(
                   raw->ptr_lib(),
                   reinterpret_cast<double*>(raw->data())));
        case ak::util::dtype::datetime64:
          return py::module::import("numpy").attr("datetime64")(
                   reinterpret_cast<uint64_t*>(raw->data()),
                   ak::util::format_to_units(raw->format()));
        case ak::util::dtype::timedelta64:
          return py::module::import("numpy").attr("timedelta64")(
                   reinterpret_cast<uint64_t*>(raw->data()),
                   ak::util::format_to_units(raw->format()));
        default:
          if (raw->ptr_lib() == ak::kernel::lib::cuda) {
            throw std::runtime_error(
              std::string("not implemented: format ")
              + ak::util::quote(raw->format())
              + std::string(" in CUDA")
              + FILENAME(__LINE__));
          }
          else {
            return py::array(py::buffer_info(
              raw->data(),
              raw->itemsize(),
              raw->format(),
              raw->ndim(),
              raw->shape(),
              raw->strides()
            )).attr("item")();
          }
      }
    }
    else {
      return py::cast(*raw);
    }
  }
  // arrays
  else if (ak::EmptyArray* raw =
           dynamic_cast<ak::EmptyArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedArray32* raw =
           dynamic_cast<ak::IndexedArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedArrayU32* raw =
           dynamic_cast<ak::IndexedArrayU32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedArray64* raw =
           dynamic_cast<ak::IndexedArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedOptionArray32* raw =
           dynamic_cast<ak::IndexedOptionArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedOptionArray64* raw =
           dynamic_cast<ak::IndexedOptionArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ByteMaskedArray* raw =
           dynamic_cast<ak::ByteMaskedArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::BitMaskedArray* raw =
           dynamic_cast<ak::BitMaskedArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnmaskedArray* raw =
           dynamic_cast<ak::UnmaskedArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListArray32* raw =
           dynamic_cast<ak::ListArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListArrayU32* raw =
           dynamic_cast<ak::ListArrayU32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListArray64* raw =
           dynamic_cast<ak::ListArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArray32* raw =
           dynamic_cast<ak::ListOffsetArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArrayU32* raw =
           dynamic_cast<ak::ListOffsetArrayU32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArray64* raw =
           dynamic_cast<ak::ListOffsetArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::RecordArray* raw =
           dynamic_cast<ak::RecordArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::RegularArray* raw =
           dynamic_cast<ak::RegularArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionArray8_32* raw =
           dynamic_cast<ak::UnionArray8_32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionArray8_U32* raw =
           dynamic_cast<ak::UnionArray8_U32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionArray8_64* raw =
           dynamic_cast<ak::UnionArray8_64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::VirtualArray* raw =
           dynamic_cast<ak::VirtualArray*>(content.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error(
      std::string("missing boxer for Content subtype: ") +
      (content.get() == nullptr ? std::string("nullptr") : content.get()->classname()) +
      std::string(" ") + FILENAME(__LINE__));
  }
}

py::object
box(const std::shared_ptr<ak::Identities>& identities) {
  if (identities.get() == nullptr) {
    return py::none();
  }
  else if (ak::Identities32* raw =
           dynamic_cast<ak::Identities32*>(identities.get())) {
    return py::cast(*raw);
  }
  else if (ak::Identities64* raw =
           dynamic_cast<ak::Identities64*>(identities.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error(
      std::string("missing boxer for Identities subtype") + FILENAME(__LINE__));
  }
}

std::shared_ptr<ak::Content>
unbox_content(const py::handle& obj) {
  try {
    obj.cast<ak::Record*>();
    throw std::invalid_argument(
      std::string("content argument must be a Content subtype (excluding Record)")
      + FILENAME(__LINE__));
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::NumpyArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::EmptyArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedArray32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedArrayU32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedArray64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedOptionArray32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedOptionArray64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ByteMaskedArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::BitMaskedArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnmaskedArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListArray32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListArrayU32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListArray64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListOffsetArray32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListOffsetArrayU32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListOffsetArray64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RecordArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RegularArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnionArray8_32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnionArray8_U32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnionArray8_64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::VirtualArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument(
    std::string("content argument must be a Content subtype")
    + FILENAME(__LINE__));
}

std::shared_ptr<ak::Identities>
unbox_identities(const py::handle& obj) {
  try {
    return obj.cast<ak::Identities32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::Identities64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument(
    std::string("id argument must be an Identities subtype")
    + FILENAME(__LINE__));
}

std::shared_ptr<ak::Identities>
unbox_identities_none(const py::handle& obj) {
  if (obj.is(py::none())) {
    return ak::Identities::none();
  }
  else {
    return unbox_identities(obj);
  }
}

////////// slicing

bool handle_as_numpy(const std::shared_ptr<ak::Content>& content) {
  // if (content.get()->parameter_equals("__array__", "\"string\"")  ||
  //     content.get()->parameter_equals("__array__", "\"bytestring\"")) {
  //   return true;
  // }
  if (ak::NumpyArray* raw =
      dynamic_cast<ak::NumpyArray*>(content.get())) {
    return true;
  }
  else if (ak::EmptyArray* raw =
           dynamic_cast<ak::EmptyArray*>(content.get())) {
    return true;
  }
  else if (ak::RegularArray* raw =
           dynamic_cast<ak::RegularArray*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::IndexedArray32* raw =
           dynamic_cast<ak::IndexedArray32*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::IndexedArrayU32* raw =
           dynamic_cast<ak::IndexedArrayU32*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::IndexedArray64* raw =
           dynamic_cast<ak::IndexedArray64*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::UnionArray8_32* raw =
           dynamic_cast<ak::UnionArray8_32*>(content.get())) {
    std::shared_ptr<ak::Content> first = raw->content(0);
    for (int64_t i = 1;  i < raw->numcontents();  i++) {
      if (!first.get()->mergeable(raw->content(i), false)) {
        return false;
      }
    }
    return handle_as_numpy(first);
  }
  else if (ak::UnionArray8_U32* raw =
           dynamic_cast<ak::UnionArray8_U32*>(content.get())) {
    std::shared_ptr<ak::Content> first = raw->content(0);
    for (int64_t i = 1;  i < raw->numcontents();  i++) {
      if (!first.get()->mergeable(raw->content(i), false)) {
        return false;
      }
    }
    return handle_as_numpy(first);
  }
  else if (ak::UnionArray8_64* raw =
           dynamic_cast<ak::UnionArray8_64*>(content.get())) {
    std::shared_ptr<ak::Content> first = raw->content(0);
    for (int64_t i = 1;  i < raw->numcontents();  i++) {
      if (!first.get()->mergeable(raw->content(i), false)) {
        return false;
      }
    }
    return handle_as_numpy(first);
  }
  else {
    return false;
  }
}

void
toslice_part(ak::Slice& slice, py::object obj) {
  int64_t length_before = slice.length();

  if (py::hasattr(obj, "__index__")) {
    bool success = true;
    int64_t index;
    try {
      py::object py_index = obj.attr("__index__")();
      index = py_index.cast<int64_t>();
    }
    catch (py::error_already_set err) {
      success = false;
    }
    if (success) {
      slice.append(std::make_shared<ak::SliceAt>(index));
      return;
    }
  }

  if (py::isinstance<py::int_>(obj)) {
    slice.append(std::make_shared<ak::SliceAt>(obj.cast<int64_t>()));
  }

  else if (py::isinstance<py::slice>(obj)) {
    py::object pystart = obj.attr("start");
    py::object pystop = obj.attr("stop");
    py::object pystep = obj.attr("step");
    int64_t start = ak::Slice::none();
    int64_t stop = ak::Slice::none();
    int64_t step = 1;
    if (!pystart.is(py::none())) {
      start = pystart.cast<int64_t>();
    }
    if (!pystop.is(py::none())) {
      stop = pystop.cast<int64_t>();
    }
    if (!pystep.is(py::none())) {
      step = pystep.cast<int64_t>();
    }
    if (step == 0) {
      throw std::invalid_argument(
        std::string("slice step must not be 0") + FILENAME(__LINE__));
    }
    slice.append(std::make_shared<ak::SliceRange>(start, stop, step));
  }

  else if (py::isinstance<py::ellipsis>(obj)) {
    slice.append(std::make_shared<ak::SliceEllipsis>());
  }

  // NumPy on Manylinux1 doesn't pass the is comparison, but it is None
  else if (obj.is_none() || obj.is(py::module::import("numpy").attr("newaxis"))) {
    slice.append(std::make_shared<ak::SliceNewAxis>());
  }

  else if (py::isinstance<py::str>(obj)) {
    slice.append(std::make_shared<ak::SliceField>(obj.cast<std::string>()));
  }

  else if (py::isinstance<py::iterable>(obj)) {
    std::vector<std::string> strings;
    bool all_strings = true;
    for (auto x : obj) {
      if (py::isinstance<py::str>(x)) {
        strings.push_back(x.cast<std::string>());
      }
      else {
        all_strings = false;
        break;
      }
    }

    if (all_strings  &&  !strings.empty()) {
      slice.append(std::make_shared<ak::SliceFields>(strings));
    }
    else {
      std::shared_ptr<ak::Content> content(nullptr);

      if (py::isinstance(
            obj,
            py::module::import("numpy").attr("ma").attr("MaskedArray"))) {
        content = unbox_content(
            py::module::import("awkward").attr("from_numpy")(obj,
                                                             false,
                                                             false,
                                                             false));
      }
      else if (py::isinstance(
            obj, py::module::import("numpy").attr("ndarray"))) {
        // content = nullptr!
      }
      else if (py::isinstance<ak::Content>(obj)) {
        content = unbox_content(obj);
        if (ak::VirtualArray* raw
              = dynamic_cast<ak::VirtualArray*>(content.get())) {
          content = raw->array();
        }
      }
      else if (py::isinstance<ak::ArrayBuilder>(obj)) {
        content = unbox_content(obj.attr("snapshot")());
      }
      else if (py::isinstance(obj, py::module::import("awkward")
                                              .attr("Array"))) {
        py::object tmp = obj.attr("layout");
        if (py::isinstance(tmp, py::module::import("awkward")
                                           .attr("partition")
                                           .attr("PartitionedArray"))) {
          content = unbox_content(tmp.attr("toContent")());
          obj = box(content);
        }
        else {
          content = unbox_content(tmp);
        }
      }
      else if (py::isinstance(obj, py::module::import("awkward")
                                              .attr("ArrayBuilder"))) {
        content = unbox_content(obj.attr("snapshot")().attr("layout"));
      }
      else if (py::isinstance(obj, py::module::import("awkward")
                                              .attr("partition")
                                              .attr("PartitionedArray"))) {
        content = unbox_content(obj.attr("toContent")());
        obj = box(content);
      }
      else {
        obj = py::module::import("awkward").attr("from_iter")(obj, false);

        bool bad = false;
        py::object asarray;
        try {
          asarray = py::module::import("awkward").attr("to_numpy")(obj, false);
        }
        catch (py::error_already_set& exc) {
          exc.restore();
          PyErr_Clear();
          bad = true;
        }

        if (!bad) {
          py::array array = asarray.cast<py::array>();
          py::buffer_info info = array.request();
          if (ak::util::format_to_dtype(info.format, info.itemsize) ==
              ak::util::dtype::NOT_PRIMITIVE) {
            bad = true;
          }
        }

        if (bad) {
          content = unbox_content(obj);
        }
        else {
          obj = asarray;
        }
      }

      if (content.get() != nullptr  &&  !handle_as_numpy(content)) {
        if (content.get()->parameter_equals("__array__", "\"string\"")  ||
            content.get()->parameter_equals("__array__", "\"bytestring\"")) {
          obj = box(content);
          obj = py::module::import("awkward").attr("to_list")(obj);
          std::vector<std::string> strings;
          for (auto x : obj) {
            strings.push_back(x.cast<std::string>());
          }
          slice.append(std::make_shared<ak::SliceFields>(strings));
        }
        else {
          slice.append(content.get()->asslice());
        }
      }
      else {
        py::array array = obj.cast<py::array>();
        if (array.ndim() == 0) {
          throw std::invalid_argument(
            std::string("arrays used as an index must have at least one dimension")
            + FILENAME(__LINE__));
        }

        py::buffer_info info = array.request();
        if (info.format.compare("?") == 0) {
          py::object nonzero_tuple =
            py::module::import("numpy").attr("nonzero")(array);
          for (auto x : nonzero_tuple.cast<py::tuple>()) {
            py::object intarray_object =
              py::module::import("numpy").attr("asarray")(
                x.cast<py::object>(),
                py::module::import("numpy").attr("int64"));
            py::array intarray = intarray_object.cast<py::array>();
            py::buffer_info intinfo = intarray.request();
            std::vector<int64_t> shape;
            std::vector<int64_t> strides;
            for (ssize_t i = 0;  i < intinfo.ndim;  i++) {
              shape.push_back((int64_t)intinfo.shape[i]);
              strides.push_back((int64_t)intinfo.strides[i] / sizeof(int64_t));
            }
            ak::Index64 index(
              std::shared_ptr<int64_t>(
                reinterpret_cast<int64_t*>(intinfo.ptr),
                pyobject_deleter<int64_t>(intarray.ptr())),
              0,
              shape[0],
              ak::kernel::lib::cpu);
            slice.append(std::make_shared<ak::SliceArray64>(index,
                                                            shape,
                                                            strides,
                                                            true));
          }
        }

        else {
          ssize_t flatlen = 1;
          for (auto x : info.shape) {
            flatlen *= x;
          }
          std::string format(info.format);
          format.erase(0, format.find_first_not_of("@=<>!"));
          if (py::isinstance<py::array>(obj) &&
              !ak::util::is_integer(
                ak::util::format_to_dtype(format, (int64_t)info.itemsize))  &&
              flatlen != 0) {
            throw std::invalid_argument(
              std::string("arrays used as an index must be a (native-endian) integer or boolean")
              + FILENAME(__LINE__));
          }

          py::object intarray_object =
            py::module::import("numpy").attr("asarray")(
              array, py::module::import("numpy").attr("int64"));
          py::array intarray = intarray_object.cast<py::array>();
          py::buffer_info intinfo = intarray.request();
          std::vector<int64_t> shape;
          std::vector<int64_t> strides;
          for (ssize_t i = 0;  i < intinfo.ndim;  i++) {
            shape.push_back((int64_t)intinfo.shape[i]);
            strides.push_back(
              (int64_t)intinfo.strides[i] / (int64_t)sizeof(int64_t));
          }
          ak::Index64 index(
            std::shared_ptr<int64_t>(
              reinterpret_cast<int64_t*>(intinfo.ptr),
              pyobject_deleter<int64_t>(intarray.ptr())),
            0,
            shape[0],
            ak::kernel::lib::cpu);
          slice.append(std::make_shared<ak::SliceArray64>(index,
                                                          shape,
                                                          strides,
                                                          false));
        }
      }
    }
  }

  else {
    throw std::invalid_argument(
      std::string("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`), "
                  "and integer or boolean arrays (possibly jagged) are valid indices")
      + FILENAME(__LINE__));
  }
}

ak::Slice
toslice(py::object obj) {
  ak::Slice out;
  if (py::isinstance<py::tuple>(obj)) {
    for (auto x : obj.cast<py::tuple>()) {
      toslice_part(out, x.cast<py::object>());
    }
  }
  else {
    toslice_part(out, obj);
  }
  out.become_sealed();
  return out;
}

int64_t
check_maxdecimals(const py::object& maxdecimals) {
  if (maxdecimals.is(py::none())) {
    return -1;
  }
  try {
    return maxdecimals.cast<int64_t>();
  }
  catch (py::cast_error err) {
    throw std::invalid_argument(
      std::string("maxdecimals must be None or an integer")
      + FILENAME(__LINE__));
  }
}

template <typename T>
std::string
tojson_string(const T& self,
              bool pretty,
              const py::object& maxdecimals,
              const char* nan_string,
              const char* infinity_string,
              const char* minus_infinity_string,
              const char* complex_real_string,
              const char* complex_imag_string) {
  return self.tojson(pretty,
                     check_maxdecimals(maxdecimals),
                     nan_string,
                     infinity_string,
                     minus_infinity_string,
                     complex_real_string,
                     complex_imag_string);
}

template <typename T>
void
tojson_file(const T& self,
            const std::string& destination,
            bool pretty,
            py::object maxdecimals,
            int64_t buffersize,
            const char* nan_string,
            const char* infinity_string,
            const char* minus_infinity_string,
            const char* complex_real_string,
            const char* complex_imag_string) {
#ifdef _MSC_VER
  FILE* file;
  if (fopen_s(&file, destination.c_str(), "wb") != 0) {
#else
  FILE* file = fopen(destination.c_str(), "wb");
  if (file == nullptr) {
#endif
    throw std::invalid_argument(
      std::string("file \"") + destination
      + std::string("\" could not be opened for writing")
      + FILENAME(__LINE__));
  }
  try {
    self.tojson(file,
                pretty,
                check_maxdecimals(maxdecimals),
                buffersize,
                nan_string,
                infinity_string,
                minus_infinity_string,
                complex_real_string,
                complex_imag_string);
  }
  catch (...) {
    fclose(file);
    throw;
  }
  fclose(file);
}

template <typename T>
py::object
getitem(const T& self, const py::object& obj) {
  if (py::isinstance<py::int_>(obj)) {
    return box(self.getitem_at(obj.cast<int64_t>()));
  }
  if (py::isinstance<py::slice>(obj)) {
    py::object pystep = obj.attr("step");
    if ((py::isinstance<py::int_>(pystep)  &&  pystep.cast<int64_t>() == 1)  ||
        pystep.is(py::none())) {
      int64_t start = ak::Slice::none();
      int64_t stop = ak::Slice::none();
      py::object pystart = obj.attr("start");
      py::object pystop = obj.attr("stop");
      if (!pystart.is(py::none())) {
        start = pystart.cast<int64_t>();
      }
      if (!pystop.is(py::none())) {
        stop = pystop.cast<int64_t>();
      }
      return box(self.getitem_range(start, stop));
    }
    // control flow can pass through here; don't make the last line an 'else'!
  }
  if (py::isinstance<py::str>(obj)) {
    return box(self.getitem_field(obj.cast<std::string>()));
  }
  if (!py::isinstance<py::tuple>(obj)  &&  py::isinstance<py::iterable>(obj)) {
    std::vector<std::string> strings;
    bool all_strings = true;
    for (auto x : obj) {
      if (py::isinstance<py::str>(x)) {
        strings.push_back(x.cast<std::string>());
      }
      else {
        all_strings = false;
        break;
      }
    }
    if (all_strings  &&  !strings.empty()) {
      return box(self.getitem_fields(strings));
    }
    // control flow can pass through here; don't make the last line an 'else'!
  }
  return box(self.getitem(toslice(obj)));
}

////////// ArrayBuilder

bool
builder_fromiter_iscomplex(const py::handle& obj) {
#if PY_MAJOR_VERSION < 3
  return py::isinstance(obj, py::module::import("__builtin__").attr("complex"));
#else
  return py::isinstance(obj, py::module::import("builtins").attr("complex"));
#endif
}

void
builder_datetime(ak::ArrayBuilder& self, const py::handle& obj) {
  if (py::isinstance<py::str>(obj)) {
    auto date_time = py::module::import("numpy").attr("datetime64")(obj);
    auto ptr = date_time.attr("astype")(py::module::import("numpy").attr("int64"));
    auto units = py::str(py::module::import("numpy").attr("dtype")(date_time)).cast<std::string>();
    self.datetime(ptr.cast<int64_t>(), units);
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("datetime64"))) {
    auto ptr = obj.attr("astype")(py::module::import("numpy").attr("int64"));
    self.datetime(ptr.cast<int64_t>(), py::str(obj.attr("dtype")));
  }
  else {
    throw std::invalid_argument(
      std::string("cannot convert ")
      + obj.attr("__repr__")().cast<std::string>() + std::string(" (type ")
      + obj.attr("__class__").attr("__name__").cast<std::string>()
      + std::string(") to an array element") + FILENAME(__LINE__));
  }
}

void
builder_timedelta(ak::ArrayBuilder& self, const py::handle& obj) {
  if (py::isinstance<py::str>(obj)) {
    auto date_time = py::module::import("numpy").attr("timedelta64")(obj);
    auto ptr = date_time.attr("astype")(py::module::import("numpy").attr("int64"));
    auto units = py::str(py::module::import("numpy").attr("dtype")(date_time)).cast<std::string>();
    self.timedelta(ptr.cast<int64_t>(), units);
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("timedelta64"))) {
    auto ptr = obj.attr("astype")(py::module::import("numpy").attr("int64"));
    self.timedelta(ptr.cast<int64_t>(), py::str(obj.attr("dtype")));
  }
  else {
    throw std::invalid_argument(
      std::string("cannot convert ")
      + obj.attr("__repr__")().cast<std::string>() + std::string(" (type ")
      + obj.attr("__class__").attr("__name__").cast<std::string>()
      + std::string(") to an array element") + FILENAME(__LINE__));
  }
}

void
builder_fromiter(ak::ArrayBuilder& self, const py::handle& obj) {
  if (obj.is(py::none())) {
    self.null();
  }
  else if (py::isinstance<py::bool_>(obj)) {
    self.boolean(obj.cast<bool>());
  }
  else if (py::isinstance<py::int_>(obj)) {
    self.integer(obj.cast<int64_t>());
  }
  else if (py::isinstance<py::float_>(obj)) {
    self.real(obj.cast<double>());
  }
  else if (builder_fromiter_iscomplex(obj)) {
    self.complex(obj.cast<std::complex<double>>());
  }
  else if (py::isinstance<py::bytes>(obj)) {
    self.bytestring(obj.cast<std::string>());
  }
  else if (py::isinstance<py::str>(obj)) {
    self.string(obj.cast<std::string>());
  }
  else if (py::isinstance<py::tuple>(obj)) {
    py::tuple tup = obj.cast<py::tuple>();
    self.begintuple(tup.size());
    for (size_t i = 0;  i < tup.size();  i++) {
      self.index((int64_t)i);
      builder_fromiter(self, tup[i]);
    }
    self.endtuple();
  }
  else if (py::isinstance<py::dict>(obj)) {
    py::dict dict = obj.cast<py::dict>();
    self.beginrecord();
    for (auto pair : dict) {
      if (!py::isinstance<py::str>(pair.first)) {
        throw std::invalid_argument(
          std::string("keys of dicts in 'fromiter' must all be strings")
          + FILENAME(__LINE__));
      }
      std::string key = pair.first.cast<std::string>();
      self.field_check(key.c_str());
      builder_fromiter(self, pair.second);
    }
    self.endrecord();
  }
  else if (py::isinstance(obj, py::module::import("awkward").attr("Array"))) {
    builder_fromiter(self, obj.attr("to_list")());
  }
  else if (py::isinstance(obj, py::module::import("awkward").attr("Record"))) {
    builder_fromiter(self, obj.attr("to_list")());
  }
  else if (py::isinstance<py::iterable>(obj)) {
    py::iterable seq = obj.cast<py::iterable>();
    self.beginlist();
    for (auto x : seq) {
      builder_fromiter(self, x);
    }
    self.endlist();
  }
  else if (py::isinstance<py::array>(obj)) {
    builder_fromiter(self, obj.attr("tolist")());
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("datetime64"))) {
    builder_datetime(self, obj);
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("timedelta64"))) {
    builder_timedelta(self, obj);
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("bool_"))) {
    self.boolean(obj.cast<bool>());
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("integer"))) {
    self.integer(obj.cast<int64_t>());
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("floating"))) {
    self.real(obj.cast<double>());
  }
  else {

    throw std::invalid_argument(
      std::string("cannot convert ")
      + obj.attr("__repr__")().cast<std::string>() + std::string(" (type ")
      + obj.attr("__class__").attr("__name__").cast<std::string>()
      + std::string(") to an array element") + FILENAME(__LINE__));
  }
}

template <>
py::object
getitem<ak::ArrayBuilder>(const ak::ArrayBuilder& self, const py::object& obj) {
  if (py::isinstance<py::int_>(obj)) {
    return box(unbox_content(::builder_snapshot(self.builder())).get()->getitem_at(obj.cast<int64_t>()));
  }
  if (py::isinstance<py::slice>(obj)) {
    py::object pystep = obj.attr("step");
    if ((py::isinstance<py::int_>(pystep)  &&  pystep.cast<int64_t>() == 1)  ||
        pystep.is(py::none())) {
      int64_t start = ak::Slice::none();
      int64_t stop = ak::Slice::none();
      py::object pystart = obj.attr("start");
      py::object pystop = obj.attr("stop");
      if (!pystart.is(py::none())) {
        start = pystart.cast<int64_t>();
      }
      if (!pystop.is(py::none())) {
        stop = pystop.cast<int64_t>();
      }
      return box(unbox_content(::builder_snapshot(self.builder())).get()->getitem_range(start, stop));
    }
    // control flow can pass through here; don't make the last line an 'else'!
  }
  if (py::isinstance<py::str>(obj)) {
    return box(unbox_content(::builder_snapshot(self.builder())).get()->getitem_field(obj.cast<std::string>()));
  }
  if (!py::isinstance<py::tuple>(obj)  &&  py::isinstance<py::iterable>(obj)) {
    std::vector<std::string> strings;
    bool all_strings = true;
    for (auto x : obj) {
      if (py::isinstance<py::str>(x)) {
        strings.push_back(x.cast<std::string>());
      }
      else {
        all_strings = false;
        break;
      }
    }
    if (all_strings  &&  !strings.empty()) {
      return box(unbox_content(::builder_snapshot(self.builder())).get()->getitem_fields(strings));
    }
    // control flow can pass through here; don't make the last line an 'else'!
  }
  return box(unbox_content(::builder_snapshot(self.builder())).get()->getitem(toslice(obj)));
}

py::class_<ak::ArrayBuilder>
make_ArrayBuilder(const py::handle& m, const std::string& name) {
  return (py::class_<ak::ArrayBuilder>(m, name.c_str())
      .def(py::init([](const int64_t initial, double resize) -> ak::ArrayBuilder {
        return ak::ArrayBuilder({initial, resize});
      }), py::arg("initial") = 1024, py::arg("resize") = 1.5)
      .def_property_readonly("_ptr",
                             [](const ak::ArrayBuilder* self) -> size_t {
        return reinterpret_cast<size_t>(self);
      })
      .def("__len__", &ak::ArrayBuilder::length)
      .def("clear", &ak::ArrayBuilder::clear)
      .def("type", [](const ak::ArrayBuilder& self, const std::map<std::string, std::string>& typestrs) -> std::shared_ptr<ak::Type> {
        return unbox_content(::builder_snapshot(self.builder()))->type(typestrs);
      })
      .def("form", [](const ak::ArrayBuilder& self) -> py::object {
        ::EmptyBuffersContainer container;
        int64_t form_key_id = 0;
        return py::str(self.to_buffers(container, form_key_id));
      })
      .def("to_buffers", [](const ak::ArrayBuilder& self) -> py::object {
        ::NumpyBuffersContainer container;
        int64_t form_key_id = 0;
        std::string form = self.to_buffers(container, form_key_id);
        py::tuple out(3);
        out[0] = py::str(form);
        out[1] = py::int_(self.length());
        out[2] = container.container();
        return out;
      })
      .def("snapshot", [](const ak::ArrayBuilder& self) -> py::object {
        return ::builder_snapshot(self.builder());
      })
      .def("__getitem__", &getitem<ak::ArrayBuilder>)
      .def("__iter__", [](const ak::ArrayBuilder& self) -> ak::Iterator {
        return ak::Iterator(unbox_content(::builder_snapshot(self.builder())));
      })
      .def("null", &ak::ArrayBuilder::null)
      .def("boolean", &ak::ArrayBuilder::boolean)
      .def("integer", &ak::ArrayBuilder::integer)
      .def("real", &ak::ArrayBuilder::real)
      .def("complex", &ak::ArrayBuilder::complex)
      .def("datetime", &builder_datetime)
      .def("timedelta", &builder_timedelta)
      .def("bytestring",
           [](ak::ArrayBuilder& self, const py::bytes& x) -> void {
        self.bytestring(x.cast<std::string>());
      })
      .def("string", [](ak::ArrayBuilder& self, const py::str& x) -> void {
        self.string(x.cast<std::string>());
      })
      .def("beginlist", &ak::ArrayBuilder::beginlist)
      .def("endlist", &ak::ArrayBuilder::endlist)
      .def("begintuple", &ak::ArrayBuilder::begintuple)
      .def("index", &ak::ArrayBuilder::index)
      .def("endtuple", &ak::ArrayBuilder::endtuple)
      .def("beginrecord",
           [](ak::ArrayBuilder& self, const py::object& name) -> void {
        if (name.is(py::none())) {
          self.beginrecord();
        }
        else {
          std::string cppname = name.cast<std::string>();
          self.beginrecord_check(cppname.c_str());
        }
      }, py::arg("name") = py::none())
      .def("field", [](ak::ArrayBuilder& self, const std::string& x) -> void {
        self.field_check(x);
      })
      .def("endrecord", &ak::ArrayBuilder::endrecord)
      .def("fromiter", &builder_fromiter)
  );
}

////////// LayoutBuilder<T, I>

namespace {
  template <typename T, typename I>
  py::object
  layoutbuilder_snapshot(const ak::FormBuilderPtr<T, I> builder, const ak::ForthOutputBufferMap& outputs) {
    if (builder.get()->classname() == "BitMaskedArrayBuilder") {
      const std::shared_ptr<const ak::BitMaskedArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::BitMaskedArrayBuilder<T, I>>(builder);
      return ::layoutbuilder_snapshot(raw.get()->content(), outputs);
    }
    if (builder.get()->classname() == "ByteMaskedArrayBuilder") {
      const std::shared_ptr<const ak::ByteMaskedArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::ByteMaskedArrayBuilder<T, I>>(builder);
      return ::layoutbuilder_snapshot(raw.get()->content(), outputs);
    }
    if (builder.get()->classname() == "EmptyArrayBuilder") {
      const std::shared_ptr<const ak::EmptyArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::EmptyArrayBuilder<T, I>>(builder);
      return py::module::import("awkward").attr("layout").attr("EmptyArray")();
    }
    if (builder.get()->classname() == "IndexedArrayBuilder") {
      const std::shared_ptr<const ak::IndexedArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::IndexedArrayBuilder<T, I>>(builder);
      auto search = outputs.find(raw.get()->vm_output_data());
      if (search != outputs.end()) {
        if (raw.get()->form_index() == "int32") {
          return box(std::make_shared<ak::IndexedArray32>(
            ak::Identities::none(),
            raw.get()->form_parameters(),
            ak::Index32(std::static_pointer_cast<int32_t>(search->second.get()->ptr()),
                    0,
                    search->second.get()->len(),
                    ak::kernel::lib::cpu),
            unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
        }
        else if (raw.get()->form_index() == "uint32") {
          return box(std::make_shared<ak::IndexedArrayU32>(
            ak::Identities::none(),
            raw.get()->form_parameters(),
            ak::IndexU32(std::static_pointer_cast<uint32_t>(search->second.get()->ptr()),
                     0,
                     search->second.get()->len(),
                     ak::kernel::lib::cpu),
            unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
        }
        else if (raw.get()->form_index() == "int64") {
          return box(std::make_shared<ak::IndexedArray64>(
            ak::Identities::none(),
            raw.get()->form_parameters(),
            ak::Index64(std::static_pointer_cast<int64_t>(search->second.get()->ptr()),
                    0,
                    search->second.get()->len(),
                    ak::kernel::lib::cpu),
            unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
        }
        else {
          // "int8" or "uint8"
          throw std::invalid_argument(
              std::string("Snapshot of a ") + builder.get()->classname()
              + std::string(" index ") + raw.get()->form_index()
              + std::string(" is not supported yet. ")
              + FILENAME(__LINE__));
        }
      }
      throw std::invalid_argument(
          std::string("Snapshot of a ") + builder.get()->classname()
          + std::string(" needs an index ")
          + FILENAME(__LINE__));

    }
    if (builder.get()->classname() == "IndexedOptionArrayBuilder") {
      const std::shared_ptr<const ak::IndexedOptionArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::IndexedOptionArrayBuilder<T, I>>(builder);
      auto search = outputs.find(raw.get()->vm_output_data());
      if (search != outputs.end()) {
         if (raw.get()->form_index() == "int32") {
            return box(std::make_shared<ak::IndexedOptionArray32>(
              ak::Identities::none(),
              raw.get()->form_parameters(),
              ak::Index32(
                std::static_pointer_cast<int32_t>(search->second.get()->ptr()),
                1,
                search->second.get()->len() - 1,
                ak::kernel::lib::cpu),
              unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
          }
          else if (raw.get()->form_index() == "int64") {
            return box(std::make_shared<ak::IndexedOptionArray64>(
              ak::Identities::none(),
              raw.get()->form_parameters(),
              ak::Index64(
                std::static_pointer_cast<int64_t>(search->second.get()->ptr()),
                1,
                search->second.get()->len() - 1,
                ak::kernel::lib::cpu),
              unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
          }
          else {
            throw std::invalid_argument(
                std::string("Snapshot of a ") + builder.get()->classname()
                + std::string(" index ") + raw.get()->form_index()
                + std::string(" is not supported yet. ")
                + FILENAME(__LINE__));
          }
      }
      throw std::invalid_argument(
        std::string("Snapshot of a ") + builder.get()->classname()
        + std::string(" needs an index ")
        + FILENAME(__LINE__));

    }
    if (builder.get()->classname() == "ListArrayBuilder") {
      const std::shared_ptr<const ak::ListArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::ListArrayBuilder<T, I>>(builder);
      auto search = outputs.find(raw.get()->vm_output_data());
      if (search != outputs.end()) {
        if (raw.get()->form_starts() == "int32") {
          ak::Index32 offsets = search->second.get()->toIndex32();
          ak::Index32 starts = ak::util::make_starts(offsets);
          ak::Index32 stops = ak::util::make_stops(offsets);
          return box(std::make_shared<ak::ListArray32>(
            ak::Identities::none(),
            raw.get()->form_parameters(),
            starts,
            stops,
            unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
        }
        else if (raw.get()->form_starts() == "uint32") {
          ak::IndexU32 offsets = search->second.get()->toIndexU32();
          ak::IndexU32 starts = ak::util::make_starts(offsets);
          ak::IndexU32 stops = ak::util::make_stops(offsets);
          return box(std::make_shared<ak::ListArrayU32>(ak::Identities::none(),
            raw.get()->form_parameters(),
            starts,
            stops,
            unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
        }
        else if (raw.get()->form_starts() == "int64") {
          ak::Index64 offsets = search->second.get()->toIndex64();
          ak::Index64 starts = ak::util::make_starts(offsets);
          ak::Index64 stops = ak::util::make_stops(offsets);
          return box(std::make_shared<ak::ListArray64>(ak::Identities::none(),
            raw.get()->form_parameters(),
            starts,
            stops,
            unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
        }
        else {
          throw std::invalid_argument(
              std::string("Snapshot of a ") + builder.get()->classname()
              + std::string(" starts ") + raw.get()->form_starts()
              + std::string(" is not supported yet. ")
              + FILENAME(__LINE__));
        }
      }
      throw std::invalid_argument(
          std::string("Snapshot of a ") + builder.get()->classname()
          + std::string(" needs offsets")
          + FILENAME(__LINE__));

    }
    if (builder.get()->classname().rfind("ListOffsetArrayBuilder", 0) == 0) {
      const std::shared_ptr<const ak::ListOffsetArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::ListOffsetArrayBuilder<T, I>>(builder);
      auto search = outputs.find(raw.get()->vm_output_data());
      if (search != outputs.end()) {
        if (raw.get()->form_offsets() == "int32") {
          return box(std::make_shared<ak::ListOffsetArray32>(ak::Identities::none(),
                                                             raw.get()->form_parameters(),
                                                             search->second.get()->toIndex32(),
                                                             unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
        }
        else if (raw.get()->form_offsets() == "uint32") {
          return box(std::make_shared<ak::ListOffsetArrayU32>(ak::Identities::none(),
                                                              raw.get()->form_parameters(),
                                                              search->second.get()->toIndexU32(),
                                                              unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
        }
        else if (raw.get()->form_offsets() == "int64") {
          return box(std::make_shared<ak::ListOffsetArray64>(ak::Identities::none(),
                                                             raw.get()->form_parameters(),
                                                             search->second.get()->toIndex64(),
                                                             unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs))));
        }
        else {
          throw std::invalid_argument(
              std::string("Snapshot of a ") + builder.get()->classname()
              + std::string(" offsets ") + raw.get()->form_offsets()
              + std::string(" is not supported yet. ")
              + FILENAME(__LINE__));
        }
      }
      throw std::invalid_argument(
          std::string("Snapshot of a ") + builder.get()->classname()
          + std::string(" needs offsets")
          + FILENAME(__LINE__));

    }
    if (builder.get()->classname() == "NumpyArrayBuilder") {
      const std::shared_ptr<const ak::NumpyArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::NumpyArrayBuilder<T, I>>(builder);
      auto search = outputs.find(raw.get()->vm_output_data());
      if (search != outputs.end()) {
        auto dtype = awkward::util::name_to_dtype(raw.get()->form_primitive());
        std::vector<ssize_t> shape = { (ssize_t)search->second.get()->len() };
        std::vector<ssize_t> strides = { (ssize_t)awkward::util::dtype_to_itemsize(dtype) };

        return box(std::make_shared<ak::NumpyArray>(ak::Identities::none(),
                                                    raw.get()->form_parameters(),
                                                    search->second.get()->ptr(),
                                                    shape,
                                                    strides,
                                                    0,
                                                    strides[0],
                                                    ak::util::dtype_to_format(ak::util::name_to_dtype(raw.get()->form_primitive())), // FIXME
                                                    dtype,
                                                    ak::kernel::lib::cpu));
      }
      throw std::invalid_argument(
          std::string("Snapshot of a ") + builder.get()->classname()
          + std::string(" needs data")
          + FILENAME(__LINE__));

    } else if (builder.get()->classname() == "RecordArrayBuilder") {
      const std::shared_ptr<const ak::RecordArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::RecordArrayBuilder<T, I>>(builder);
      ak::ContentPtrVec contents;
      for (size_t i = 0;  i < raw.get()->contents().size();  i++) {
        contents.push_back(unbox_content(layoutbuilder_snapshot(raw.get()->contents()[i], outputs)));
      }
      return box(std::make_shared<ak::RecordArray>(ak::Identities::none(),
                                                   raw.get()->form_parameters(),
                                                   contents,
                                                   raw.get()->form_recordlookup()));

    }
    if (builder.get()->classname() == "RegularArrayBuilder") {
      const std::shared_ptr<const ak::RegularArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::RegularArrayBuilder<T, I>>(builder);
      ak::ContentPtr out;
      if(raw.get()->content() != nullptr) {
        out = std::make_shared<ak::RegularArray>(ak::Identities::none(),
                                                 raw.get()->form_parameters(),
                                                 unbox_content(layoutbuilder_snapshot(raw.get()->content(), outputs)),
                                                 raw.get()->form_size());
      }
      return box(out);

    }
    if (builder.get()->classname() == "UnionArrayBuilder") {
      const std::shared_ptr<const ak::UnionArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::UnionArrayBuilder<T, I>>(builder);
      auto search_tags = outputs.find(raw.get()->vm_output_tags());
      if (search_tags != outputs.end()) {
        ak::Index8 tags(std::static_pointer_cast<int8_t>(search_tags->second.get()->ptr()),
                                                         0,
                                                         search_tags->second.get()->len(),
                                                         ak::kernel::lib::cpu);

        ak::ContentPtrVec contents;
        for (auto content : raw.get()->contents()) {
          contents.push_back(unbox_content(layoutbuilder_snapshot(content, outputs)));
        }

        int64_t lentags = tags.length();

        if (raw.get()->form_index() == "int32") {
          ak::Index32 current(lentags);
          ak::Index32 outindex(lentags);
          struct Error err = ak::kernel::UnionArray_regular_index<int8_t, int32_t>(
            ak::kernel::lib::cpu,   // DERIVE
            outindex.data(),
            current.data(),
            lentags,
            tags.data(),
            lentags);
          ak::util::handle_error(err, "UnionArray", nullptr);

          return box(ak::UnionArray8_32(ak::Identities::none(),
                                        ak::util::Parameters(),
                                        tags,
                                        outindex,
                                        contents).simplify_uniontype(false, false));

        }
        else if (raw.get()->form_index() == "uint32") {
          ak::IndexU32 current(lentags);
          ak::IndexU32 outindex(lentags);
          struct Error err = ak::kernel::UnionArray_regular_index<int8_t, uint32_t>(
            ak::kernel::lib::cpu,   // DERIVE
            outindex.data(),
            current.data(),
            lentags,
            tags.data(),
            lentags);
          ak::util::handle_error(err, "UnionArray", nullptr);

          return box(ak::UnionArray8_U32(ak::Identities::none(),
                                         ak::util::Parameters(),
                                         tags,
                                         outindex,
                                         contents).simplify_uniontype(false, false));
        }
        else if (raw.get()->form_index() == "int64") {
          ak::Index64 current(lentags);
          ak::Index64 outindex(lentags);
          struct Error err = ak::kernel::UnionArray_regular_index<int8_t, int64_t>(
            ak::kernel::lib::cpu,   // DERIVE
            outindex.data(),
            current.data(),
            lentags,
            tags.data(),
            lentags);
          ak::util::handle_error(err, "UnionArray", nullptr);

          return box(ak::UnionArray8_64(ak::Identities::none(),
                                        ak::util::Parameters(),
                                        tags,
                                        outindex,
                                        contents).simplify_uniontype(false, false));
        }
      }
      throw std::invalid_argument(
          std::string("Snapshot of a ") + builder.get()->classname()
          + std::string(" needs tags and index ")
          + FILENAME(__LINE__));

    }
    if (builder.get()->classname() == "UnmaskedArrayBuilder") {
      // FIXME: how to define a mask? is it needed?
      const std::shared_ptr<const ak::UnmaskedArrayBuilder<T, I>> raw = std::dynamic_pointer_cast<const ak::UnmaskedArrayBuilder<T, I>>(builder);
      return layoutbuilder_snapshot(raw.get()->content(), outputs);

    }

    throw std::invalid_argument(std::string("unrecognized form builder") + FILENAME(__LINE__));
  }
}

template <>
py::object
getitem<ak::LayoutBuilder32>(const ak::LayoutBuilder32& self, const py::object& obj) {
  if (py::isinstance<py::int_>(obj)) {
    return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem_at(obj.cast<int64_t>()));
  }
  if (py::isinstance<py::slice>(obj)) {
    py::object pystep = obj.attr("step");
    if ((py::isinstance<py::int_>(pystep)  &&  pystep.cast<int64_t>() == 1)  ||
        pystep.is(py::none())) {
      int64_t start = ak::Slice::none();
      int64_t stop = ak::Slice::none();
      py::object pystart = obj.attr("start");
      py::object pystop = obj.attr("stop");
      if (!pystart.is(py::none())) {
        start = pystart.cast<int64_t>();
      }
      if (!pystop.is(py::none())) {
        stop = pystop.cast<int64_t>();
      }
      return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem_range(start, stop));
    }
    // control flow can pass through here; don't make the last line an 'else'!
  }
  if (py::isinstance<py::str>(obj)) {
    return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem_field(obj.cast<std::string>()));
  }
  if (!py::isinstance<py::tuple>(obj)  &&  py::isinstance<py::iterable>(obj)) {
    std::vector<std::string> strings;
    bool all_strings = true;
    for (auto x : obj) {
      if (py::isinstance<py::str>(x)) {
        strings.push_back(x.cast<std::string>());
      }
      else {
        all_strings = false;
        break;
      }
    }
    if (all_strings  &&  !strings.empty()) {
      return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem_fields(strings));
    }
    // control flow can pass through here; don't make the last line an 'else'!
  }
  return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem(toslice(obj)));
}

template <>
py::object
getitem<ak::LayoutBuilder64>(const ak::LayoutBuilder64& self, const py::object& obj) {
  if (py::isinstance<py::int_>(obj)) {
    return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem_at(obj.cast<int64_t>()));
  }
  if (py::isinstance<py::slice>(obj)) {
    py::object pystep = obj.attr("step");
    if ((py::isinstance<py::int_>(pystep)  &&  pystep.cast<int64_t>() == 1)  ||
        pystep.is(py::none())) {
      int64_t start = ak::Slice::none();
      int64_t stop = ak::Slice::none();
      py::object pystart = obj.attr("start");
      py::object pystop = obj.attr("stop");
      if (!pystart.is(py::none())) {
        start = pystart.cast<int64_t>();
      }
      if (!pystop.is(py::none())) {
        stop = pystop.cast<int64_t>();
      }
      return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem_range(start, stop));
    }
    // control flow can pass through here; don't make the last line an 'else'!
  }
  if (py::isinstance<py::str>(obj)) {
    return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem_field(obj.cast<std::string>()));
  }
  if (!py::isinstance<py::tuple>(obj)  &&  py::isinstance<py::iterable>(obj)) {
    std::vector<std::string> strings;
    bool all_strings = true;
    for (auto x : obj) {
      if (py::isinstance<py::str>(x)) {
        strings.push_back(x.cast<std::string>());
      }
      else {
        all_strings = false;
        break;
      }
    }
    if (all_strings  &&  !strings.empty()) {
      return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem_fields(strings));
    }
    // control flow can pass through here; don't make the last line an 'else'!
  }
  return box(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())).get()->getitem(toslice(obj)));
}

template <typename T, typename I>
py::class_<ak::LayoutBuilder<T, I>>
make_LayoutBuilder(const py::handle& m, const std::string& name) {
  return (py::class_<ak::LayoutBuilder<T, I>>(m, name.c_str())
      .def(py::init([](const std::string& form, const int64_t initial, double resize, bool vm_init) -> ak::LayoutBuilder<T, I> {
        return ak::LayoutBuilder<T, I>(form, initial, vm_init);
      }), py::arg("form"), py::arg("initial") = 8, py::arg("resize") = 1.5, py::arg("vm_init") = true)
      .def_property_readonly("_ptr",
                             [](const ak::LayoutBuilder<T, I>* self) -> size_t {
        return reinterpret_cast<size_t>(self);
      })
      .def("__len__", &ak::LayoutBuilder<T, I>::length)
      .def("type", [](const ak::LayoutBuilder<T, I>& self, const std::map<std::string, std::string>& typestrs) -> std::shared_ptr<ak::Type> {
        return unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs()))->type(typestrs);
      })
      .def("snapshot", [](const ak::LayoutBuilder<T, I>& self) -> py::object {
        return ::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs());
      })
      .def("__getitem__", &getitem<ak::LayoutBuilder<T, I>>)
      .def("__iter__", [](const ak::LayoutBuilder<T, I>& self) -> ak::Iterator {
        return ak::Iterator(unbox_content(::layoutbuilder_snapshot(self.builder(), self.vm().get()->outputs())));
      })
      .def("json_form", [](const ak::LayoutBuilder<T, I>& self) -> py::object {
        return py::str(self.json_form());
      })
      .def("form", [](const ak::LayoutBuilder<T, I>& self) -> py::object {
        ::EmptyBuffersContainer container;
        return py::str(self.to_buffers(container));
      })
      .def("to_buffers", [](const ak::LayoutBuilder<T, I>& self) -> py::object {
        ::NumpyBuffersContainer container;
        std::string form = self.to_buffers(container);
        py::tuple out(3);
        out[0] = py::str(form);
        out[1] = py::int_(self.length());
        out[2] = container.container();
        return out;
      })
      .def("null", &ak::LayoutBuilder<T, I>::null)
      .def("boolean", &ak::LayoutBuilder<T, I>::boolean)
      .def("int64", &ak::LayoutBuilder<T, I>::int64)
      .def("float64", &ak::LayoutBuilder<T, I>::float64)
      .def("complex", &ak::LayoutBuilder<T, I>::complex)
      .def("bytestring",
           [](ak::LayoutBuilder<T, I>& self, const py::bytes& x) -> void {
        self.bytestring(x.cast<std::string>());
      })
      .def("string", [](ak::LayoutBuilder<T, I>& self, const py::str& x) -> void {
        self.string(x.cast<std::string>());
      })
      .def("begin_list", &ak::LayoutBuilder<T, I>::begin_list)
      .def("end_list", &ak::LayoutBuilder<T, I>::end_list)
      .def("tag", [](ak::LayoutBuilder<T, I>& self, int64_t tag) -> void {
        self.tag(tag);
      })
      .def("debug_step",
           [](const ak::LayoutBuilder<T, I>& self) -> void {
        return self.debug_step();
      })
      .def("vm_source",
            [](ak::LayoutBuilder<T, I>& self) -> const std::string {
         return self.vm_source();
      })
      .def("connect",
           [](ak::LayoutBuilder<T, I>& self,
              const std::shared_ptr<ak::ForthMachineOf<T, I>>& vm) -> void {
        self.connect(vm);
      })
      .def("vm",
           [](ak::LayoutBuilder<T, I>& self) -> const std::shared_ptr<ak::ForthMachineOf<T, I>> {
        return self.vm();
      })
      .def("resume",
            [](ak::LayoutBuilder<T, I>& self) -> void {
         return self.resume();
      })
  );
}

template py::class_<ak::LayoutBuilder32>
make_LayoutBuilder(const py::handle& m, const std::string& name);

template py::class_<ak::LayoutBuilder64>
make_LayoutBuilder(const py::handle& m, const std::string& name);


////////// Iterator

py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>>
make_Iterator(const py::handle& m, const std::string& name) {
  auto next = [](ak::Iterator& iterator) -> py::object {
    if (iterator.isdone()) {
      throw py::stop_iteration();
    }
    return box(iterator.next());
  };

  return (py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>>(m,
                                                                  name.c_str())
      .def(py::init([](const py::object& content) -> ak::Iterator {
        return ak::Iterator(unbox_content(content));
      }))
      .def("__repr__", &ak::Iterator::tostring)
      .def("__next__", next)
      .def("next", next)
      .def("__iter__",
           [](const py::object& self) -> py::object { return self; })
  );
}

////////// Content

PersistentSharedPtr::PersistentSharedPtr(
  const std::shared_ptr<ak::Content>& ptr)
    : ptr_(ptr) { }

py::object
PersistentSharedPtr::layout() const {
  return box(ptr_);
}

size_t
PersistentSharedPtr::ptr() const {
  return reinterpret_cast<size_t>(&ptr_);
}

py::class_<PersistentSharedPtr>
make_PersistentSharedPtr(const py::handle& m, const std::string& name) {
  return py::class_<PersistentSharedPtr>(m, name.c_str())
             .def("layout", &PersistentSharedPtr::layout)
             .def("ptr", &PersistentSharedPtr::ptr);
}

py::class_<ak::Content, std::shared_ptr<ak::Content>>
make_Content(const py::handle& m, const std::string& name) {
  return py::class_<ak::Content, std::shared_ptr<ak::Content>>(m,
                                                               name.c_str())
             .def("axis_wrap_if_negative",
               &ak::Content::axis_wrap_if_negative,
               py::arg("axis"))
  ;
}

template <typename T>
py::tuple
identity(const T& self) {
  if (self.identities().get() == nullptr) {
    throw std::invalid_argument(
      self.classname()
      + std::string(" instance has no associated identities (use "
                    "'setidentities' to assign one to the array it is in)")
      + FILENAME(__LINE__));
  }
  ak::Identities::FieldLoc fieldloc = self.identities().get()->fieldloc();
  if (self.isscalar()) {
    py::tuple out((size_t)(self.identities().get()->width())
                  + fieldloc.size());
    size_t j = 0;
    for (int64_t i = 0;  i < self.identities().get()->width();  i++) {
      out[j] = py::cast(self.identities().get()->value(0, i));
      j++;
      for (auto pair : fieldloc) {
        if (pair.first == i) {
          out[j] = py::cast(pair.second);
          j++;
        }
      }
    }
    return out;
  }
  else {
    py::tuple out((size_t)(self.identities().get()->width() - 1)
                  + fieldloc.size());
    size_t j = 0;
    for (int64_t i = 0;  i < self.identities().get()->width();  i++) {
      if (i < self.identities().get()->width() - 1) {
        out[j] = py::cast(self.identities().get()->value(0, i));
        j++;
      }
      for (auto pair : fieldloc) {
        if (pair.first == i) {
          out[j] = py::cast(pair.second);
          j++;
        }
      }
    }
    return out;
  }
}

template <typename T>
ak::Iterator
iter(const T& self) {
  return ak::Iterator(self.shallow_copy());
}

ak::util::Parameters
dict2parameters(const py::object& in) {
  ak::util::Parameters out;
  if (in.is(py::none())) {
    // None is equivalent to an empty dict
  }
  else if (py::isinstance<py::dict>(in)) {
    for (auto pair : in.cast<py::dict>()) {
      std::string key = pair.first.cast<std::string>();
      py::object value = py::module::import("json").attr("dumps")(pair.second);
      out[key] = value.cast<std::string>();
    }
  }
  else {
    throw std::invalid_argument(
      std::string("type parameters must be a dict (or None)") + FILENAME(__LINE__));
  }
  return out;
}

py::dict
parameters2dict(const ak::util::Parameters& in) {
  py::dict out;
  for (auto pair : in) {
    std::string cppkey = pair.first;
    std::string cppvalue = pair.second;
    py::str pykey(PyUnicode_DecodeUTF8(cppkey.data(),
                                       cppkey.length(),
                                       "surrogateescape"));
    py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(),
                                         cppvalue.length(),
                                         "surrogateescape"));
    out[pykey] = py::module::import("json").attr("loads")(pyvalue);
  }
  return out;
}

template <typename T>
py::dict
getparameters(const T& self) {
  return parameters2dict(self.parameters());
}

template <typename T>
py::object
parameter(const T& self, const std::string& key) {
  std::string cppvalue = self.parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(),
                                       cppvalue.length(),
                                       "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
py::object
purelist_parameter(const T& self, const std::string& key) {
  std::string cppvalue = self.purelist_parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(),
                                       cppvalue.length(),
                                       "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
void
setparameters(T& self, const py::object& parameters) {
  self.setparameters(dict2parameters(parameters));
}

template <typename T>
void
setparameter(T& self, const std::string& key, const py::object& value) {
  py::object valuestr = py::module::import("json").attr("dumps")(value);
  self.setparameter(key, valuestr.cast<std::string>());
}

template <typename T>
py::object
withparameter(T& self, const std::string& key, const py::object& value) {
  py::object valuestr = py::module::import("json").attr("dumps")(value);
  ak::ContentPtr out = self.shallow_copy();
  out.get()->setparameter(key, valuestr.cast<std::string>());
  return box(out);
}

template <typename T>
py::class_<T, std::shared_ptr<T>, ak::Content>
content_methods(py::class_<T, std::shared_ptr<T>, ak::Content>& x) {
  return x.def("__repr__", &repr<T>)
          .def_property(
            "identities",
            [](const T& self) -> py::object {
            return box(self.identities());
          },
            [](T& self, const py::object& identities) -> void {
            self.setidentities(unbox_identities_none(identities));
          })
          .def("setidentities",
               [](T& self, const py::object& identities) -> void {
           self.setidentities(unbox_identities_none(identities));
          })
          .def("setidentities", [](T& self) -> void {
            self.setidentities();
          })
          .def_property("parameters", &getparameters<T>, &setparameters<T>)
          .def("setparameter", &setparameter<T>)
          .def("withparameter", &withparameter<T>)
          .def("parameter", &parameter<T>)
          .def("purelist_parameter", &purelist_parameter<T>)
          .def("type",
               [](const T& self,
                  const std::map<std::string, std::string>& typestrs)
               -> std::shared_ptr<ak::Type> {
            return self.type(typestrs);
          })
          .def_property_readonly("form", [](const T& self)
                                         -> std::shared_ptr<ak::Form> {
            return self.form(false);
          })
          .def("__len__", &len<T>)
          .def("__getitem__", &getitem<T>)
          .def("__iter__", &iter<T>)
          .def_property_readonly("kernels", [](const T& self) -> py::str {
            switch (self.kernels()) {
              case ak::kernel::lib::cpu:
                return py::str("cpu");
              case ak::kernel::lib::cuda:
                return py::str("cuda");
              default:
                return py::str("mixed");
            }
          })
          .def_property_readonly("caches", [](const T& self) -> py::list {
            std::vector<ak::ArrayCachePtr> out1;
            self.caches(out1);
            py::list out2(out1.size());
            for (size_t i = 0;  i < out1.size();  i++) {
              if (std::shared_ptr<PyArrayCache> ptr =
                  std::dynamic_pointer_cast<PyArrayCache>(out1[i])) {
                out2[i] = py::cast(ptr);
              }
              else {
                throw std::runtime_error(
                  std::string("VirtualArray's cache is not a PyArrayCache")
                  + FILENAME(__LINE__));
              }
            }
            return out2;
          })
          .def("tojson",
               &tojson_string<T>,
               py::arg("pretty") = false,
               py::arg("maxdecimals") = py::none(),
               py::arg("nan_string") = nullptr,
               py::arg("infinity_string") = nullptr,
               py::arg("minus_infinity_string") = nullptr,
               py::arg("complex_real_string") = nullptr,
               py::arg("complex_imag_string") = nullptr)
          .def("tojson",
               &tojson_file<T>,
               py::arg("destination"),
               py::arg("pretty") = false,
               py::arg("maxdecimals") = py::none(),
               py::arg("buffersize") = 65536,
               py::arg("nan_string") = nullptr,
               py::arg("infinity_string") = nullptr,
               py::arg("minus_infinity_string") = nullptr,
               py::arg("complex_real_string") = nullptr,
               py::arg("complex_imag_string") = nullptr)
          .def_property_readonly("nbytes", &T::nbytes)
          .def("deep_copy",
               &T::deep_copy,
               py::arg("copyarrays") = true,
               py::arg("copyindexes") = true,
               py::arg("copyidentities") = true)
          .def_property_readonly("identity", &identity<T>)
          .def_property_readonly("numfields", &T::numfields)
          .def("fieldindex", &T::fieldindex)
          .def("key", &T::key)
          .def("haskey", &T::haskey)
          .def("keys", &T::keys)
          .def_property_readonly("istuple", &T::istuple)
          .def_property_readonly("purelist_isregular", &T::purelist_isregular)
          .def_property_readonly("purelist_depth", &T::purelist_depth)
          .def_property_readonly("branch_depth", [](const T& self)
                                                 -> py::object {
            std::pair<bool, int64_t> branch_depth = self.branch_depth();
            py::tuple pair(2);
            pair[0] = branch_depth.first;
            pair[1] = branch_depth.second;
            return pair;
          })
          .def_property_readonly("minmax_depth", [](const T& self)
                                                 -> py::object {
            std::pair<int64_t, int64_t> minmax_depth = self.minmax_depth();
            py::tuple pair(2);
            pair[0] = minmax_depth.first;
            pair[1] = minmax_depth.second;
            return pair;
          })
          .def("getitem_nothing", &T::getitem_nothing)
          .def("getitem_at_nowrap", [](const T& self, int64_t at) -> py::object {
            return box(self.getitem_at_nowrap(at));
          })
          .def("getitem_range_nowrap", &T::getitem_range_nowrap)
          .def_property_readonly(
            "_persistent_shared_ptr",
            [](std::shared_ptr<ak::Content>& self) -> PersistentSharedPtr {
            return PersistentSharedPtr(self);
          })

          // operations
          .def("validityerror", [](const T& self) -> py::object {
            std::string out = self.validityerror(std::string("layout"));
            if (out.empty()) {
              return py::none();
            }
            else {
              py::str pyvalue(PyUnicode_DecodeUTF8(out.data(),
                                                   out.length(),
                                                   "surrogateescape"));
              return pyvalue;
            }
          })
          .def("fillna",
               [](const T&self, const py::object&  value) -> py::object {
            return box(self.fillna(unbox_content(value)));
          })
          .def("num", [](const T& self, int64_t axis) -> py::object {
            return box(self.num(axis, 0));
          }, py::arg("axis") = 1)
          .def("flatten", [](const T& self, int64_t axis) -> py::object {
            std::pair<ak::Index64, std::shared_ptr<ak::Content>> pair =
              self.offsets_and_flattened(axis, 0);
            return box(pair.second);
          }, py::arg("axis") = 1)
          .def("offsets_and_flatten",
               [](const T& self, int64_t axis) -> py::object {
            std::pair<ak::Index64, std::shared_ptr<ak::Content>> pair =
              self.offsets_and_flattened(axis, 0);
            return py::make_tuple(py::cast(pair.first), box(pair.second));
          }, py::arg("axis") = 1)
          .def("rpad",
               [](const T&self, int64_t length, int64_t axis) -> py::object {
            return box(self.rpad(length, axis, 0));
          })
          .def("rpad_and_clip",
               [](const T&self, int64_t length, int64_t axis) -> py::object {
            return box(self.rpad_and_clip(length, axis, 0));
          })
          .def("mergeable",
               [](const T& self, const py::object& other, bool mergebool)
               -> bool {
            return self.mergeable(unbox_content(other), mergebool);
          }, py::arg("other"), py::arg("mergebool") = false)
          .def("merge",
               [](const T& self, const py::object& other) -> py::object {
            return box(self.merge(unbox_content(other)));
          })
          .def("merge_as_union",
               [](const T& self, const py::object& other) -> py::object {
            return box(self.merge_as_union(unbox_content(other)));
          })
          .def("mergemany",   // FIXME: temporary!
               [](const T& self, const py::iterable& pyothers) -> py::object {
            ak::ContentPtrVec others;
            for (auto pyother : pyothers) {
              others.push_back(unbox_content(pyother));
            }
            return box(self.mergemany(others));
          })
          .def("axis_wrap_if_negative",
            [](const T& self, int64_t axis) {
              return self.axis_wrap_if_negative(axis);
          })
          .def("count",
               [](const T& self, int64_t axis, bool mask, bool keepdims)
               -> py::object {
            ak::ReducerCount reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1,
             py::arg("mask") = false,
             py::arg("keepdims") = false)
          .def("count_nonzero",
               [](const T& self, int64_t axis, bool mask, bool keepdims)
               -> py::object {
            ak::ReducerCountNonzero reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1,
             py::arg("mask") = false,
             py::arg("keepdims") = false)
          .def("sum",
               [](const T& self, int64_t axis, bool mask, bool keepdims)
               -> py::object {
            ak::ReducerSum reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1,
             py::arg("mask") = false,
               py::arg("keepdims") = false)
          .def("prod",
               [](const T& self, int64_t axis, bool mask, bool keepdims)
               -> py::object {
            ak::ReducerProd reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1,
             py::arg("mask") = false,
             py::arg("keepdims") = false)
          .def("any",
               [](const T& self, int64_t axis, bool mask, bool keepdims)
               -> py::object {
            ak::ReducerAny reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1,
             py::arg("mask") = false,
             py::arg("keepdims") = false)
          .def("all",
               [](const T& self, int64_t axis, bool mask, bool keepdims)
               -> py::object {
            ak::ReducerAll reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1,
             py::arg("mask") = false,
             py::arg("keepdims") = false)
          .def("min", [](const T& self,
                         int64_t axis,
                         bool mask,
                         bool keepdims,
                         const py::object& initial) -> py::object {
            if (initial.is(py::none())) {
              ak::ReducerMin reducer;
              return box(self.reduce(reducer, axis, mask, keepdims));
            }
            else {
              double initial_f64 = initial.cast<double>();
              uint64_t initial_u64 = (initial_f64 > 0 ? initial.cast<uint64_t>() : 0);
              int64_t initial_i64 = initial.cast<int64_t>();
              ak::ReducerMin reducer(initial_f64, initial_u64, initial_i64);
              return box(self.reduce(reducer, axis, mask, keepdims));
            }
          }, py::arg("axis") = -1,
             py::arg("mask") = true,
             py::arg("keepdims") = false,
             py::arg("initial") = py::none())
          .def("max", [](const T& self,
                         int64_t axis,
                         bool mask,
                         bool keepdims,
                         const py::object& initial) -> py::object {
            if (initial.is(py::none())) {
              ak::ReducerMax reducer;
              return box(self.reduce(reducer, axis, mask, keepdims));
            }
            else {
              double initial_f64 = initial.cast<double>();
              uint64_t initial_u64 = (initial_f64 > 0 ? initial.cast<uint64_t>() : 0);
              int64_t initial_i64 = initial.cast<int64_t>();
              ak::ReducerMax reducer(initial_f64, initial_u64, initial_i64);
              return box(self.reduce(reducer, axis, mask, keepdims));
            }
          }, py::arg("axis") = -1,
             py::arg("mask") = true,
             py::arg("keepdims") = false,
             py::arg("initial") = py::none())
          .def("argmin",
               [](const T& self, int64_t axis, bool mask, bool keepdims)
               -> py::object {
            ak::ReducerArgmin reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1,
             py::arg("mask") = true,
             py::arg("keepdims") = false)
          .def("argmax",
               [](const T& self, int64_t axis, bool mask, bool keepdims)
               -> py::object {
            ak::ReducerArgmax reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1,
             py::arg("mask") = true,
             py::arg("keepdims") = false)
          .def("localindex", [](const T& self, int64_t axis) -> py::object {
            return box(self.localindex(axis, 0));
          }, py::arg("axis") = 1)
          .def("combinations",
               [](const T& self,
                  int64_t n,
                  bool replacement,
                  py::object keys,
                  py::object parameters,
                  int64_t axis) -> py::object {
            std::shared_ptr<ak::util::RecordLookup> recordlookup(nullptr);
            if (!keys.is(py::none())) {
              recordlookup = std::make_shared<ak::util::RecordLookup>();
              for (auto x : keys.cast<py::iterable>()) {
                recordlookup.get()->push_back(x.cast<std::string>());
              }
              if (n != recordlookup.get()->size()) {
                throw std::invalid_argument(
                  std::string("if provided, the length of 'keys' must be 'n'")
                  + FILENAME(__LINE__));
              }
            }
            return box(self.combinations(n,
                                         replacement,
                                         recordlookup,
                                         dict2parameters(parameters),
                                         axis,
                                         0));
          }, py::arg("n"),
             py::arg("replacement") = false,
             py::arg("keys") = py::none(),
             py::arg("parameters") = py::none(),
             py::arg("axis") = 1)
          .def("sort",
               [](const T& self,
                  int64_t axis,
                  bool ascending,
                  bool stable) -> py::object {
               return box(self.sort(axis, ascending, stable));
          })
          .def("argsort",
               [](const T& self,
                  int64_t axis,
                  bool ascending,
                  bool stable) -> py::object {
               return box(self.argsort(axis, ascending, stable));
          })
          .def("numbers_to_type",
               [](const T& self,
                  const std::string& name) -> py::object {
               return box(self.numbers_to_type(name));
          })
          .def("is_unique",
               [](const T& self) -> bool {
               return self.is_unique();
          })
          .def("copy_to",
               [](const T& self, const std::string& ptr_lib) -> py::object {
               if (ptr_lib == "cpu") {
                 return box(self.copy_to(ak::kernel::lib::cpu));
               }
               else if (ptr_lib == "cuda") {
                 return box(self.copy_to(ak::kernel::lib::cuda));
               }
               else {
                 throw std::invalid_argument(
                   std::string("specify 'cpu' or 'cuda'") + FILENAME(__LINE__));
               }
          })
          .def("carry", &T::carry)
    ;
  }
