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
