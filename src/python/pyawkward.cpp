// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstdio>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "awkward/Index.h"
#include "awkward/Slice.h"
#include "awkward/Identities.h"
#include "awkward/Content.h"
#include "awkward/Iterator.h"
#include "awkward/array/None.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/Record.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/type/Type.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/type/RegularType.h"
#include "awkward/type/ListType.h"
#include "awkward/type/OptionType.h"
#include "awkward/type/UnionType.h"
#include "awkward/type/RecordType.h"
#include "awkward/io/json.h"
#include "awkward/io/root.h"

namespace py = pybind11;
namespace ak = awkward;

template<typename T>
class pyobject_deleter {
public:
  pyobject_deleter(PyObject *pyobj): pyobj_(pyobj) {
    Py_INCREF(pyobj_);
  }
  void operator()(T const *p) {
    Py_DECREF(pyobj_);
  }
private:
  PyObject* pyobj_;
};

py::object box(std::shared_ptr<ak::Type> t) {
  if (ak::ArrayType* raw = dynamic_cast<ak::ArrayType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListType* raw = dynamic_cast<ak::ListType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::OptionType* raw = dynamic_cast<ak::OptionType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::PrimitiveType* raw = dynamic_cast<ak::PrimitiveType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::RegularType* raw = dynamic_cast<ak::RegularType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionType* raw = dynamic_cast<ak::UnionType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::RecordType* raw = dynamic_cast<ak::RecordType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnknownType* raw = dynamic_cast<ak::UnknownType*>(t.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error("missing boxer for Type subtype");
  }
}

py::object box(std::shared_ptr<ak::Content> content) {
  if (ak::NumpyArray* raw = dynamic_cast<ak::NumpyArray*>(content.get())) {
    if (raw->isscalar()) {
      return py::array(py::buffer_info(
        raw->byteptr(),
        raw->itemsize(),
        raw->format(),
        raw->ndim(),
        raw->shape(),
        raw->strides()
      )).attr("item")();
    }
    else {
      return py::cast(*raw);
    }
  }
  else if (ak::None* raw = dynamic_cast<ak::None*>(content.get())) {
    return py::none();
  }
  else if (ak::ListArray32* raw = dynamic_cast<ak::ListArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListArrayU32* raw = dynamic_cast<ak::ListArrayU32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListArray64* raw = dynamic_cast<ak::ListArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArray32* raw = dynamic_cast<ak::ListOffsetArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArrayU32* raw = dynamic_cast<ak::ListOffsetArrayU32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArray64* raw = dynamic_cast<ak::ListOffsetArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::EmptyArray* raw = dynamic_cast<ak::EmptyArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::RegularArray* raw = dynamic_cast<ak::RegularArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::Record* raw = dynamic_cast<ak::Record*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::RecordArray* raw = dynamic_cast<ak::RecordArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedArray32* raw = dynamic_cast<ak::IndexedArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedArrayU32* raw = dynamic_cast<ak::IndexedArrayU32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedArray64* raw = dynamic_cast<ak::IndexedArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedOptionArray32* raw = dynamic_cast<ak::IndexedOptionArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedOptionArray64* raw = dynamic_cast<ak::IndexedOptionArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionArray8_32* raw = dynamic_cast<ak::UnionArray8_32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionArray8_U32* raw = dynamic_cast<ak::UnionArray8_U32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionArray8_64* raw = dynamic_cast<ak::UnionArray8_64*>(content.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error("missing boxer for Content subtype");
  }
}

py::object box(std::shared_ptr<ak::Identities> identities) {
  if (identities.get() == nullptr) {
    return py::none();
  }
  else if (ak::Identities32* raw = dynamic_cast<ak::Identities32*>(identities.get())) {
    return py::cast(*raw);
  }
  else if (ak::Identities64* raw = dynamic_cast<ak::Identities64*>(identities.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error("missing boxer for Identities subtype");
  }
}

std::shared_ptr<ak::Type> unbox_type(py::handle obj) {
  try {
    return obj.cast<ak::ArrayType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnknownType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::PrimitiveType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RegularType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::OptionType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnionType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RecordType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument("argument must be a Type subtype");
}

std::shared_ptr<ak::Type> unbox_type_none(py::handle obj) {
  if (obj.is(py::none())) {
    return ak::Type::none();
  }
  else {
    return unbox_type(obj);
  }
}

std::shared_ptr<ak::Content> unbox_content(py::handle obj) {
  try {
    return obj.cast<ak::NumpyArray*>()->shallow_copy();
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
    return obj.cast<ak::EmptyArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RegularArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    obj.cast<ak::Record*>();
    throw std::invalid_argument("content argument must be a Content subtype (excluding Record)");
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RecordArray*>()->shallow_copy();
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
  throw std::invalid_argument("content argument must be a Content subtype");
}

std::shared_ptr<ak::Identities> unbox_identities_none(py::handle obj) {
  if (obj.is(py::none())) {
    return ak::Identities::none();
  }
  try {
    return obj.cast<ak::Identities32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::Identities64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument("id argument must be an Identities subtype");
}

template <typename T>
std::string repr(T& self) {
  return self.tostring();
}

template <typename T>
int64_t len(T& self) {
  return self.length();
}

/////////////////////////////////////////////////////////////// Index

template <typename T>
py::class_<ak::IndexOf<T>> make_IndexOf(py::handle m, std::string name) {
  return (py::class_<ak::IndexOf<T>>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](ak::IndexOf<T>& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<void*>(reinterpret_cast<ssize_t>(self.ptr().get()) + self.offset()*sizeof(T)),
          sizeof(T),
          py::format_descriptor<T>::format(),
          1,
          { (ssize_t)self.length() },
          { (ssize_t)sizeof(T) });
        })

      .def(py::init([name](py::array_t<T, py::array::c_style | py::array::forcecast> array) -> ak::IndexOf<T> {
        py::buffer_info info = array.request();
        if (info.ndim != 1) {
          throw std::invalid_argument(name + std::string(" must be built from a one-dimensional array; try array.ravel()"));
        }
        if (info.strides[0] != sizeof(T)) {
          throw std::invalid_argument(name + std::string(" must be built from a contiguous array (array.strides == (array.itemsize,)); try array.copy()"));
        }
        return ak::IndexOf<T>(
          std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr), pyobject_deleter<T>(array.ptr())),
          0,
          (int64_t)info.shape[0]);
      }))

      .def("__repr__", &ak::IndexOf<T>::tostring)
      .def("__len__", &ak::IndexOf<T>::length)
      .def("__getitem__", &ak::IndexOf<T>::getitem_at)
      .def("__getitem__", &ak::IndexOf<T>::getitem_range)

  );
}

/////////////////////////////////////////////////////////////// Identities

template <typename T>
py::tuple identity(const T& self) {
  if (self.identities().get() == nullptr) {
    throw std::invalid_argument(self.classname() + std::string(" instance has no associated identities (use 'setidentities' to assign one to the array it is in)"));
  }
  ak::Identities::FieldLoc fieldloc = self.identities().get()->fieldloc();
  if (self.isscalar()) {
    py::tuple out((size_t)(self.identities().get()->width()) + fieldloc.size());
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
    py::tuple out((size_t)(self.identities().get()->width() - 1) + fieldloc.size());
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
py::object getidentities(T& self) {
  return box(self.identities());
}

template <typename T>
void setidentities(T& self, py::object identities) {
  self.setidentities(unbox_identities_none(identities));
}

template <typename T>
void setidentities_noarg(T& self) {
  self.setidentities();
}

template <typename T>
py::class_<ak::IdentitiesOf<T>> make_IdentitiesOf(py::handle m, std::string name) {
  return (py::class_<ak::IdentitiesOf<T>>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](ak::IdentitiesOf<T>& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<void*>(reinterpret_cast<ssize_t>(self.ptr().get()) + self.offset()*sizeof(T)),
          sizeof(T),
          py::format_descriptor<T>::format(),
          2,
          { (ssize_t)self.length(), (ssize_t)self.width() },
          { (ssize_t)(sizeof(T)*self.width()), (ssize_t)sizeof(T) });
        })

      .def_static("newref", &ak::Identities::newref)

      .def(py::init([](ak::Identities::Ref ref, ak::Identities::FieldLoc fieldloc, int64_t width, int64_t length) {
        return ak::IdentitiesOf<T>(ref, fieldloc, width, length);
      }))

      .def(py::init([name](ak::Identities::Ref ref, ak::Identities::FieldLoc fieldloc, py::array_t<T, py::array::c_style | py::array::forcecast> array) {
        py::buffer_info info = array.request();
        if (info.ndim != 2) {
          throw std::invalid_argument(name + std::string(" must be built from a two-dimensional array"));
        }
        if (info.strides[0] != sizeof(T)*info.shape[1]  ||  info.strides[1] != sizeof(T)) {
          throw std::invalid_argument(name + std::string(" must be built from a contiguous array (array.stries == (array.shape[1]*array.itemsize, array.itemsize)); try array.copy()"));
        }
        return ak::IdentitiesOf<T>(ref, fieldloc, 0, info.shape[1], info.shape[0],
            std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr), pyobject_deleter<T>(array.ptr())));
      }))

      .def("__repr__", &ak::IdentitiesOf<T>::tostring)
      .def("__len__", &ak::IdentitiesOf<T>::length)
      .def("__getitem__", &ak::IdentitiesOf<T>::getitem_at)
      .def("__getitem__", &ak::IdentitiesOf<T>::getitem_range)

      .def_property_readonly("ref", &ak::IdentitiesOf<T>::ref)
      .def_property_readonly("fieldloc", &ak::IdentitiesOf<T>::fieldloc)
      .def_property_readonly("width", &ak::IdentitiesOf<T>::width)
      .def_property_readonly("length", &ak::IdentitiesOf<T>::length)
      .def_property_readonly("array", [](py::buffer& self) -> py::array {
        return py::array(self);
      })
      .def("identity_at_str", &ak::IdentitiesOf<T>::identity_at)
      .def("identity_at", [](const ak::Identities& self, int64_t at) -> py::tuple {
        ak::Identities::FieldLoc fieldloc = self.fieldloc();
        py::tuple out((size_t)self.width() + fieldloc.size());
        size_t j = 0;
        for (int64_t i = 0;  i < self.width();  i++) {
          out[j] = py::cast(self.value(at, i));
          j++;
          for (auto pair : fieldloc) {
            if (pair.first == i) {
              out[j] = py::cast(pair.second);
              j++;
            }
          }
        }
        return out;
      })

  );
}

/////////////////////////////////////////////////////////////// Slice

bool handle_as_numpy(const std::shared_ptr<ak::Content>& content) {
  // if (content.get()->parameter_equals("__array__", "\"string\"")) {
  //   return true;
  // }
  if (ak::NumpyArray* raw = dynamic_cast<ak::NumpyArray*>(content.get())) {
    return true;
  }
  else if (ak::EmptyArray* raw = dynamic_cast<ak::EmptyArray*>(content.get())) {
    return true;
  }
  else if (ak::RegularArray* raw = dynamic_cast<ak::RegularArray*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::IndexedArray32* raw = dynamic_cast<ak::IndexedArray32*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::IndexedArrayU32* raw = dynamic_cast<ak::IndexedArrayU32*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::IndexedArray64* raw = dynamic_cast<ak::IndexedArray64*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::UnionArray8_32* raw = dynamic_cast<ak::UnionArray8_32*>(content.get())) {
    std::shared_ptr<ak::Content> first = raw->content(0);
    for (int64_t i = 1;  i < raw->numcontents();  i++) {
      if (!first.get()->mergeable(raw->content(i), false)) {
        return false;
      }
    }
    return handle_as_numpy(first);
  }
  else if (ak::UnionArray8_U32* raw = dynamic_cast<ak::UnionArray8_U32*>(content.get())) {
    std::shared_ptr<ak::Content> first = raw->content(0);
    for (int64_t i = 1;  i < raw->numcontents();  i++) {
      if (!first.get()->mergeable(raw->content(i), false)) {
        return false;
      }
    }
    return handle_as_numpy(first);
  }
  else if (ak::UnionArray8_64* raw = dynamic_cast<ak::UnionArray8_64*>(content.get())) {
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

void toslice_part(ak::Slice& slice, py::object obj) {
  int64_t length_before = slice.length();

  if (py::isinstance<py::int_>(obj)) {
    // FIXME: what happens if you give this a Numpy integer? a Numpy 0-dimensional array?
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
      throw std::invalid_argument("slice step must not be 0");
    }
    slice.append(std::make_shared<ak::SliceRange>(start, stop, step));
  }

#if PY_MAJOR_VERSION >= 3
  else if (py::isinstance<py::ellipsis>(obj)) {
    slice.append(std::make_shared<ak::SliceEllipsis>());
  }
#endif

  else if (obj.is(py::module::import("numpy").attr("newaxis"))) {
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

      if (py::isinstance(obj, py::module::import("numpy").attr("ma").attr("MaskedArray"))) {
        content = unbox_content(py::module::import("awkward1").attr("fromnumpy")(obj, false, false));
      }
      else if (py::isinstance(obj, py::module::import("numpy").attr("ndarray"))) {
        // content = nullptr!
      }
      else if (py::isinstance<ak::Content>(obj)) {
        content = unbox_content(obj);
      }
      else if (py::isinstance<ak::FillableArray>(obj)) {
        content = unbox_content(obj.attr("snapshot")());
      }
      else if (py::isinstance(obj, py::module::import("awkward1").attr("Array"))) {
        content = unbox_content(obj.attr("layout"));
      }
      else if (py::isinstance(obj, py::module::import("awkward1").attr("FillableArray"))) {
        content = unbox_content(obj.attr("snapshot")().attr("layout"));
      }
      else {
        bool bad = false;
        try {
          obj = py::module::import("numpy").attr("asarray")(obj);
        }
        catch (py::error_already_set& exc) {
          exc.restore();
          PyErr_Clear();
          bad = true;
        }
        if (!bad) {
          py::array array = obj.cast<py::array>();
          py::buffer_info info = array.request();
          if (info.format.compare("O") == 0) {
            bad = true;
          }
        }
        if (bad) {
          content = unbox_content(py::module::import("awkward1").attr("fromiter")(obj, false));
        }
      }

      if (content.get() != nullptr  &&  !handle_as_numpy(content)) {
        if (content.get()->parameter_equals("__array__", "\"string\"")) {
          obj = box(content);
          obj = py::module::import("awkward1").attr("tolist")(obj);
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
          throw std::invalid_argument("arrays used as an index must have at least one dimension");
        }

        py::buffer_info info = array.request();
        if (info.format.compare("?") == 0) {
          py::object nonzero_tuple = py::module::import("numpy").attr("nonzero")(array);
          for (auto x : nonzero_tuple.cast<py::tuple>()) {
            py::object intarray_object = py::module::import("numpy").attr("asarray")(x.cast<py::object>(), py::module::import("numpy").attr("int64"));
            py::array intarray = intarray_object.cast<py::array>();
            py::buffer_info intinfo = intarray.request();
            std::vector<int64_t> shape;
            std::vector<int64_t> strides;
            for (ssize_t i = 0;  i < intinfo.ndim;  i++) {
              shape.push_back((int64_t)intinfo.shape[i]);
              strides.push_back((int64_t)intinfo.strides[i] / sizeof(int64_t));
            }
            ak::Index64 index(std::shared_ptr<int64_t>(reinterpret_cast<int64_t*>(intinfo.ptr), pyobject_deleter<int64_t>(intarray.ptr())), 0, shape[0]);
            slice.append(std::make_shared<ak::SliceArray64>(index, shape, strides, true));
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
              format.compare("c") != 0       &&
              format.compare("b") != 0       &&
              format.compare("B") != 0       &&
              format.compare("h") != 0       &&
              format.compare("H") != 0       &&
              format.compare("i") != 0       &&
              format.compare("I") != 0       &&
              format.compare("l") != 0       &&
              format.compare("L") != 0       &&
              format.compare("q") != 0       &&
              format.compare("Q") != 0       &&
              flatlen != 0) {
            throw std::invalid_argument("arrays used as an index must be integer or boolean");
          }

          py::object intarray_object = py::module::import("numpy").attr("asarray")(array, py::module::import("numpy").attr("int64"));
          py::array intarray = intarray_object.cast<py::array>();
          py::buffer_info intinfo = intarray.request();
          std::vector<int64_t> shape;
          std::vector<int64_t> strides;
          for (ssize_t i = 0;  i < intinfo.ndim;  i++) {
            shape.push_back((int64_t)intinfo.shape[i]);
            strides.push_back((int64_t)intinfo.strides[i] / (int64_t)sizeof(int64_t));
          }
          ak::Index64 index(std::shared_ptr<int64_t>(reinterpret_cast<int64_t*>(intinfo.ptr), pyobject_deleter<int64_t>(intarray.ptr())), 0, shape[0]);
          slice.append(std::make_shared<ak::SliceArray64>(index, shape, strides, false));
        }
      }
    }
  }

  else {
    throw std::invalid_argument("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`), and integer or boolean arrays (possibly jagged) are valid indices");
  }
}

ak::Slice toslice(py::object obj) {
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

/////////////////////////////////////////////////////////////// Iterator

py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>> make_Iterator(py::handle m, std::string name) {
  auto next = [](ak::Iterator& iterator) -> py::object {
    if (iterator.isdone()) {
      throw py::stop_iteration();
    }
    return box(iterator.next());
  };

  return (py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>>(m, name.c_str())
      .def(py::init([](py::object content) -> ak::Iterator {
        return ak::Iterator(unbox_content(content));
      }))
      .def("__repr__", &ak::Iterator::tostring)
      .def("__next__", next)
      .def("next", next)
      .def("__iter__", [](py::object self) -> py::object { return self; })
  );
}

/////////////////////////////////////////////////////////////// FillableArray

template <typename T>
py::object getitem(T& self, py::object obj) {
  if (py::isinstance<py::int_>(obj)) {
    return box(self.getitem_at(obj.cast<int64_t>()));
  }
  if (py::isinstance<py::slice>(obj)) {
    py::object pystep = obj.attr("step");
    if ((py::isinstance<py::int_>(pystep)  &&  pystep.cast<int64_t>() == 1)  ||  pystep.is(py::none())) {
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
    // NOTE: control flow can pass through here; don't make the last line an 'else'!
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
    // NOTE: control flow can pass through here; don't make the last line an 'else'!
  }
  return box(self.getitem(toslice(obj)));
}

void fillable_fill(ak::FillableArray& self, py::handle obj) {
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
      fillable_fill(self, tup[i]);
    }
    self.endtuple();
  }
  else if (py::isinstance<py::dict>(obj)) {
    py::dict dict = obj.cast<py::dict>();
    self.beginrecord();
    for (auto pair : dict) {
      if (!py::isinstance<py::str>(pair.first)) {
        throw std::invalid_argument("keys of dicts in 'fromiter' must all be strings");
      }
      std::string key = pair.first.cast<std::string>();
      self.field_check(key.c_str());
      fillable_fill(self, pair.second);
    }
    self.endrecord();
  }
  else if (py::isinstance<py::iterable>(obj)) {
    py::iterable seq = obj.cast<py::iterable>();
    self.beginlist();
    for (auto x : seq) {
      fillable_fill(self, x);
    }
    self.endlist();
  }
  else {
    throw std::invalid_argument(std::string("cannot convert ") + obj.attr("__repr__")().cast<std::string>() + std::string(" to an array element"));
  }
}

py::class_<ak::FillableArray> make_FillableArray(py::handle m, std::string name) {
  return (py::class_<ak::FillableArray>(m, name.c_str())
      .def(py::init([](int64_t initial, double resize) -> ak::FillableArray {
        return ak::FillableArray(ak::FillableOptions(initial, resize));
      }), py::arg("initial") = 1024, py::arg("resize") = 2.0)
      .def_property_readonly("_ptr", [](ak::FillableArray* self) -> size_t { return reinterpret_cast<size_t>(self); })
      .def("__repr__", &ak::FillableArray::tostring)
      .def("__len__", &ak::FillableArray::length)
      .def("clear", &ak::FillableArray::clear)
      .def_property_readonly("type", &ak::FillableArray::type)
      .def("snapshot", [](ak::FillableArray& self) -> py::object {
        return box(self.snapshot());
      })
      .def("__getitem__", &getitem<ak::FillableArray>)
      .def("__iter__", [](ak::FillableArray& self) -> ak::Iterator {
        return ak::Iterator(self.snapshot());
      })
      .def("null", &ak::FillableArray::null)
      .def("boolean", &ak::FillableArray::boolean)
      .def("integer", &ak::FillableArray::integer)
      .def("real", &ak::FillableArray::real)
      .def("bytestring", [](ak::FillableArray& self, py::bytes x) -> void {
        self.bytestring(x.cast<std::string>());
      })
      .def("string", [](ak::FillableArray& self, py::str x) -> void {
        self.string(x.cast<std::string>());
      })
      .def("beginlist", &ak::FillableArray::beginlist)
      .def("endlist", &ak::FillableArray::endlist)
      .def("begintuple", &ak::FillableArray::begintuple)
      .def("index", &ak::FillableArray::index)
      .def("endtuple", &ak::FillableArray::endtuple)
      .def("beginrecord", [](ak::FillableArray& self, py::object name) -> void {
        if (name.is(py::none())) {
          self.beginrecord();
        }
        else {
          std::string cppname = name.cast<std::string>();
          self.beginrecord_check(cppname.c_str());
        }
      }, py::arg("name") = py::none())
      .def("field", [](ak::FillableArray& self, const std::string& x) -> void {
        self.field_check(x);
      })
      .def("endrecord", &ak::FillableArray::endrecord)
      .def("fill", &fillable_fill)
  );
}

/////////////////////////////////////////////////////////////// Type

ak::util::Parameters dict2parameters(py::object in) {
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
    throw std::invalid_argument("type parameters must be a dict (or None)");
  }
  return out;
}

py::dict parameters2dict(const ak::util::Parameters& in) {
  py::dict out;
  for (auto pair : in) {
    std::string cppkey = pair.first;
    std::string cppvalue = pair.second;
    py::str pykey(PyUnicode_DecodeUTF8(cppkey.data(), cppkey.length(), "surrogateescape"));
    py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
    out[pykey] = py::module::import("json").attr("loads")(pyvalue);
  }
  return out;
}

template <typename T>
py::dict getparameters(T& self) {
  return parameters2dict(self.parameters());
}

template <typename T>
void setparameters(T& self, py::object parameters) {
  self.setparameters(dict2parameters(parameters));
}

template <typename T>
void setparameter(T& self, std::string& key, py::object value) {
  py::object valuestr = py::module::import("json").attr("dumps")(value);
  self.setparameter(key, valuestr.cast<std::string>());
}

template <typename T>
py::object parameter(T& self, std::string& key) {
  std::string cppvalue = self.parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
py::object purelist_parameter(T& self, std::string& key) {
  std::string cppvalue = self.purelist_parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

py::class_<ak::Type, std::shared_ptr<ak::Type>> make_Type(py::handle m, std::string name) {
  return (py::class_<ak::Type, std::shared_ptr<ak::Type>>(m, name.c_str())
      .def("__eq__", [](std::shared_ptr<ak::Type> self, std::shared_ptr<ak::Type> other) -> bool {
        return self.get()->equal(other, true);
      })
      .def("__ne__", [](std::shared_ptr<ak::Type> self, std::shared_ptr<ak::Type> other) -> bool {
        return !self.get()->equal(other, true);
      })
  );
}

template <typename T>
py::class_<T, ak::Type> type_methods(py::class_<T, std::shared_ptr<T>, ak::Type>& x) {
  return x.def("__repr__", &T::tostring)
          .def_property("parameters", &getparameters<T>, &setparameters<T>)
          .def("setparameter", &setparameter<T>)
          .def_property_readonly("numfields", &T::numfields)
          .def("fieldindex", &T::fieldindex)
          .def("key", &T::key)
          .def("haskey", &T::haskey)
          .def("keys", &T::keys)
          .def("empty", &T::empty)
  ;
}

py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type> make_ArrayType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::Type> type, int64_t length, py::object parameters) -> ak::ArrayType {
        return ak::ArrayType(dict2parameters(parameters), type, length);
      }), py::arg("type"), py::arg("length"), py::arg("parameters") = py::none())
      .def_property_readonly("type", &ak::ArrayType::type)
      .def_property_readonly("length", &ak::ArrayType::length)
      .def(py::pickle([](const ak::ArrayType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), box(self.type()), py::cast(self.length()));
      }, [](py::tuple state) {
        return ak::ArrayType(dict2parameters(state[0]), unbox_type(state[1]), state[2].cast<int64_t>());
      }))
  );
}

py::class_<ak::UnknownType, std::shared_ptr<ak::UnknownType>, ak::Type> make_UnknownType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::UnknownType, std::shared_ptr<ak::UnknownType>, ak::Type>(m, name.c_str())
      .def(py::init([](py::object parameters) -> ak::UnknownType {
        return ak::UnknownType(dict2parameters(parameters));
      }), py::arg("parameters") = py::none())
      .def(py::pickle([](const ak::UnknownType& self) {
        return py::make_tuple(parameters2dict(self.parameters()));
      }, [](py::tuple state) {
        return ak::UnknownType(dict2parameters(state[0]));
      }))
  );
}

py::class_<ak::PrimitiveType, std::shared_ptr<ak::PrimitiveType>, ak::Type> make_PrimitiveType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::PrimitiveType, std::shared_ptr<ak::PrimitiveType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::string dtype, py::object parameters) -> ak::PrimitiveType {
        if (dtype == std::string("bool")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::boolean);
        }
        else if (dtype == std::string("int8")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::int8);
        }
        else if (dtype == std::string("int16")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::int16);
        }
        else if (dtype == std::string("int32")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::int32);
        }
        else if (dtype == std::string("int64")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::int64);
        }
        else if (dtype == std::string("uint8")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::uint8);
        }
        else if (dtype == std::string("uint16")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::uint16);
        }
        else if (dtype == std::string("uint32")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::uint32);
        }
        else if (dtype == std::string("uint64")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::uint64);
        }
        else if (dtype == std::string("float32")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::float32);
        }
        else if (dtype == std::string("float64")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::float64);
        }
        else {
          throw std::invalid_argument(std::string("unrecognized primitive type: ") + dtype);
        }
      }), py::arg("dtype"), py::arg("parameters") = py::none())
      .def_property_readonly("dtype", [](ak::PrimitiveType& self) -> std::string {
        switch (self.dtype()) {
          case ak::PrimitiveType::boolean: return std::string("bool");
          case ak::PrimitiveType::int8: return std::string("int8");
          case ak::PrimitiveType::int16: return std::string("int16");
          case ak::PrimitiveType::int32: return std::string("int32");
          case ak::PrimitiveType::int64: return std::string("int64");
          case ak::PrimitiveType::uint8: return std::string("uint8");
          case ak::PrimitiveType::uint16: return std::string("uint16");
          case ak::PrimitiveType::uint32: return std::string("uint32");
          case ak::PrimitiveType::uint64: return std::string("uint64");
          case ak::PrimitiveType::float32: return std::string("float32");
          case ak::PrimitiveType::float64: return std::string("float64");
          default:
          throw std::invalid_argument(std::string("unrecognized primitive type: ") + std::to_string(self.dtype()));
        }
      })
      .def(py::pickle([](const ak::PrimitiveType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), py::cast((int64_t)self.dtype()));
      }, [](py::tuple state) {
        return ak::PrimitiveType(dict2parameters(state[0]), (ak::PrimitiveType::DType)state[1].cast<int64_t>());
      }))
  );
}

py::class_<ak::RegularType, std::shared_ptr<ak::RegularType>, ak::Type> make_RegularType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::RegularType, std::shared_ptr<ak::RegularType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::Type> type, int64_t size, py::object parameters) -> ak::RegularType {
        return ak::RegularType(dict2parameters(parameters), type, size);
      }), py::arg("type"), py::arg("size"), py::arg("parameters") = py::none())
      .def_property_readonly("type", &ak::RegularType::type)
      .def_property_readonly("size", &ak::RegularType::size)
      .def(py::pickle([](const ak::RegularType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), box(self.type()), py::cast(self.size()));
      }, [](py::tuple state) {
        return ak::RegularType(dict2parameters(state[0]), unbox_type(state[1]), state[2].cast<int64_t>());
      }))
  );
}

py::class_<ak::ListType, std::shared_ptr<ak::ListType>, ak::Type> make_ListType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::ListType, std::shared_ptr<ak::ListType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::Type> type, py::object parameters) -> ak::ListType {
        return ak::ListType(dict2parameters(parameters), type);
      }), py::arg("type"), py::arg("parameters") = py::none())
      .def_property_readonly("type", &ak::ListType::type)
      .def(py::pickle([](const ak::ListType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), box(self.type()));
      }, [](py::tuple state) {
        return ak::ListType(dict2parameters(state[0]), unbox_type(state[1]));
      }))
  );
}

py::class_<ak::OptionType, std::shared_ptr<ak::OptionType>, ak::Type> make_OptionType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::OptionType, std::shared_ptr<ak::OptionType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::Type> type, py::object parameters) -> ak::OptionType {
        return ak::OptionType(dict2parameters(parameters), type);
      }), py::arg("type"), py::arg("parameters") = py::none())
      .def_property_readonly("type", &ak::OptionType::type)
      .def(py::pickle([](const ak::OptionType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), box(self.type()));
      }, [](py::tuple state) {
        return ak::OptionType(dict2parameters(state[0]), unbox_type(state[1]));
      }))
  );
}

py::class_<ak::UnionType, std::shared_ptr<ak::UnionType>, ak::Type> make_UnionType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::UnionType, std::shared_ptr<ak::UnionType>, ak::Type>(m, name.c_str())
      .def(py::init([](py::iterable types, py::object parameters) -> ak::UnionType {
        std::vector<std::shared_ptr<ak::Type>> out;
        for (auto x : types) {
          out.push_back(unbox_type(x));
        }
        return ak::UnionType(dict2parameters(parameters), out);
      }), py::arg("types"), py::arg("parameters") = py::none())
      .def_property_readonly("numtypes", &ak::UnionType::numtypes)
      .def_property_readonly("types", [](ak::UnionType& self) -> py::tuple {
        py::tuple types((size_t)self.numtypes());
        for (int64_t i = 0;  i < self.numtypes();  i++) {
          types[(size_t)i] = box(self.type(i));
        }
        return types;
      })
      .def("type", &ak::UnionType::type)
      .def(py::pickle([](const ak::UnionType& self) {
        py::tuple types((size_t)self.numtypes());
        for (int64_t i = 0;  i < self.numtypes();  i++) {
          types[(size_t)i] = box(self.type(i));
        }
        return py::make_tuple(parameters2dict(self.parameters()), types);
      }, [](py::tuple state) {
        std::vector<std::shared_ptr<ak::Type>> types;
        for (auto x : state[1]) {
          types.push_back(unbox_type(x));
        }
        return ak::UnionType(dict2parameters(state[0]), types);
      }))
  );
}

ak::RecordType iterable_to_RecordType(py::iterable types, py::object keys, py::object parameters) {
  std::vector<std::shared_ptr<ak::Type>> out;
  for (auto x : types) {
    out.push_back(unbox_type(x));
  }
  if (keys.is(py::none())) {
    return ak::RecordType(dict2parameters(parameters), out, std::shared_ptr<ak::util::RecordLookup>(nullptr));
  }
  else {
    std::shared_ptr<ak::util::RecordLookup> recordlookup = std::make_shared<ak::util::RecordLookup>();
    for (auto x : keys.cast<py::iterable>()) {
      recordlookup.get()->push_back(x.cast<std::string>());
    }
    if (out.size() != recordlookup.get()->size()) {
      throw std::invalid_argument("if provided, 'keys' must have the same length as 'types'");
    }
    return ak::RecordType(dict2parameters(parameters), out, recordlookup);
  }
}

py::class_<ak::RecordType, std::shared_ptr<ak::RecordType>, ak::Type> make_RecordType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::RecordType, std::shared_ptr<ak::RecordType>, ak::Type>(m, name.c_str())
      .def(py::init([](py::dict types, py::object parameters) -> ak::RecordType {
        std::shared_ptr<ak::util::RecordLookup> recordlookup = std::make_shared<ak::util::RecordLookup>();
        std::vector<std::shared_ptr<ak::Type>> out;
        for (auto x : types) {
          std::string key = x.first.cast<std::string>();
          recordlookup.get()->push_back(key);
          out.push_back(unbox_type(x.second));
        }
        return ak::RecordType(dict2parameters(parameters), out, recordlookup);
      }), py::arg("types"), py::arg("parameters") = py::none())
      .def(py::init(&iterable_to_RecordType), py::arg("types"), py::arg("keys") = py::none(), py::arg("parameters") = py::none())
      .def("__getitem__", [](ak::RecordType& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("__getitem__", [](ak::RecordType& self, std::string key) -> py::object {
        return box(self.field(key));
      })
      .def_property_readonly("istuple", &ak::RecordType::istuple)
      .def_property_readonly("types", [](ak::RecordType& self) -> py::object {
        std::vector<std::shared_ptr<ak::Type>> types = self.types();
        py::tuple pytypes(types.size());
        for (size_t i = 0;  i < types.size();  i++) {
          pytypes[i] = box(types[i]);
        }
        return pytypes;
      })
      .def("field", [](ak::RecordType& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("field", [](ak::RecordType& self, std::string key) -> py::object {
        return box(self.field(key));
      })
      .def("fields", [](ak::RecordType& self) -> py::object {
        py::list out;
        for (auto item : self.fields()) {
          out.append(box(item));
        }
        return out;
      })
      .def("fielditems", [](ak::RecordType& self) -> py::object {
        py::list out;
        for (auto item : self.fielditems()) {
          py::str key(item.first);
          py::object val(box(item.second));
          py::tuple pair(2);
          pair[0] = key;
          pair[1] = val;
          out.append(pair);
        }
        return out;
      })
      .def(py::pickle([](const ak::RecordType& self) {
        py::tuple pytypes((size_t)self.numfields());
        for (int64_t i = 0;  i < self.numfields();  i++) {
          pytypes[(size_t)i] = box(self.field(i));
        }
        std::shared_ptr<ak::util::RecordLookup> recordlookup = self.recordlookup();
        if (recordlookup.get() == nullptr) {
          return py::make_tuple(pytypes, py::none(), parameters2dict(self.parameters()));
        }
        else {
          py::tuple pyrecordlookup((size_t)self.numfields());
          for (size_t i = 0;  i < (size_t)self.numfields();  i++) {
            pyrecordlookup[i] = py::cast(recordlookup.get()->at(i));
          }
          return py::make_tuple(pytypes, pyrecordlookup, parameters2dict(self.parameters()));
        }
      }, [](py::tuple state) {
        return iterable_to_RecordType(state[0].cast<py::iterable>(), state[1], state[2]);
      }))
  );
}

/////////////////////////////////////////////////////////////// Content

template <typename T>
ak::Iterator iter(T& self) {
  return ak::Iterator(self.shallow_copy());
}

int64_t check_maxdecimals(py::object maxdecimals) {
  if (maxdecimals.is(py::none())) {
    return -1;
  }
  try {
    return maxdecimals.cast<int64_t>();
  }
  catch (py::cast_error err) {
    throw std::invalid_argument("maxdecimals must be None or an integer");
  }
}

template <typename T>
std::string tojson_string(T& self, bool pretty, py::object maxdecimals) {
  return self.tojson(pretty, check_maxdecimals(maxdecimals));
}

template <typename T>
void tojson_file(T& self, std::string destination, bool pretty, py::object maxdecimals, int64_t buffersize) {
#ifdef _MSC_VER
  FILE* file;
  if (fopen_s(&file, destination.c_str(), "wb") != 0) {
#else
  FILE* file = fopen(destination.c_str(), "wb");
  if (file == nullptr) {
#endif
    throw std::invalid_argument(std::string("file \"") + destination + std::string("\" could not be opened for writing"));
  }
  try {
    self.tojson(file, pretty, check_maxdecimals(maxdecimals), buffersize);
  }
  catch (...) {
    fclose(file);
    throw;
  }
  fclose(file);
}

template <typename T>
py::class_<T, std::shared_ptr<T>, ak::Content> content_methods(py::class_<T, std::shared_ptr<T>, ak::Content>& x) {
  return x.def("__repr__", &repr<T>)
          .def_property("identities", [](T& self) -> py::object { return box(self.identities()); }, [](T& self, py::object identities) -> void { self.setidentities(unbox_identities_none(identities)); })
          .def("setidentities", [](T& self, py::object identities) -> void {
           self.setidentities(unbox_identities_none(identities));
          })
          .def("setidentities", [](T& self) -> void {
            self.setidentities();
          })
          .def_property("parameters", &getparameters<T>, &setparameters<T>)
          .def("setparameter", &setparameter<T>)
          .def("parameter", &parameter<T>)
          .def("purelist_parameter", &purelist_parameter<T>)
          .def_property_readonly("type", [](T& self) -> py::object {
            return box(self.type());
          })
          .def("astype", [](T& self, std::shared_ptr<ak::Type>& type) -> py::object {
            return box(self.astype(type));
          })
          .def("__len__", &len<T>)
          .def("__getitem__", &getitem<T>)
          .def("__iter__", &iter<T>)
          .def("tojson", &tojson_string<T>, py::arg("pretty") = false, py::arg("maxdecimals") = py::none())
          .def("tojson", &tojson_file<T>, py::arg("destination"), py::arg("pretty") = false, py::arg("maxdecimals") = py::none(), py::arg("buffersize") = 65536)
          .def_property_readonly("nbytes", &T::nbytes)
          .def("deep_copy", &T::deep_copy, py::arg("copyarrays") = true, py::arg("copyindexes") = true, py::arg("copyidentities") = true)
          .def_property_readonly("identity", &identity<T>)
          .def_property_readonly("numfields", &T::numfields)
          .def("fieldindex", &T::fieldindex)
          .def("key", &T::key)
          .def("haskey", &T::haskey)
          .def("keys", &T::keys)
          .def_property_readonly("purelist_isregular", &T::purelist_isregular)
          .def_property_readonly("purelist_depth", &T::purelist_depth)
          .def("getitem_nothing", &T::getitem_nothing)

          // operations
          .def("count", [](T& self, int64_t axis) -> py::object {
            return box(self.count(axis));
          }, py::arg("axis") = 0)
          .def("flatten", [](T& self, int64_t axis) -> py::object {
            return box(self.flatten(axis));
          }, py::arg("axis") = 0)
          .def("mergeable", [](T& self, py::object other, bool mergebool) -> bool {
            return self.mergeable(unbox_content(other), mergebool);
          }, py::arg("other"), py::arg("mergebool") = false)
          .def("merge", [](T& self, py::object other) -> py::object {
            return box(self.merge(unbox_content(other)));
          })
          .def("merge_as_union", [](T& self, py::object other) -> py::object {
            return box(self.merge_as_union(unbox_content(other)));
          })

  ;
}

py::class_<ak::Content, std::shared_ptr<ak::Content>> make_Content(py::handle m, std::string name) {
  return py::class_<ak::Content, std::shared_ptr<ak::Content>>(m, name.c_str());
}

/////////////////////////////////////////////////////////////// NumpyArray

py::class_<ak::NumpyArray, std::shared_ptr<ak::NumpyArray>, ak::Content> make_NumpyArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::NumpyArray, std::shared_ptr<ak::NumpyArray>, ak::Content>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](ak::NumpyArray& self) -> py::buffer_info {
        return py::buffer_info(
          self.byteptr(),
          self.itemsize(),
          self.format(),
          self.ndim(),
          self.shape(),
          self.strides());
      })

      .def(py::init([](py::array array, py::object identities, py::object parameters) -> ak::NumpyArray {
        py::buffer_info info = array.request();
        if (info.ndim == 0) {
          throw std::invalid_argument("NumpyArray must not be scalar; try array.reshape(1)");
        }
        if (info.shape.size() != info.ndim  ||  info.strides.size() != info.ndim) {
          throw std::invalid_argument("NumpyArray len(shape) != ndim or len(strides) != ndim");
        }
        return ak::NumpyArray(unbox_identities_none(identities), dict2parameters(parameters), std::shared_ptr<void>(
          reinterpret_cast<void*>(info.ptr), pyobject_deleter<void>(array.ptr())),
          info.shape,
          info.strides,
          0,
          info.itemsize,
          info.format);
      }), py::arg("array"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("shape", &ak::NumpyArray::shape)
      .def_property_readonly("strides", &ak::NumpyArray::strides)
      .def_property_readonly("itemsize", &ak::NumpyArray::itemsize)
      .def_property_readonly("format", &ak::NumpyArray::format)
      .def_property_readonly("ndim", &ak::NumpyArray::ndim)
      .def_property_readonly("isscalar", &ak::NumpyArray::isscalar)
      .def_property_readonly("isempty", &ak::NumpyArray::isempty)
      .def("toRegularArray", &ak::NumpyArray::toRegularArray)

      .def_property_readonly("iscontiguous", &ak::NumpyArray::iscontiguous)
      .def("contiguous", &ak::NumpyArray::contiguous)
      .def("become_contiguous", &ak::NumpyArray::become_contiguous)
  );
}

/////////////////////////////////////////////////////////////// ListArray

template <typename T>
py::class_<ak::ListArrayOf<T>, std::shared_ptr<ak::ListArrayOf<T>>, ak::Content> make_ListArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::ListArrayOf<T>, std::shared_ptr<ak::ListArrayOf<T>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& starts, ak::IndexOf<T>& stops, py::object content, py::object identities, py::object parameters) -> ak::ListArrayOf<T> {
        return ak::ListArrayOf<T>(unbox_identities_none(identities), dict2parameters(parameters), starts, stops, unbox_content(content));
      }), py::arg("starts"), py::arg("stops"), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("starts", &ak::ListArrayOf<T>::starts)
      .def_property_readonly("stops", &ak::ListArrayOf<T>::stops)
      .def_property_readonly("content", &ak::ListArrayOf<T>::content)
      .def("compact_offsets64", &ak::ListArrayOf<T>::compact_offsets64)
      .def("broadcast_tooffsets64", &ak::ListArrayOf<T>::broadcast_tooffsets64)
      .def("toRegularArray", &ak::ListArrayOf<T>::toRegularArray)
  );
}

/////////////////////////////////////////////////////////////// ListOffsetArray

template <typename T>
py::class_<ak::ListOffsetArrayOf<T>, std::shared_ptr<ak::ListOffsetArrayOf<T>>, ak::Content> make_ListOffsetArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::ListOffsetArrayOf<T>, std::shared_ptr<ak::ListOffsetArrayOf<T>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& offsets, py::object content, py::object identities, py::object parameters) -> ak::ListOffsetArrayOf<T> {
        return ak::ListOffsetArrayOf<T>(unbox_identities_none(identities), dict2parameters(parameters), offsets, std::shared_ptr<ak::Content>(unbox_content(content)));
      }), py::arg("offsets"), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("starts", &ak::ListOffsetArrayOf<T>::starts)
      .def_property_readonly("stops", &ak::ListOffsetArrayOf<T>::stops)
      .def_property_readonly("offsets", &ak::ListOffsetArrayOf<T>::offsets)
      .def_property_readonly("content", &ak::ListOffsetArrayOf<T>::content)
      .def("compact_offsets64", &ak::ListOffsetArrayOf<T>::compact_offsets64)
      .def("broadcast_tooffsets64", &ak::ListOffsetArrayOf<T>::broadcast_tooffsets64)
      .def("toRegularArray", &ak::ListOffsetArrayOf<T>::toRegularArray)
  );
}

/////////////////////////////////////////////////////////////// EmptyArray

py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content> make_EmptyArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content>(m, name.c_str())
      .def(py::init([](py::object identities, py::object parameters) -> ak::EmptyArray {
        return ak::EmptyArray(unbox_identities_none(identities), dict2parameters(parameters));
      }), py::arg("identities") = py::none(), py::arg("parameters") = py::none())
  );
}

/////////////////////////////////////////////////////////////// RegularArray

py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content> make_RegularArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content>(m, name.c_str())
      .def(py::init([](py::object content, int64_t size, py::object identities, py::object parameters) -> ak::RegularArray {
        return ak::RegularArray(unbox_identities_none(identities), dict2parameters(parameters), std::shared_ptr<ak::Content>(unbox_content(content)), size);
      }), py::arg("content"), py::arg("size"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("size", &ak::RegularArray::size)
      .def_property_readonly("content", &ak::RegularArray::content)
      .def("compact_offsets64", &ak::RegularArray::compact_offsets64)
      .def("broadcast_tooffsets64", &ak::RegularArray::broadcast_tooffsets64)
  );
}

/////////////////////////////////////////////////////////////// RecordArray

ak::RecordArray iterable_to_RecordArray(py::iterable contents, py::object keys, py::object identities, py::object parameters) {
  std::vector<std::shared_ptr<ak::Content>> out;
  for (auto x : contents) {
    out.push_back(unbox_content(x));
  }
  if (out.empty()) {
    throw std::invalid_argument("construct RecordArrays without fields using RecordArray(length) where length is an integer");
  }
  if (keys.is(py::none())) {
    return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, std::shared_ptr<ak::util::RecordLookup>(nullptr));
  }
  else {
    std::shared_ptr<ak::util::RecordLookup> recordlookup = std::make_shared<ak::util::RecordLookup>();
    for (auto x : keys.cast<py::iterable>()) {
      recordlookup.get()->push_back(x.cast<std::string>());
    }
    if (out.size() != recordlookup.get()->size()) {
      throw std::invalid_argument("if provided, 'keys' must have the same length as 'types'");
    }
    return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, recordlookup);
  }
}

py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content> make_RecordArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content>(m, name.c_str())
      .def(py::init([](py::dict contents, py::object identities, py::object parameters) -> ak::RecordArray {
        std::shared_ptr<ak::util::RecordLookup> recordlookup = std::make_shared<ak::util::RecordLookup>();
        std::vector<std::shared_ptr<ak::Content>> out;
        for (auto x : contents) {
          std::string key = x.first.cast<std::string>();
          recordlookup.get()->push_back(key);
          out.push_back(unbox_content(x.second));
        }
        if (out.empty()) {
          throw std::invalid_argument("construct RecordArrays without fields using RecordArray(length) where length is an integer");
        }
        return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, recordlookup);
      }), py::arg("contents"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())
      .def(py::init(&iterable_to_RecordArray), py::arg("contents"), py::arg("keys") = py::none(), py::arg("identities") = py::none(), py::arg("parameters") = py::none())
      .def(py::init([](int64_t length, bool istuple, py::object identities, py::object parameters) -> ak::RecordArray {
        return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), length, istuple);
      }), py::arg("length"), py::arg("istuple") = false, py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("istuple", &ak::RecordArray::istuple)
      .def_property_readonly("contents", &ak::RecordArray::contents)
      .def("setitem_field", [](ak::RecordArray& self, py::object where, py::object what) -> py::object {
        std::shared_ptr<ak::Content> mywhat = unbox_content(what);
        if (where.is(py::none())) {
          return box(self.setitem_field(self.numfields(), mywhat));
        }
        else {
          try {
            std::string mywhere = where.cast<std::string>();
            return box(self.setitem_field(mywhere, mywhat));
          }
          catch (py::cast_error err) {
            try {
              int64_t mywhere = where.cast<int64_t>();
              return box(self.setitem_field(mywhere, mywhat));
            }
            catch (py::cast_error err) {
              throw std::invalid_argument("where must be None, int, or str");
            }
          }
        }
      }, py::arg("where"), py::arg("what"))

      .def("field", [](ak::RecordArray& self, int64_t fieldindex) -> std::shared_ptr<ak::Content> {
        return self.field(fieldindex);
      })
      .def("field", [](ak::RecordArray& self, std::string key) -> std::shared_ptr<ak::Content> {
        return self.field(key);
      })
      .def("fields", [](ak::RecordArray& self) -> py::object {
        py::list out;
        for (auto item : self.fields()) {
          out.append(box(item));
        }
        return out;
      })
      .def("fielditems", [](ak::RecordArray& self) -> py::object {
        py::list out;
        for (auto item : self.fielditems()) {
          py::str key(item.first);
          py::object val(box(item.second));
          py::tuple pair(2);
          pair[0] = key;
          pair[1] = val;
          out.append(pair);
        }
        return out;
      })
      .def_property_readonly("astuple", [](ak::RecordArray& self) -> py::object {
        return box(self.astuple());
      })

  );
}

py::class_<ak::Record, std::shared_ptr<ak::Record>> make_Record(py::handle m, std::string name) {
  return py::class_<ak::Record, std::shared_ptr<ak::Record>>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::RecordArray> recordarray, int64_t at) -> ak::Record {
        return ak::Record(recordarray, at);
      }), py::arg("recordarray"), py::arg("at"))
      .def("__repr__", &repr<ak::Record>)
      .def_property_readonly("identities", [](ak::Record& self) -> py::object { return box(self.identities()); })
      .def("__getitem__", &getitem<ak::Record>)
      .def_property_readonly("type", [](ak::Record& self) -> py::object {
        return box(self.type());
      })
      .def_property("parameters", &getparameters<ak::Record>, &setparameters<ak::Record>)
      .def("setparameter", &setparameter<ak::Record>)
      .def("parameter", &parameter<ak::Record>)
      .def("purelist_parameter", &purelist_parameter<ak::Record>)
      .def_property_readonly("type", [](ak::Record& self) -> py::object {
        return box(self.type());
      })
      .def("astype", [](ak::Record& self, std::shared_ptr<ak::Type>& type) -> py::object {
        return box(self.astype(type));
      })
      .def("tojson", &tojson_string<ak::Record>, py::arg("pretty") = false, py::arg("maxdecimals") = py::none())
      .def("tojson", &tojson_file<ak::Record>, py::arg("destination"), py::arg("pretty") = false, py::arg("maxdecimals") = py::none(), py::arg("buffersize") = 65536)

      .def_property_readonly("array", [](ak::Record& self) -> py::object { return box(self.array()); })
      .def_property_readonly("at", &ak::Record::at)
      .def_property_readonly("istuple", &ak::Record::istuple)
      .def_property_readonly("numfields", &ak::Record::numfields)
      .def("fieldindex", &ak::Record::fieldindex)
      .def("key", &ak::Record::key)
      .def("haskey", &ak::Record::haskey)
      .def("keys", &ak::Record::keys)
      .def("field", [](ak::Record& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("field", [](ak::Record& self, std::string key) -> py::object {
        return box(self.field(key));
      })
      .def("fields", [](ak::Record& self) -> py::object {
        py::list out;
        for (auto item : self.fields()) {
          out.append(box(item));
        }
        return out;
      })
      .def("fielditems", [](ak::Record& self) -> py::object {
        py::list out;
        for (auto item : self.fielditems()) {
          py::str key(item.first);
          py::object val(box(item.second));
          py::tuple pair(2);
          pair[0] = key;
          pair[1] = val;
          out.append(pair);
        }
        return out;
      })
      .def_property_readonly("astuple", [](ak::Record& self) -> py::object {
        return box(self.astuple());
      })
     .def_property_readonly("identity", &identity<ak::Record>)

  ;
}

/////////////////////////////////////////////////////////////// IndexedArray

template <typename T, bool ISOPTION>
py::class_<ak::IndexedArrayOf<T, ISOPTION>, std::shared_ptr<ak::IndexedArrayOf<T, ISOPTION>>, ak::Content> make_IndexedArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::IndexedArrayOf<T, ISOPTION>, std::shared_ptr<ak::IndexedArrayOf<T, ISOPTION>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& index, py::object content, py::object identities, py::object parameters) -> ak::IndexedArrayOf<T, ISOPTION> {
        return ak::IndexedArrayOf<T, ISOPTION>(unbox_identities_none(identities), dict2parameters(parameters), index, std::shared_ptr<ak::Content>(unbox_content(content)));
      }), py::arg("index"), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("index", &ak::IndexedArrayOf<T, ISOPTION>::index)
      .def_property_readonly("content", &ak::IndexedArrayOf<T, ISOPTION>::content)
      .def_property_readonly("isoption", &ak::IndexedArrayOf<T, ISOPTION>::isoption)
      .def("project", [](ak::IndexedArrayOf<T, ISOPTION>& self, py::object mask) {
        if (mask.is(py::none())) {
          return box(self.project());
        }
        else {
          return box(self.project(mask.cast<ak::Index8>()));
        }
      }, py::arg("mask") = py::none())
      .def("bytemask", &ak::IndexedArrayOf<T, ISOPTION>::bytemask)
      .def("simplify", [](ak::IndexedArrayOf<T, ISOPTION>& self) {
        return box(self.simplify());
      })
  );
}

/////////////////////////////////////////////////////////////// UnionArray

template <typename T, typename I>
py::class_<ak::UnionArrayOf<T, I>, std::shared_ptr<ak::UnionArrayOf<T, I>>, ak::Content> make_UnionArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::UnionArrayOf<T, I>, std::shared_ptr<ak::UnionArrayOf<T, I>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& tags, ak::IndexOf<I>& index, py::iterable contents, py::object identities, py::object parameters) -> ak::UnionArrayOf<T, I> {
        std::vector<std::shared_ptr<ak::Content>> out;
        for (auto content : contents) {
          out.push_back(std::shared_ptr<ak::Content>(unbox_content(content)));
        }
        return ak::UnionArrayOf<T, I>(unbox_identities_none(identities), dict2parameters(parameters), tags, index, out);
      }), py::arg("tags"), py::arg("index"), py::arg("contents"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_static("regular_index", &ak::UnionArrayOf<T, I>::regular_index)
      .def_property_readonly("tags", &ak::UnionArrayOf<T, I>::tags)
      .def_property_readonly("index", &ak::UnionArrayOf<T, I>::index)
      .def_property_readonly("contents", &ak::UnionArrayOf<T, I>::contents)
      .def_property_readonly("numcontents", &ak::UnionArrayOf<T, I>::numcontents)
      .def("content", &ak::UnionArrayOf<T, I>::content)
      .def("project", &ak::UnionArrayOf<T, I>::project)
      .def("simplify", [](ak::UnionArrayOf<T, I>& self, bool mergebool) -> py::object {
        return box(self.simplify(mergebool));
      }, py::arg("mergebool") = false)

  );
}

/////////////////////////////////////////////////////////////// module

PYBIND11_MODULE(layout, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  make_IndexOf<int8_t>(m,   "Index8");
  make_IndexOf<uint8_t>(m,  "IndexU8");
  make_IndexOf<int32_t>(m,  "Index32");
  make_IndexOf<uint32_t>(m, "IndexU32");
  make_IndexOf<int64_t>(m,  "Index64");

  make_IdentitiesOf<int32_t>(m, "Identities32");
  make_IdentitiesOf<int64_t>(m, "Identities64");

  make_Iterator(m, "Iterator");

  make_FillableArray(m, "FillableArray");

  make_Type(m, "Type");
  make_ArrayType(m, "ArrayType");
  make_PrimitiveType(m, "PrimitiveType");
  make_RegularType(m, "RegularType");
  make_UnknownType(m, "UnknownType");
  make_ListType(m, "ListType");
  make_OptionType(m, "OptionType");
  make_UnionType(m, "UnionType");
  make_RecordType(m, "RecordType");

  make_Content(m, "Content");

  make_NumpyArray(m, "NumpyArray");

  make_ListArrayOf<int32_t>(m,  "ListArray32");
  make_ListArrayOf<uint32_t>(m, "ListArrayU32");
  make_ListArrayOf<int64_t>(m,  "ListArray64");

  make_ListOffsetArrayOf<int32_t>(m,  "ListOffsetArray32");
  make_ListOffsetArrayOf<uint32_t>(m, "ListOffsetArrayU32");
  make_ListOffsetArrayOf<int64_t>(m,  "ListOffsetArray64");

  make_EmptyArray(m, "EmptyArray");

  make_RegularArray(m, "RegularArray");

  make_RecordArray(m, "RecordArray");
  make_Record(m, "Record");

  make_IndexedArrayOf<int32_t, false>(m,  "IndexedArray32");
  make_IndexedArrayOf<uint32_t, false>(m, "IndexedArrayU32");
  make_IndexedArrayOf<int64_t, false>(m,  "IndexedArray64");
  make_IndexedArrayOf<int32_t, true>(m,  "IndexedOptionArray32");
  make_IndexedArrayOf<int64_t, true>(m,  "IndexedOptionArray64");

  make_UnionArrayOf<int8_t, int32_t>(m,  "UnionArray8_32");
  make_UnionArrayOf<int8_t, uint32_t>(m, "UnionArray8_U32");
  make_UnionArrayOf<int8_t, int64_t>(m,  "UnionArray8_64");

  m.def("slice_tostring", [](py::object obj) -> std::string {
    return toslice(obj).tostring();
  });

  m.def("fromjson", [](std::string source, int64_t initial, double resize, int64_t buffersize) -> py::object {
    bool isarray = false;
    for (char const &x: source) {
      if (x != 9  &&  x != 10  &&  x != 13  &&  x != 32) {  // whitespace
        if (x == 91) {       // opening square bracket
          isarray = true;
        }
        break;
      }
    }
    if (isarray) {
      return box(ak::FromJsonString(source.c_str(), ak::FillableOptions(initial, resize)));
    }
    else {
#ifdef _MSC_VER
      FILE* file;
      if (fopen_s(&file, source.c_str(), "rb") != 0) {
#else
      FILE* file = fopen(source.c_str(), "rb");
      if (file == nullptr) {
#endif
        throw std::invalid_argument(std::string("file \"") + source + std::string("\" could not be opened for reading"));
      }
      std::shared_ptr<ak::Content> out(nullptr);
      try {
        out = FromJsonFile(file, ak::FillableOptions(initial, resize), buffersize);
      }
      catch (...) {
        fclose(file);
        throw;
      }
      fclose(file);
      return box(out);
    }
  }, py::arg("source"), py::arg("initial") = 1024, py::arg("resize") = 2.0, py::arg("buffersize") = 65536);

  m.def("fromroot_nestedvector", [](ak::Index64& byteoffsets, ak::NumpyArray& rawdata, int64_t depth, int64_t itemsize, std::string format, int64_t initial, double resize) -> py::object {
      return box(FromROOT_nestedvector(byteoffsets, rawdata, depth, itemsize, format, ak::FillableOptions(initial, resize)));
  }, py::arg("byteoffsets"), py::arg("rawdata"), py::arg("depth"), py::arg("itemsize"), py::arg("format"), py::arg("initial") = 1024, py::arg("resize") = 2.0);

}
