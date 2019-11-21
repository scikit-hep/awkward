// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstdio>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "awkward/Index.h"
#include "awkward/Slice.h"
#include "awkward/Identity.h"
#include "awkward/Content.h"
#include "awkward/Iterator.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/RecordArray.h"
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
  else if (ak::RecordArray* raw = dynamic_cast<ak::RecordArray*>(content.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error("missing boxer for Content subtype");
  }
}

py::object box(std::shared_ptr<ak::Identity> id) {
  if (id.get() == nullptr) {
    return py::none();
  }
  else if (ak::Identity32* raw = dynamic_cast<ak::Identity32*>(id.get())) {
    return py::cast(*raw);
  }
  else if (ak::Identity64* raw = dynamic_cast<ak::Identity64*>(id.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error("missing boxer for Identity subtype");
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
  throw std::invalid_argument("argument must be a Type subtype");
}

std::shared_ptr<ak::Content> unbox_content(py::object obj) {
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
    return obj.cast<ak::RecordArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument("content argument must be a Content subtype");
}

std::shared_ptr<ak::Identity> unbox_id(py::object id) {
  if (id.is(py::none())) {
    return std::shared_ptr<ak::Identity>(nullptr);
  }
  try {
    return id.cast<ak::Identity32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return id.cast<ak::Identity64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument("id argument must be an Identity subtype");
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
          (T)info.shape[0]);
      }))

      .def("__repr__", &ak::IndexOf<T>::tostring)
      .def("__len__", &ak::IndexOf<T>::length)
      .def("__getitem__", &ak::IndexOf<T>::getitem_at)
      .def("__getitem__", &ak::IndexOf<T>::getitem_range)

  );
}

/////////////////////////////////////////////////////////////// Identity

template <typename T>
py::object getid(T& self) {
  return box(self.id());
}

template <typename T>
void setid(T& self, py::object id) {
  self.setid(unbox_id(id));
}

template <typename T>
void setid_noarg(T& self) {
  self.setid();
}

template <typename T>
py::class_<ak::IdentityOf<T>> make_IdentityOf(py::handle m, std::string name) {
  return (py::class_<ak::IdentityOf<T>>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](ak::IdentityOf<T>& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<void*>(reinterpret_cast<ssize_t>(self.ptr().get()) + self.offset()*sizeof(T)),
          sizeof(T),
          py::format_descriptor<T>::format(),
          2,
          { (ssize_t)self.length(), (ssize_t)self.width() },
          { (ssize_t)(sizeof(T)*self.width()), (ssize_t)sizeof(T) });
        })

      .def_static("newref", &ak::Identity::newref)

      .def(py::init([](ak::Identity::Ref ref, ak::Identity::FieldLoc fieldloc, int64_t width, int64_t length) {
        return ak::IdentityOf<T>(ref, fieldloc, width, length);
      }))

      .def(py::init([name](ak::Identity::Ref ref, ak::Identity::FieldLoc fieldloc, py::array_t<T, py::array::c_style | py::array::forcecast> array) {
        py::buffer_info info = array.request();
        if (info.ndim != 2) {
          throw std::invalid_argument(name + std::string(" must be built from a two-dimensional array"));
        }
        if (info.strides[0] != sizeof(T)*info.shape[1]  ||  info.strides[1] != sizeof(T)) {
          throw std::invalid_argument(name + std::string(" must be built from a contiguous array (array.stries == (array.shape[1]*array.itemsize, array.itemsize)); try array.copy()"));
        }
        return ak::IdentityOf<T>(ref, fieldloc, 0, info.shape[1], info.shape[0],
            std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr), pyobject_deleter<T>(array.ptr())));
      }))

      .def("__repr__", &ak::IdentityOf<T>::tostring)
      .def("__len__", &ak::IdentityOf<T>::length)
      .def("__getitem__", &ak::IdentityOf<T>::getitem_at)
      .def("__getitem__", &ak::IdentityOf<T>::getitem_range)

      .def_property_readonly("ref", &ak::IdentityOf<T>::ref)
      .def_property_readonly("fieldloc", &ak::IdentityOf<T>::fieldloc)
      .def_property_readonly("width", &ak::IdentityOf<T>::width)
      .def_property_readonly("length", &ak::IdentityOf<T>::length)
      .def_property_readonly("array", [](py::buffer& self) -> py::array {
        return py::array(self);
      })

  );
}

/////////////////////////////////////////////////////////////// Slice

void toslice_part(ak::Slice& slice, py::object obj) {
  if (py::isinstance<py::int_>(obj)) {
    // FIXME: what happens if you give this a Numpy integer? a Numpy 0-dimensional array?
    slice.append(std::shared_ptr<ak::SliceItem>(new ak::SliceAt(obj.cast<int64_t>())));
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
    slice.append(std::shared_ptr<ak::SliceItem>(new ak::SliceRange(start, stop, step)));
  }
#if PY_MAJOR_VERSION >= 3
  else if (py::isinstance<py::ellipsis>(obj)) {
    slice.append(std::shared_ptr<ak::SliceItem>(new ak::SliceEllipsis()));
  }
#endif
  else if (obj.is(py::module::import("numpy").attr("newaxis"))) {
    slice.append(std::shared_ptr<ak::SliceItem>(new ak::SliceNewAxis()));
  }
  else if (py::isinstance<py::iterable>(obj)) {
    py::object objarray = py::module::import("numpy").attr("asarray")(obj);
    if (!py::isinstance<py::array>(objarray)) {
      throw std::invalid_argument("iterable cannot be cast as an array");
    }
    py::array array = objarray.cast<py::array>();
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
        slice.append(std::shared_ptr<ak::SliceItem>(new ak::SliceArray64(index, shape, strides)));
      }
    }

    else {
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
          format.compare("Q") != 0) {
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
      slice.append(std::shared_ptr<ak::SliceItem>(new ak::SliceArray64(index, shape, strides)));
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

py::class_<ak::Slice> make_Slice(py::handle m, std::string name) {
  return (py::class_<ak::Slice>(m, name.c_str())
      .def(py::init([](py::object obj) {
        return toslice(obj);
      }))

      .def("__repr__", [](ak::Slice& self) -> const std::string {
        return self.tostring();
      })
  );
}

/////////////////////////////////////////////////////////////// Iterator

py::class_<ak::Iterator> make_Iterator(py::handle m, std::string name) {
  auto next = [](ak::Iterator& iterator) -> py::object {
    if (iterator.isdone()) {
      throw py::stop_iteration();
    }
    return box(iterator.next());
  };

  return (py::class_<ak::Iterator>(m, name.c_str())
      .def(py::init([](py::object content) -> ak::Iterator {
        return ak::Iterator(unbox_content(content));
      }))
      .def("__repr__", &ak::Iterator::tostring)
      .def("__next__", next)
      .def("next", next)
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
  return box(self.getitem(toslice(obj)));
}

void fillable_fill(ak::FillableArray& self, py::object obj) {
  if (py::isinstance<py::bool_>(obj)) {
    self.boolean(obj.cast<bool>());
  }
  else if (py::isinstance<py::int_>(obj)) {
    self.integer(obj.cast<int64_t>());
  }
  else if (py::isinstance<py::float_>(obj)) {
    self.real(obj.cast<double>());
  }
  // FIXME: strings, dicts...
  else if (py::isinstance<py::sequence>(obj)) {
    py::sequence seq = obj.cast<py::sequence>();
    self.beginlist();
    for (auto x : seq) {
      fillable_fill(self, x);
    }
    self.endlist();
    return;
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
      .def("boolean", &ak::FillableArray::boolean)
      .def("integer", &ak::FillableArray::integer)
      .def("real", &ak::FillableArray::real)
      .def("beginlist", &ak::FillableArray::beginlist)
      .def("endlist", &ak::FillableArray::endlist)
      .def("fill", &fillable_fill)
  );
}

/////////////////////////////////////////////////////////////// Type

py::class_<ak::Type, std::shared_ptr<ak::Type>> make_Type(py::handle m, std::string name) {
  return (py::class_<ak::Type, std::shared_ptr<ak::Type>>(m, name.c_str())
      .def("__ne__", [](std::shared_ptr<ak::Type> self, std::shared_ptr<ak::Type> other) -> bool {
        return !self.get()->equal(other);
      })
  );
}

py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type> make_ArrayType(py::handle m, std::string name) {
  return (py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type>(m, name.c_str())
      .def(py::init<std::shared_ptr<ak::Type>, int64_t>())
      .def("type", &ak::ArrayType::type)
      .def("length", &ak::ArrayType::length)
      .def("__repr__", &ak::ArrayType::tostring)
      .def("__eq__", &ak::ArrayType::equal)
      .def("compatible", &ak::ArrayType::compatible, py::arg("other"), py::arg("bool_is_int") = false, py::arg("int_is_float") = false, py::arg("ignore_null") = true, py::arg("unknown_is_anything") = true)
  );
}

py::class_<ak::UnknownType, std::shared_ptr<ak::UnknownType>, ak::Type> make_UnknownType(py::handle m, std::string name) {
  return (py::class_<ak::UnknownType, std::shared_ptr<ak::UnknownType>, ak::Type>(m, name.c_str())
      .def(py::init<>())
      .def("__repr__", &ak::UnknownType::tostring)
      .def("__eq__", &ak::UnknownType::equal)
      .def("compatible", &ak::UnknownType::compatible, py::arg("other"), py::arg("bool_is_int") = false, py::arg("int_is_float") = false, py::arg("ignore_null") = true, py::arg("unknown_is_anything") = true)
  );
}

py::class_<ak::PrimitiveType, std::shared_ptr<ak::PrimitiveType>, ak::Type> make_PrimitiveType(py::handle m, std::string name) {
  return (py::class_<ak::PrimitiveType, std::shared_ptr<ak::PrimitiveType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::string dtype) -> ak::PrimitiveType {
        if (dtype == std::string("bool")) {
          return ak::PrimitiveType(ak::PrimitiveType::boolean);
        }
        else if (dtype == std::string("int8")) {
          return ak::PrimitiveType(ak::PrimitiveType::int8);
        }
        else if (dtype == std::string("int16")) {
          return ak::PrimitiveType(ak::PrimitiveType::int16);
        }
        else if (dtype == std::string("int32")) {
          return ak::PrimitiveType(ak::PrimitiveType::int32);
        }
        else if (dtype == std::string("int64")) {
          return ak::PrimitiveType(ak::PrimitiveType::int64);
        }
        else if (dtype == std::string("uint8")) {
          return ak::PrimitiveType(ak::PrimitiveType::uint8);
        }
        else if (dtype == std::string("uint16")) {
          return ak::PrimitiveType(ak::PrimitiveType::uint16);
        }
        else if (dtype == std::string("uint32")) {
          return ak::PrimitiveType(ak::PrimitiveType::uint32);
        }
        else if (dtype == std::string("uint64")) {
          return ak::PrimitiveType(ak::PrimitiveType::uint64);
        }
        else if (dtype == std::string("float32")) {
          return ak::PrimitiveType(ak::PrimitiveType::float32);
        }
        else if (dtype == std::string("float64")) {
          return ak::PrimitiveType(ak::PrimitiveType::float64);
        }
        else {
          throw std::invalid_argument(std::string("unrecognized primitive type: ") + dtype);
        }
      }))
      .def("__repr__", &ak::PrimitiveType::tostring)
      .def("__eq__", &ak::PrimitiveType::equal)
      .def("compatible", &ak::PrimitiveType::compatible, py::arg("other"), py::arg("bool_is_int") = false, py::arg("int_is_float") = false, py::arg("ignore_null") = true, py::arg("unknown_is_anything") = true)
  );
}

py::class_<ak::RegularType, std::shared_ptr<ak::RegularType>, ak::Type> make_RegularType(py::handle m, std::string name) {
  return (py::class_<ak::RegularType, std::shared_ptr<ak::RegularType>, ak::Type>(m, name.c_str())
      .def(py::init<std::shared_ptr<ak::Type>, int64_t>())
      .def_property_readonly("type", &ak::RegularType::type)
      .def_property_readonly("size", &ak::RegularType::size)
      .def("__repr__", &ak::RegularType::tostring)
      .def("__eq__", &ak::RegularType::equal)
      .def("compatible", &ak::RegularType::compatible, py::arg("other"), py::arg("bool_is_int") = false, py::arg("int_is_float") = false, py::arg("ignore_null") = true, py::arg("unknown_is_anything") = true)
  );
}

py::class_<ak::ListType, std::shared_ptr<ak::ListType>, ak::Type> make_ListType(py::handle m, std::string name) {
  return (py::class_<ak::ListType, std::shared_ptr<ak::ListType>, ak::Type>(m, name.c_str())
      .def(py::init<std::shared_ptr<ak::Type>>())
      .def_property_readonly("type", &ak::ListType::type)
      .def("__repr__", &ak::ListType::tostring)
      .def("__eq__", &ak::ListType::equal)
      .def("compatible", &ak::ListType::compatible, py::arg("other"), py::arg("bool_is_int") = false, py::arg("int_is_float") = false, py::arg("ignore_null") = true, py::arg("unknown_is_anything") = true)
  );
}

py::class_<ak::OptionType, std::shared_ptr<ak::OptionType>, ak::Type> make_OptionType(py::handle m, std::string name) {
  return (py::class_<ak::OptionType, std::shared_ptr<ak::OptionType>, ak::Type>(m, name.c_str())
      .def(py::init<std::shared_ptr<ak::Type>>())
      .def_property_readonly("type", &ak::OptionType::type)
      .def("__repr__", &ak::OptionType::tostring)
      .def("__eq__", &ak::OptionType::equal)
      .def("compatible", &ak::OptionType::compatible, py::arg("other"), py::arg("bool_is_int") = false, py::arg("int_is_float") = false, py::arg("ignore_null") = true, py::arg("unknown_is_anything") = true)
  );
}

py::class_<ak::UnionType, std::shared_ptr<ak::UnionType>, ak::Type> make_UnionType(py::handle m, std::string name) {
  return (py::class_<ak::UnionType, std::shared_ptr<ak::UnionType>, ak::Type>(m, name.c_str())
      .def(py::init([](py::args args) -> ak::UnionType {
        std::vector<std::shared_ptr<ak::Type>> types;
        for (auto x : args) {
          types.push_back(unbox_type(x));
        }
        return ak::UnionType(types);
      }))
      .def_property_readonly("numtypes", &ak::UnionType::numtypes)
      .def_property_readonly("types", [](ak::UnionType& self) -> py::tuple {
        py::tuple types((size_t)self.numtypes());
        for (int64_t i = 0;  i < self.numtypes();  i++) {
          types[(size_t)i] = box(self.type(i));
        }
        return types;
      })
      .def("type", &ak::UnionType::type)
      .def("__repr__", &ak::UnionType::tostring)
      .def("__eq__", &ak::UnionType::equal)
      .def("compatible", &ak::UnionType::compatible, py::arg("other"), py::arg("bool_is_int") = false, py::arg("int_is_float") = false, py::arg("ignore_null") = true, py::arg("unknown_is_anything") = true)
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
py::class_<T, ak::Content> content(py::class_<T, ak::Content>& x) {
  return x.def("__repr__", &repr<T>)
          .def_property("id", [](T& self) -> py::object { return box(self.id()); }, [](T& self, py::object id) -> void { self.setid(unbox_id(id)); })
          .def("setid", [](T& self, py::object id) -> void {
           self.setid(unbox_id(id));
          })
         .def("setid", [](T& self) -> void {
           self.setid();
         })
         .def("__len__", &len<T>)
         .def("__getitem__", &getitem<T>)
         .def("__iter__", &iter<T>)
         .def("tojson", &tojson_string<T>, py::arg("pretty") = false, py::arg("maxdecimals") = py::none())
         .def("tojson", &tojson_file<T>, py::arg("destination"), py::arg("pretty") = false, py::arg("maxdecimals") = py::none(), py::arg("buffersize") = 65536)
         .def_property_readonly("type", &ak::Content::type);
}

py::class_<ak::Content> make_Content(py::handle m, std::string name) {
  return py::class_<ak::Content>(m, name.c_str());
}

/////////////////////////////////////////////////////////////// NumpyArray

py::class_<ak::NumpyArray, ak::Content> make_NumpyArray(py::handle m, std::string name) {
  return content(py::class_<ak::NumpyArray, ak::Content>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](ak::NumpyArray& self) -> py::buffer_info {
        return py::buffer_info(
          self.byteptr(),
          self.itemsize(),
          self.format(),
          self.ndim(),
          self.shape(),
          self.strides());
      })

      .def(py::init([](py::array array, py::object id) -> ak::NumpyArray {
        py::buffer_info info = array.request();
        if (info.ndim == 0) {
          throw std::invalid_argument("NumpyArray must not be scalar; try array.reshape(1)");
        }
        if (info.shape.size() != info.ndim  ||  info.strides.size() != info.ndim) {
          throw std::invalid_argument("NumpyArray len(shape) != ndim or len(strides) != ndim");
        }
        return ak::NumpyArray(unbox_id(id), std::shared_ptr<void>(
          reinterpret_cast<void*>(info.ptr), pyobject_deleter<void>(array.ptr())),
          info.shape,
          info.strides,
          0,
          info.itemsize,
          info.format);
      }), py::arg("array"), py::arg("id") = py::none())

      .def_property_readonly("shape", &ak::NumpyArray::shape)
      .def_property_readonly("strides", &ak::NumpyArray::strides)
      .def_property_readonly("itemsize", &ak::NumpyArray::itemsize)
      .def_property_readonly("format", &ak::NumpyArray::format)
      .def_property_readonly("ndim", &ak::NumpyArray::ndim)
      .def_property_readonly("isscalar", &ak::NumpyArray::isscalar)
      .def_property_readonly("isempty", &ak::NumpyArray::isempty)

      .def_property_readonly("iscontiguous", &ak::NumpyArray::iscontiguous)
      .def("contiguous", &ak::NumpyArray::contiguous)
      .def("become_contiguous", &ak::NumpyArray::become_contiguous)
  );
}

/////////////////////////////////////////////////////////////// ListArray

template <typename T>
py::class_<ak::ListArrayOf<T>, ak::Content> make_ListArrayOf(py::handle m, std::string name) {
  return content(py::class_<ak::ListArrayOf<T>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& starts, ak::IndexOf<T>& stops, py::object content, py::object id) -> ak::ListArrayOf<T> {
        return ak::ListArrayOf<T>(unbox_id(id), starts, stops, unbox_content(content));
      }), py::arg("starts"), py::arg("stops"), py::arg("content"), py::arg("id") = py::none())

      .def_property_readonly("starts", &ak::ListArrayOf<T>::starts)
      .def_property_readonly("stops", &ak::ListArrayOf<T>::stops)
      .def_property_readonly("content", [](ak::ListArrayOf<T>& self) -> py::object {
        return box(self.content());
      })
  );
}

/////////////////////////////////////////////////////////////// ListOffsetArray

template <typename T>
py::class_<ak::ListOffsetArrayOf<T>, ak::Content> make_ListOffsetArrayOf(py::handle m, std::string name) {
  return content(py::class_<ak::ListOffsetArrayOf<T>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& offsets, py::object content, py::object id) -> ak::ListOffsetArrayOf<T> {
        return ak::ListOffsetArrayOf<T>(unbox_id(id), offsets, std::shared_ptr<ak::Content>(unbox_content(content)));
      }), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none())

      .def_property_readonly("offsets", &ak::ListOffsetArrayOf<T>::offsets)
      .def_property_readonly("content", [](ak::ListOffsetArrayOf<T>& self) -> py::object {
        return box(self.content());
      })
  );
}

/////////////////////////////////////////////////////////////// EmptyArray

py::class_<ak::EmptyArray, ak::Content> make_EmptyArray(py::handle m, std::string name) {
  return content(py::class_<ak::EmptyArray, ak::Content>(m, name.c_str())
      .def(py::init([](py::object id) -> ak::EmptyArray {
        return ak::EmptyArray(unbox_id(id));
      }), py::arg("id") = py::none())
  );
}

/////////////////////////////////////////////////////////////// RegularArray

py::class_<ak::RegularArray, ak::Content> make_RegularArray(py::handle m, std::string name) {
  return content(py::class_<ak::RegularArray, ak::Content>(m, name.c_str())
      .def(py::init([](py::object content, int64_t size, py::object id) -> ak::RegularArray {
        return ak::RegularArray(unbox_id(id), std::shared_ptr<ak::Content>(unbox_content(content)), size);
      }), py::arg("content"), py::arg("size"), py::arg("id") = py::none())

      .def_property_readonly("size", &ak::RegularArray::size)
      .def_property_readonly("content", [](ak::RegularArray& self) -> py::object {
        return box(self.content());
      })
  );
}

/////////////////////////////////////////////////////////////// RecordArray

py::class_<ak::RecordArray, ak::Content> make_RecordArray(py::handle m, std::string name) {
  return content(py::class_<ak::RecordArray, ak::Content>(m, name.c_str())
      .def(py::init([](py::object id) -> ak::RecordArray {
        return ak::RecordArray(unbox_id(id));
      }), py::arg("id") = py::none())

      .def_property_readonly("numfields", &ak::RecordArray::numfields)
      .def("index", &ak::RecordArray::index)
      .def("key", &ak::RecordArray::key)
      .def("has", &ak::RecordArray::has)
      .def("aliases", [](ak::RecordArray& self, int64_t index) -> std::vector<std::string> {
        return self.aliases(index);
      })
      .def("aliases", [](ak::RecordArray& self, std::string key) -> std::vector<std::string> {
        return self.aliases(key);
      })
      .def("field", [](ak::RecordArray& self, int64_t index) -> py::object {
        return box(self.field(index));
      })
      .def("field", [](ak::RecordArray& self, std::string key) -> py::object {
        return box(self.field(key));
      })
      .def("append", [](ak::RecordArray& self, py::object content, py::object key) -> void {
        if (key.is(py::none())) {
          self.append(unbox_content(content));
        }
        else {
          self.append(unbox_content(content), key.cast<std::string>());
        }
      }, py::arg("content"), py::arg("key") = py::none())
      .def("setkey", &ak::RecordArray::setkey)

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

  make_IdentityOf<int32_t>(m, "Identity32");
  make_IdentityOf<int64_t>(m, "Identity64");

  make_Slice(m, "Slice");

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
