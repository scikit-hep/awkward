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
#include "awkward/array/Record.h"
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
  throw std::invalid_argument("content argument must be a Content subtype");
}

std::shared_ptr<ak::Identity> unbox_id_none(py::handle obj) {
  if (obj.is(py::none())) {
    return ak::Identity::none();
  }
  try {
    return obj.cast<ak::Identity32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::Identity64*>()->shallow_copy();
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
py::tuple location(const T& self) {
  if (self.id().get() == nullptr) {
    throw std::invalid_argument(self.classname() + std::string(" instance has no associated id (use 'setid' to assign one to the array it is in)"));
  }
  ak::Identity::FieldLoc fieldloc = self.id().get()->fieldloc();
  if (self.isscalar()) {
    py::tuple out((size_t)(self.id().get()->width()) + fieldloc.size());
    size_t j = 0;
    for (int64_t i = 0;  i < self.id().get()->width();  i++) {
      out[j] = py::cast(self.id().get()->value(0, i));
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
    py::tuple out((size_t)(self.id().get()->width() - 1) + fieldloc.size());
    size_t j = 0;
    for (int64_t i = 0;  i < self.id().get()->width();  i++) {
      if (i < self.id().get()->width() - 1) {
        out[j] = py::cast(self.id().get()->value(0, i));
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
py::object getid(T& self) {
  return box(self.id());
}

template <typename T>
void setid(T& self, py::object id) {
  self.setid(unbox_id_none(id));
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
      .def("location_at_str", &ak::IdentityOf<T>::location_at)
      .def("location_at", [](const ak::Identity& self, int64_t at) -> py::tuple {
        ak::Identity::FieldLoc fieldloc = self.fieldloc();
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

  else if (py::isinstance<py::str>(obj)) {
    slice.append(std::shared_ptr<ak::SliceItem>(new ak::SliceField(obj.cast<std::string>())));
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

    if (all_strings  &&  strings.size() != 0) {
      slice.append(std::shared_ptr<ak::SliceItem>(new ak::SliceFields(strings)));
    }
    else {
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

py::class_<ak::Slice, std::shared_ptr<ak::Slice>> make_Slice(py::handle m, std::string name) {
  return (py::class_<ak::Slice, std::shared_ptr<ak::Slice>>(m, name.c_str())
      .def(py::init([](py::object obj) {
        return toslice(obj);
      }))

      .def("__repr__", [](ak::Slice& self) -> const std::string {
        return self.tostring();
      })
  );
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
    if (all_strings  &&  strings.size() != 0) {
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

template <typename T>
std::shared_ptr<ak::Type> inner(T& self) {
  return self.inner();
}

template <typename T>
std::shared_ptr<ak::Type> inner_key(T& self, std::string key) {
  return self.inner(key);
}

template <typename T>
py::dict emptydict(T& self) {
  return py::dict();
}

ak::Type::Parameters dict2parameters(py::object in) {
  ak::Type::Parameters out;
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

py::dict parameters2dict(const ak::Type::Parameters& in) {
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
py::class_<T, ak::Type> type_methods(py::class_<T, std::shared_ptr<T>, ak::Type>& x) {
  return x.def("__repr__", &T::tostring)
          .def("nolength", &T::nolength)
          .def("level", &T::level)
          .def("inner", &inner<T>)
          .def("inner", &inner_key<T>)
          .def_property_readonly("numfields", &T::numfields)
          .def("fieldindex", &T::fieldindex)
          .def("key", &T::key)
          .def("haskey", &T::haskey)
          .def("keyaliases", [](T& self, int64_t fieldindex) -> std::vector<std::string> {
            return self.keyaliases(fieldindex);
          })
          .def("keyaliases", [](T& self, std::string key) -> std::vector<std::string> {
            return self.keyaliases(key);
          })
          .def("keys", &T::keys)
  ;
}

py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type> make_ArrayType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::Type> type, int64_t length, py::object parameters) -> ak::ArrayType {
        return ak::ArrayType(dict2parameters(parameters), type, length);
      }), py::arg("type"), py::arg("length"), py::arg("parameters") = py::none())
      .def_property_readonly("type", &ak::ArrayType::type)
      .def_property_readonly("length", &ak::ArrayType::length)
      .def_property("parameters", &getparameters<ak::ArrayType>, &setparameters<ak::ArrayType>)
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
      .def_property("parameters", &getparameters<ak::UnknownType>, &setparameters<ak::UnknownType>)
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
      .def_property("parameters", &getparameters<ak::PrimitiveType>, &setparameters<ak::PrimitiveType>)
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
      .def_property("parameters", &getparameters<ak::RegularType>, &setparameters<ak::RegularType>)
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
      .def_property("parameters", &getparameters<ak::ListType>, &setparameters<ak::ListType>)
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
      .def_property("parameters", &getparameters<ak::OptionType>, &setparameters<ak::OptionType>)
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
      .def_property("parameters", &getparameters<ak::UnionType>, &setparameters<ak::UnionType>)
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

template <typename T>
py::object lookup(const T& self) {
  std::shared_ptr<ak::RecordType::Lookup> lookup = self.lookup();
  if (lookup.get() == nullptr) {
    return py::none();
  }
  else {
    py::dict out;
    for (auto pair : *lookup.get()) {
      std::string cppkey = pair.first;
      py::str pykey(PyUnicode_DecodeUTF8(cppkey.data(), cppkey.length(), "surrogateescape"));
      out[pykey] = py::cast(pair.second);
    }
    return out;
  }
}

template <typename T>
py::object reverselookup(const T& self) {
  std::shared_ptr<ak::RecordType::ReverseLookup> reverselookup = self.reverselookup();
  if (reverselookup.get() == nullptr) {
    return py::none();
  }
  else {
    py::list out;
    for (auto item : *reverselookup.get()) {
      std::string cppkey = item;
      py::str pykey(PyUnicode_DecodeUTF8(cppkey.data(), cppkey.length(), "surrogateescape"));
      out.append(pykey);
    }
    return out;
  }
}

template <typename L, typename R>
void from_lookup(std::shared_ptr<L> lookup, std::shared_ptr<R> reverselookup, py::dict pylookup, py::object pyreverselookup, int64_t numfields) {
  for (auto x : pylookup) {
    std::string key = x.first.cast<std::string>();
    (*lookup)[key] = (size_t)x.second.cast<int64_t>();
  }
  if (pyreverselookup.is(py::none())) {
    for (int64_t i = 0;  i < numfields;  i++) {
      reverselookup.get()->push_back(std::to_string(i));
    }
    for (auto x : *lookup.get()) {
      if ((int64_t)x.second > numfields) {
        throw std::invalid_argument(std::string("lookup[") + ak::util::quote(x.first, true) + std::string("] is ") + std::to_string(x.second) + std::string(" but there are only ") + std::to_string(numfields) + std::string(" fields"));
      }
      (*reverselookup)[x.second] = x.first;
    }
  }
  else {
    if (!py::isinstance<py::iterable>(pyreverselookup)) {
      throw std::invalid_argument("reverselookup must be iterable");
    }
    for (auto x : pyreverselookup.cast<py::iterable>()) {
      if (!py::isinstance<py::str>(x)) {
        throw std::invalid_argument("elements of reverselookup must all be strings");
      }
      reverselookup.get()->push_back(x.cast<std::string>());
    }
  }
}

py::class_<ak::RecordType, std::shared_ptr<ak::RecordType>, ak::Type> make_RecordType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::RecordType, std::shared_ptr<ak::RecordType>, ak::Type>(m, name.c_str())
      .def(py::init([](py::tuple types, py::object parameters) -> ak::RecordType {
        std::vector<std::shared_ptr<ak::Type>> out;
        for (auto x : types) {
          out.push_back(unbox_type(x));
        }
        return ak::RecordType(dict2parameters(parameters), out, std::shared_ptr<ak::RecordType::Lookup>(nullptr), std::shared_ptr<ak::RecordType::ReverseLookup>(nullptr));
      }), py::arg("types"), py::arg("parameters") = py::none())
      .def(py::init([](py::dict types, py::object parameters) -> ak::RecordType {
        std::shared_ptr<ak::RecordType::Lookup> lookup(new ak::RecordType::Lookup);
        std::shared_ptr<ak::RecordType::ReverseLookup> reverselookup(new ak::RecordType::ReverseLookup);
        std::vector<std::shared_ptr<ak::Type>> out;
        for (auto x : types) {
          std::string key = x.first.cast<std::string>();
          (*lookup.get())[key] = out.size();
          reverselookup.get()->push_back(key);
          out.push_back(unbox_type(x.second));
        }
        return ak::RecordType(dict2parameters(parameters), out, lookup, reverselookup);
      }), py::arg("types"), py::arg("parameters") = py::none())
      .def_static("from_lookup", [](py::iterable types, py::dict pylookup, py::object pyreverselookup, py::object parameters) -> ak::RecordType {
        std::vector<std::shared_ptr<ak::Type>> out;
        for (auto x : types) {
          out.push_back(unbox_type(x));
        }
        std::shared_ptr<ak::RecordType::Lookup> lookup(new ak::RecordType::Lookup);
        std::shared_ptr<ak::RecordType::ReverseLookup> reverselookup(new ak::RecordType::ReverseLookup);
        from_lookup<ak::RecordType::Lookup, ak::RecordType::ReverseLookup>(lookup, reverselookup, pylookup, pyreverselookup, (int64_t)out.size());
        return ak::RecordType(dict2parameters(parameters), out, lookup, reverselookup);
      }, py::arg("types"), py::arg("lookup"), py::arg("reverselookup") = py::none(), py::arg("parameters") = py::none())
      .def("__getitem__", [](ak::RecordType& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("__getitem__", [](ak::RecordType& self, std::string key) -> py::object {
        return box(self.field(key));
      })
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
      .def_property_readonly("lookup", &lookup<ak::RecordType>)
      .def_property_readonly("reverselookup", &reverselookup<ak::RecordType>)
      .def_property("parameters", &getparameters<ak::RecordType>, &setparameters<ak::RecordType>)
      .def(py::pickle([](const ak::RecordType& self) {
        py::tuple fields((size_t)self.numfields());
        for (int64_t i = 0;  i < self.numfields();  i++) {
          fields[(size_t)i] = box(self.field(i));
        }
        return py::make_tuple(parameters2dict(self.parameters()), fields, lookup<ak::RecordType>(self), reverselookup<ak::RecordType>(self));
      }, [](py::tuple state) {
        std::vector<std::shared_ptr<ak::Type>> fields;
        for (auto x : state[1]) {
          fields.push_back(unbox_type(x));
        }
        std::shared_ptr<ak::RecordType::Lookup> lookup(new ak::RecordType::Lookup);
        std::shared_ptr<ak::RecordType::ReverseLookup> reverselookup(new ak::RecordType::ReverseLookup);
        from_lookup<ak::RecordType::Lookup, ak::RecordType::ReverseLookup>(lookup, reverselookup, state[2].cast<py::dict>(), state[3], (int64_t)fields.size());
        return ak::RecordType(dict2parameters(state[0]), fields, lookup, reverselookup);
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
          .def_property("id", [](T& self) -> py::object { return box(self.id()); }, [](T& self, py::object id) -> void { self.setid(unbox_id_none(id)); })
          .def("setid", [](T& self, py::object id) -> void {
           self.setid(unbox_id_none(id));
          })
          .def("setid", [](T& self) -> void {
            self.setid();
          })
          .def_property_readonly("baretype", &ak::Content::baretype)
          .def_property_readonly("isbare", &ak::Content::isbare)
          .def_property("type", [](T& self) -> py::object {
            return box(self.type());
          }, [](T& self, py::object type) -> void {
            self.settype(unbox_type(type));
          })
          .def("accepts", [](T& self, py::object type) -> bool {
            return self.accepts(unbox_type(type));
          })
          .def("__len__", &len<T>)
          .def("__getitem__", &getitem<T>)
          .def("__iter__", &iter<T>)
          .def("tojson", &tojson_string<T>, py::arg("pretty") = false, py::arg("maxdecimals") = py::none())
          .def("tojson", &tojson_file<T>, py::arg("destination"), py::arg("pretty") = false, py::arg("maxdecimals") = py::none(), py::arg("buffersize") = 65536)
          .def_property_readonly("location", &location<T>)
          .def_property_readonly("numfields", &T::numfields)
          .def("fieldindex", &T::fieldindex)
          .def("key", &T::key)
          .def("haskey", &T::haskey)
          .def("keyaliases", [](T& self, int64_t fieldindex) -> std::vector<std::string> {
            return self.keyaliases(fieldindex);
          })
          .def("keyaliases", [](T& self, std::string key) -> std::vector<std::string> {
            return self.keyaliases(key);
          })
          .def("keys", &T::keys)
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

      .def(py::init([](py::array array, py::object id, py::object type) -> ak::NumpyArray {
        py::buffer_info info = array.request();
        if (info.ndim == 0) {
          throw std::invalid_argument("NumpyArray must not be scalar; try array.reshape(1)");
        }
        if (info.shape.size() != info.ndim  ||  info.strides.size() != info.ndim) {
          throw std::invalid_argument("NumpyArray len(shape) != ndim or len(strides) != ndim");
        }
        return ak::NumpyArray(unbox_id_none(id), unbox_type_none(type), std::shared_ptr<void>(
          reinterpret_cast<void*>(info.ptr), pyobject_deleter<void>(array.ptr())),
          info.shape,
          info.strides,
          0,
          info.itemsize,
          info.format);
      }), py::arg("array"), py::arg("id") = py::none(), py::arg("type") = py::none())

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
py::class_<ak::ListArrayOf<T>, std::shared_ptr<ak::ListArrayOf<T>>, ak::Content> make_ListArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::ListArrayOf<T>, std::shared_ptr<ak::ListArrayOf<T>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& starts, ak::IndexOf<T>& stops, py::object content, py::object id, py::object type) -> ak::ListArrayOf<T> {
        return ak::ListArrayOf<T>(unbox_id_none(id), unbox_type_none(type), starts, stops, unbox_content(content));
      }), py::arg("starts"), py::arg("stops"), py::arg("content"), py::arg("id") = py::none(), py::arg("type") = py::none())

      .def_property_readonly("starts", &ak::ListArrayOf<T>::starts)
      .def_property_readonly("stops", &ak::ListArrayOf<T>::stops)
      .def_property_readonly("content", &ak::ListArrayOf<T>::content)
  );
}

/////////////////////////////////////////////////////////////// ListOffsetArray

template <typename T>
py::class_<ak::ListOffsetArrayOf<T>, std::shared_ptr<ak::ListOffsetArrayOf<T>>, ak::Content> make_ListOffsetArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::ListOffsetArrayOf<T>, std::shared_ptr<ak::ListOffsetArrayOf<T>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& offsets, py::object content, py::object id, py::object type) -> ak::ListOffsetArrayOf<T> {
        return ak::ListOffsetArrayOf<T>(unbox_id_none(id), unbox_type_none(type), offsets, std::shared_ptr<ak::Content>(unbox_content(content)));
      }), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none(), py::arg("type") = py::none())

      .def_property_readonly("offsets", &ak::ListOffsetArrayOf<T>::offsets)
      .def_property_readonly("content", &ak::ListOffsetArrayOf<T>::content)
  );
}

/////////////////////////////////////////////////////////////// EmptyArray

py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content> make_EmptyArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content>(m, name.c_str())
      .def(py::init([](py::object id, py::object type) -> ak::EmptyArray {
        return ak::EmptyArray(unbox_id_none(id), unbox_type_none(type));
      }), py::arg("id") = py::none(), py::arg("type") = py::none())
  );
}

/////////////////////////////////////////////////////////////// RegularArray

py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content> make_RegularArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content>(m, name.c_str())
      .def(py::init([](py::object content, int64_t size, py::object id, py::object type) -> ak::RegularArray {
        return ak::RegularArray(unbox_id_none(id), unbox_type_none(type), std::shared_ptr<ak::Content>(unbox_content(content)), size);
      }), py::arg("content"), py::arg("size"), py::arg("id") = py::none(), py::arg("type") = py::none())

      .def_property_readonly("size", &ak::RegularArray::size)
      .def_property_readonly("content", &ak::RegularArray::content)
  );
}

/////////////////////////////////////////////////////////////// RecordArray

py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content> make_RecordArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content>(m, name.c_str())
      .def(py::init([](py::dict contents, py::object id, py::object type) -> ak::RecordArray {
        std::shared_ptr<ak::RecordArray::Lookup> lookup(new ak::RecordArray::Lookup);
        std::shared_ptr<ak::RecordArray::ReverseLookup> reverselookup(new ak::RecordArray::ReverseLookup);
        std::vector<std::shared_ptr<ak::Content>> out;
        for (auto x : contents) {
          std::string key = x.first.cast<std::string>();
          (*lookup.get())[key] = out.size();
          reverselookup.get()->push_back(key);
          out.push_back(unbox_content(x.second));
        }
        if (out.size() == 0) {
          throw std::invalid_argument("construct RecordArrays without fields using RecordArray(length) where length is an integer");
        }
        return ak::RecordArray(unbox_id_none(id), unbox_type_none(type), out, lookup, reverselookup);
      }), py::arg("contents"), py::arg("id") = py::none(), py::arg("type") = py::none())
      .def(py::init([](py::iterable contents, py::object id, py::object type) -> ak::RecordArray {
        std::vector<std::shared_ptr<ak::Content>> out;
        for (auto x : contents) {
          out.push_back(unbox_content(x));
        }
        if (out.size() == 0) {
          throw std::invalid_argument("construct RecordArrays without fields using RecordArray(length) where length is an integer");
        }
        return ak::RecordArray(unbox_id_none(id), unbox_type_none(type), out, std::shared_ptr<ak::RecordArray::Lookup>(nullptr), std::shared_ptr<ak::RecordArray::ReverseLookup>(nullptr));
      }), py::arg("contents"), py::arg("id") = py::none(), py::arg("type") = py::none())
      .def(py::init([](int64_t length, bool istuple, py::object id, py::object type) -> ak::RecordArray {
        return ak::RecordArray(unbox_id_none(id), unbox_type_none(type), length, istuple);
      }), py::arg("length"), py::arg("istuple") = false, py::arg("id") = py::none(), py::arg("type") = py::none())
      .def_static("from_lookup", [](py::iterable contents, py::dict pylookup, py::object pyreverselookup, py::object id, py::object type) -> ak::RecordArray {
        std::vector<std::shared_ptr<ak::Content>> out;
        for (auto x : contents) {
          out.push_back(unbox_content(x));
        }
        if (out.size() == 0) {
          throw std::invalid_argument("construct RecordArrays without fields using RecordArray(length) where length is an integer");
        }
        std::shared_ptr<ak::RecordArray::Lookup> lookup(new ak::RecordArray::Lookup);
        std::shared_ptr<ak::RecordArray::ReverseLookup> reverselookup(new ak::RecordArray::ReverseLookup);
        from_lookup<ak::RecordArray::Lookup, ak::RecordArray::ReverseLookup>(lookup, reverselookup, pylookup, pyreverselookup, (int64_t)out.size());
        return ak::RecordArray(unbox_id_none(id), unbox_type_none(type), out, lookup, reverselookup);
      }, py::arg("contents"), py::arg("lookup"), py::arg("reverselookup") = py::none(), py::arg("id") = py::none(), py::arg("type") = py::none())

      .def_property_readonly("istuple", &ak::RecordArray::istuple)
      .def_property_readonly("contents", &ak::RecordArray::contents)
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
      .def_property_readonly("lookup", &lookup<ak::RecordArray>)
      .def_property_readonly("reverselookup", &reverselookup<ak::RecordArray>)
      .def_property_readonly("astuple", [](ak::RecordArray& self) -> py::object {
        return box(self.astuple().shallow_copy());
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

py::class_<ak::Record, std::shared_ptr<ak::Record>> make_Record(py::handle m, std::string name) {
  return py::class_<ak::Record, std::shared_ptr<ak::Record>>(m, name.c_str())
      .def(py::init<ak::RecordArray, int64_t>())
      .def("__repr__", &repr<ak::Record>)
      .def_property_readonly("id", [](ak::Record& self) -> py::object { return box(self.id()); })
      .def("__getitem__", &getitem<ak::Record>)
      .def_property_readonly("baretype", &ak::Record::baretype)
      .def_property_readonly("isbare", &ak::Record::isbare)
      .def_property_readonly("type", [](ak::Record& self) -> py::object {
        return box(self.type());
      })
      .def("tojson", &tojson_string<ak::Record>, py::arg("pretty") = false, py::arg("maxdecimals") = py::none())
      .def("tojson", &tojson_file<ak::Record>, py::arg("destination"), py::arg("pretty") = false, py::arg("maxdecimals") = py::none(), py::arg("buffersize") = 65536)
      .def_property_readonly("type", &ak::Content::type)

      .def_property_readonly("array", [](ak::Record& self) -> py::object { return box(self.array()); })
      .def_property_readonly("at", &ak::Record::at)
      .def_property_readonly("istuple", &ak::Record::istuple)
      .def_property_readonly("numfields", &ak::Record::numfields)
      .def("fieldindex", &ak::Record::fieldindex)
      .def("key", &ak::Record::key)
      .def("haskey", &ak::Record::haskey)
      .def("keyaliases", [](ak::Record& self, int64_t fieldindex) -> std::vector<std::string> {
        return self.keyaliases(fieldindex);
      })
      .def("keyaliases", [](ak::Record& self, std::string key) -> std::vector<std::string> {
        return self.keyaliases(key);
      })
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
      .def_property_readonly("lookup", &lookup<ak::Record>)
      .def_property_readonly("reverselookup", &reverselookup<ak::Record>)
      .def_property_readonly("astuple", [](ak::Record& self) -> py::object {
        return box(self.astuple().shallow_copy());
      })
     .def_property_readonly("location", &location<ak::Record>)

  ;
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
