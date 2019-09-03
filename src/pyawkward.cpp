// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "awkward/Index.h"
#include "awkward/Identity.h"
#include "awkward/Content.h"
#include "awkward/Iterator.h"
#include "awkward/NumpyArray.h"
#include "awkward/ListOffsetArray.h"

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

py::object unwrap(std::shared_ptr<ak::Content> content) {
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
  else if (ak::ListOffsetArray32* raw = dynamic_cast<ak::ListOffsetArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArray64* raw = dynamic_cast<ak::ListOffsetArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error("missing unwrapper for Content subtype");
  }
}

py::object unwrap(std::shared_ptr<ak::Identity> id) {
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
    throw std::runtime_error("missing unwrapper for Identity subtype");
  }
}

template <typename CONTENT>
void setid(CONTENT& self, py::object obj) {
  ak::Identity32* id32;
  ak::Identity64* id64;
  try {
    id32 = obj.cast<ak::Identity32*>();
  }
  catch (py::cast_error err) {
    try {
      id64 = obj.cast<ak::Identity64*>();
    }
    catch (py::cast_error err) {
      throw std::invalid_argument("'id' member must be Identity32 or Identity64");
    }
    if (id64->length() != self.length()) {
      throw std::invalid_argument("Identity must have the same length as the Content to which it is assigned");
    }
    if (id64 != nullptr) {
      self.setid(std::shared_ptr<ak::Identity>(new ak::Identity64(id64->ref(), id64->fieldloc(), id64->offset(), id64->width(), id64->length(), id64->ptr())));
      return;
    }
  }
  if (id32 != nullptr) {
    self.setid(std::shared_ptr<ak::Identity>(new ak::Identity32(id32->ref(), id32->fieldloc(), id32->offset(), id32->width(), id32->length(), id32->ptr())));
  }
  else {
    self.setid(std::shared_ptr<ak::Identity>(nullptr));
  }
}

/////////////////////////////////////////////////////////////// Index

template <typename T>
py::class_<ak::IndexOf<T>> make_IndexOf(py::handle m, std::string name) {
  return py::class_<ak::IndexOf<T>>(m, name.c_str(), py::buffer_protocol())
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
          throw std::invalid_argument(name + std::string(" must be built from a compact array (array.strides == (array.itemsize,)); try array.copy()"));
        }
        return ak::IndexOf<T>(
          std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr), pyobject_deleter<T>(array.ptr())),
          0,
          (T)info.shape[0]);
      }))

      .def("__repr__", [](ak::IndexOf<T>& self) -> const std::string {
        return self.tostring();
      })

      .def("__len__", &ak::IndexOf<T>::length)
      .def("__getitem__", &ak::IndexOf<T>::get)
      .def("__getitem__", [](ak::IndexOf<T>& self, py::slice slice) -> ak::IndexOf<T> {
        size_t start, stop, step, length;
        if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
          throw py::error_already_set();
        }
        return self.slice((int64_t)start, (int64_t)stop);
      })

  ;
}

/////////////////////////////////////////////////////////////// Identity

template <typename T>
py::class_<ak::IdentityOf<T>> make_IdentityOf(py::handle m, std::string name) {
  return py::class_<ak::IdentityOf<T>>(m, name.c_str(), py::buffer_protocol())
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
          throw std::invalid_argument(name + std::string(" must be built from a compact array (array.stries == (array.shape[1]*array.itemsize, array.itemsize)); try array.copy()"));
        }
        return ak::IdentityOf<T>(ref, fieldloc, 0, info.shape[1], info.shape[0],
            std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr), pyobject_deleter<T>(array.ptr())));
      }))

      .def("__repr__", [](ak::IdentityOf<T>& self) -> const std::string {
        return self.tostring();
      })

      .def("__len__", &ak::IdentityOf<T>::length)
      .def("__getitem__", &ak::IdentityOf<T>::get)
      .def("__getitem__", [](ak::IdentityOf<T>& self, py::slice slice) -> ak::IdentityOf<T> {
        size_t start, stop, step, length;
        if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
          throw py::error_already_set();
        }
        std::shared_ptr<ak::Identity> out = self.slice((int64_t)start, (int64_t)stop);
        ak::IdentityOf<T>* raw = dynamic_cast<ak::IdentityOf<T>*>(out.get());
        return ak::IdentityOf<T>(raw->ref(), raw->fieldloc(), raw->offset(), raw->width(), raw->length(), raw->ptr());
      })

      .def_property_readonly("ref", &ak::IdentityOf<T>::ref)
      .def_property_readonly("fieldloc", &ak::IdentityOf<T>::fieldloc)
      .def_property_readonly("width", &ak::IdentityOf<T>::width)
      .def_property_readonly("length", &ak::IdentityOf<T>::length)
      .def_property_readonly("array", [](py::buffer& self) -> py::array {
        return py::array(self);
      })

  ;
}

/////////////////////////////////////////////////////////////// Iterator

py::class_<ak::Iterator> make_Iterator(py::handle m, std::string name) {
  auto next = [](ak::Iterator& iterator) -> py::object {
    if (iterator.isdone()) {
      throw py::stop_iteration();
    }
    return unwrap(iterator.next());
  };

  return py::class_<ak::Iterator>(m, name.c_str())
      .def(py::init([](ak::NumpyArray& content) -> ak::Iterator {
        return ak::Iterator(std::shared_ptr<ak::Content>(new ak::NumpyArray(content)));
      }))
      .def(py::init([](ak::ListOffsetArrayOf<int32_t>& content) -> ak::Iterator {
        return ak::Iterator(std::shared_ptr<ak::Content>(new ak::ListOffsetArrayOf<int32_t>(content)));
      }))
      .def(py::init([](ak::ListOffsetArrayOf<int64_t>& content) -> ak::Iterator {
        return ak::Iterator(std::shared_ptr<ak::Content>(new ak::ListOffsetArrayOf<int64_t>(content)));
      }))

      .def("__next__", next)
      .def("next", next)

      .def("__repr__", [](ak::Iterator& self) -> const std::string {
        return self.tostring();
      })

  ;
}

/////////////////////////////////////////////////////////////// NumpyArray

template <typename IDENTITY>
ak::NumpyArray init_NumpyArray(py::array array, py::object id) {
  py::buffer_info info = array.request();
  if (info.ndim == 0) {
    throw std::invalid_argument("NumpyArray must not be scalar; try array.reshape(1)");
  }
  if (info.shape.size() != info.ndim  ||  info.strides.size() != info.ndim) {
    throw std::invalid_argument("NumpyArray len(shape) != ndim or len(strides) != ndim");
  }
  ak::NumpyArray out = ak::NumpyArray(std::shared_ptr<ak::Identity>(nullptr), std::shared_ptr<void>(
        reinterpret_cast<void*>(info.ptr), pyobject_deleter<void>(array.ptr())),
      info.shape,
      info.strides,
      0,
      info.itemsize,
      info.format);
  setid<ak::NumpyArray>(out, id);
  return out;
}

py::class_<ak::NumpyArray> make_NumpyArray(py::handle m, std::string name) {
  return py::class_<ak::NumpyArray>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](ak::NumpyArray& self) -> py::buffer_info {
        return py::buffer_info(
          self.byteptr(),
          self.itemsize(),
          self.format(),
          self.ndim(),
          self.shape(),
          self.strides());
      })

      .def(py::init(&init_NumpyArray<ak::Identity32>), py::arg("array"), py::arg("id") = py::none())
      .def(py::init(&init_NumpyArray<ak::Identity64>), py::arg("array"), py::arg("id") = py::none())

      .def_property("id", [](ak::NumpyArray& self) -> py::object { return unwrap(self.id()); }, &setid<ak::NumpyArray>)
      .def("setid", &setid<ak::NumpyArray>)
      .def("setid", [](ak::NumpyArray& self) -> void { self.setid(); })
      .def("__repr__", [](ak::NumpyArray& self) -> const std::string {
        return self.tostring();
      })

      .def_property_readonly("shape", &ak::NumpyArray::shape)
      .def_property_readonly("strides", &ak::NumpyArray::strides)
      .def_property_readonly("itemsize", &ak::NumpyArray::itemsize)
      .def_property_readonly("format", &ak::NumpyArray::format)
      .def_property_readonly("ndim", &ak::NumpyArray::ndim)
      .def_property_readonly("isscalar", &ak::NumpyArray::isscalar)
      .def_property_readonly("isempty", &ak::NumpyArray::isempty)
      .def_property_readonly("iscompact", &ak::NumpyArray::iscompact)

      .def("__len__", &ak::NumpyArray::length)
      .def("__getitem__", [](ak::NumpyArray& self, int64_t at) -> py::object {
        return unwrap(self.get(at));
      })
      .def("__getitem__", [](ak::NumpyArray& self, py::slice slice) -> py::object {
        size_t start, stop, step, length;
        if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
          throw py::error_already_set();
        }
        return unwrap(self.slice((int64_t)start, (int64_t)stop));
      })

      .def("__iter__", [](ak::NumpyArray& self) -> ak::Iterator {
        return ak::Iterator(std::shared_ptr<ak::Content>(new ak::NumpyArray(self)));
      })

  ;
}

/////////////////////////////////////////////////////////////// ListOffsetArray

template <typename T, typename CONTENT, typename IDENTITY>
ak::ListOffsetArrayOf<T> init_ListOffsetArrayOf(ak::IndexOf<T>& offsets, CONTENT& content, py::object id) {
  ak::ListOffsetArrayOf<T> out = ak::ListOffsetArrayOf<T>(std::shared_ptr<ak::Identity>(nullptr), offsets, std::shared_ptr<ak::Content>(content.shallow_copy()));
  setid(out, id);
  return out;
}

template <typename T>
py::class_<ak::ListOffsetArrayOf<T>> make_ListOffsetArrayOf(py::handle m, std::string name) {
  return py::class_<ak::ListOffsetArrayOf<T>>(m, name.c_str())
      .def(py::init(&init_ListOffsetArrayOf<T, ak::NumpyArray, ak::Identity32>), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none())
      .def(py::init(&init_ListOffsetArrayOf<T, ak::NumpyArray, ak::Identity64>), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none())
      .def(py::init(&init_ListOffsetArrayOf<T, ak::ListOffsetArray32, ak::Identity32>), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none())
      .def(py::init(&init_ListOffsetArrayOf<T, ak::ListOffsetArray32, ak::Identity64>), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none())
      .def(py::init(&init_ListOffsetArrayOf<T, ak::ListOffsetArray64, ak::Identity32>), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none())
      .def(py::init(&init_ListOffsetArrayOf<T, ak::ListOffsetArray64, ak::Identity64>), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none())

      .def_property_readonly("offsets", &ak::ListOffsetArrayOf<T>::offsets)
      .def_property_readonly("content", [](ak::ListOffsetArrayOf<T>& self) -> py::object {
        return unwrap(self.content());
      })

      .def_property("id", [](ak::ListOffsetArrayOf<T>& self) -> py::object { return unwrap(self.id()); }, &setid<ak::ListOffsetArrayOf<T>>)
      .def("setid", &setid<ak::ListOffsetArrayOf<T>>)
      .def("setid", [](ak::ListOffsetArrayOf<T>& self) -> void { self.setid(); })
      .def("__repr__", [](ak::ListOffsetArrayOf<T>& self) -> const std::string {
        return self.tostring();
      })

      .def("__len__", &ak::ListOffsetArrayOf<T>::length)
      .def("__getitem__", [](ak::ListOffsetArrayOf<T>& self, int64_t at) -> py::object {
        return unwrap(self.get(at));
      })
      .def("__getitem__", [](ak::ListOffsetArrayOf<T>& self, py::slice slice) -> py::object {
        size_t start, stop, step, length;
        if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
          throw py::error_already_set();
        }
        return unwrap(self.slice((int64_t)start, (int64_t)stop));
      })

      .def("__iter__", [](ak::ListOffsetArrayOf<T>& self) -> ak::Iterator {
        return ak::Iterator(std::shared_ptr<ak::Content>(new ak::ListOffsetArrayOf<T>(self)));
      })

  ;
}

PYBIND11_MODULE(layout, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  make_IndexOf<int32_t>(m, "Index32");
  make_IndexOf<int64_t>(m, "Index64");

  make_IdentityOf<int32_t>(m, "Identity32");
  make_IdentityOf<int64_t>(m, "Identity64");

  make_Iterator(m, "Iterator");

  make_NumpyArray(m, "NumpyArray");

  make_ListOffsetArrayOf<int32_t>(m, "ListOffsetArray32");
  make_ListOffsetArrayOf<int64_t>(m, "ListOffsetArray64");
}
