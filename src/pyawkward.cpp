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
  if (ak::NumpyArray* x = dynamic_cast<ak::NumpyArray*>(content.get())) {
    if (x->isscalar()) {
      return py::array(py::buffer_info(
        x->byteptr(),
        x->itemsize(),
        x->format(),
        x->ndim(),
        x->shape(),
        x->strides()
      )).attr("item")();
    }
    else {
      return py::cast(*x);
    }
  }
  else if (ak::ListOffsetArray32* x = dynamic_cast<ak::ListOffsetArray32*>(content.get())) {
    return py::cast(*x);
  }
  else if (ak::ListOffsetArray64* x = dynamic_cast<ak::ListOffsetArray64*>(content.get())) {
    return py::cast(*x);
  }
  else {
    throw std::runtime_error("missing unwrapper for Content subtype");
  }
}

void setid(ak::Content* obj, ak::Identity* id) {
  if (id == nullptr) {
    obj->setid(std::shared_ptr<ak::Identity>(nullptr));
  }
  else {
    if (id->length() != obj->length()) {
      throw std::invalid_argument("Identity must have the same length as array");
    }
    if (ak::Identity32* id32 = dynamic_cast<ak::Identity32*>(id)) {
      obj->setid(std::shared_ptr<ak::Identity>(new ak::Identity32(id->ref(), id->fieldloc(), id->offset(), id->width(), id->length(), id32->ptr())));
    }
    else if (ak::Identity64* id64 = dynamic_cast<ak::Identity64*>(id)) {
      obj->setid(std::shared_ptr<ak::Identity>(new ak::Identity64(id->ref(), id->fieldloc(), id->offset(), id->width(), id->length(), id64->ptr())));
    }
    else {
      throw std::runtime_error("unrecognized Identity specialization");
    }
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
          { self.length() },
          { sizeof(T) });
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

  //     .def(py::init([](py::array_t<IndexType, py::array::c_style | py::array::forcecast> array) -> ak::Index {
  //       py::buffer_info info = array.request();
  //       if (info.ndim != 1) {
  //         throw std::invalid_argument("Index must be built from a one-dimensional array; try array.ravel()");
  //       }
  //       if (info.strides[0] != sizeof(IndexType)) {
  //         throw std::invalid_argument("Index must be built from a compact array (array.strides == (array.itemsize,)); try array.copy()");
  //       }
  //       return ak::Index(std::shared_ptr<IndexType>(
  //         reinterpret_cast<IndexType*>(info.ptr),
  //         pyobject_deleter<IndexType>(array.ptr())),
  //         0,
  //         (IndexType)info.shape[0]);
  //     }))
  //
  //     .def("__repr__", [](ak::Index& self) -> const std::string {
  //       return self.repr("", "", "");
  //     })
  //     .def("__len__", &ak::Index::length)
  //
  //     .def("__getitem__", &ak::Index::get)
  //
  //     .def("__getitem__", [](ak::Index& self, py::slice slice) -> ak::Index {
  //       size_t start, stop, step, length;
  //       if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
  //         throw py::error_already_set();
  //       }
  //       return self.slice((IndexType)start, (IndexType)stop);
  //     })
  //
  // ;
  //
  // /////////////////////////////////////////////////////////////// Identity
  //
  // py::class_<ak::Identity>(m, "Identity", py::buffer_protocol())
  //     .def_buffer([](ak::Identity& self) -> py::buffer_info {
  //       return py::buffer_info(
  //         reinterpret_cast<IndexType*>(reinterpret_cast<ssize_t>(self.ptr().get()) +
  //                                          self.offset()*sizeof(IndexType)),
  //         sizeof(IndexType),
  //         py::format_descriptor<IndexType>::format(),
  //         2,
  //         { self.length(), self.keydepth() },
  //         { sizeof(IndexType)*self.keydepth(), sizeof(IndexType)}
  //       );
  //     })
  //
  //     .def_static("newref", &ak::Identity::newref)
  //
  //     .def(py::init([](RefType ref, ak::FieldLocation fieldloc, IndexType chunkdepth, IndexType indexdepth, py::array_t<IndexType, py::array::c_style | py::array::forcecast> array) {
  //       py::buffer_info info = array.request();
  //       if (info.ndim != 2) {
  //         throw std::invalid_argument("Identity must be built from a two-dimensional array");
  //       }
  //       if (ak::Identity::keydepth(chunkdepth, indexdepth) != info.shape[1]) {
  //         throw std::invalid_argument("second dimension of array must be consistent with chunkdepth and indexdepth");
  //       }
  //       if (info.strides[0] != sizeof(IndexType)*info.shape[1]  ||  info.strides[1] != sizeof(IndexType)) {
  //         throw std::invalid_argument("Identity must be built from a compact array (array.strides == (array.shape[1]*array.itemsize, array.itemsize)); try array.copy()");
  //       }
  //       return ak::Identity(ref, fieldloc, chunkdepth, indexdepth,
  //                           std::shared_ptr<IndexType>(reinterpret_cast<IndexType*>(info.ptr),
  //                                                          pyobject_deleter<IndexType>(array.ptr())),
  //                           0,
  //                           (IndexType)info.shape[0]);
  //     }))
  //
  //     .def("__repr__", [](ak::Identity& self) -> const std::string {
  //       return self.repr("", "", "");
  //     })
  //     .def("__len__", &ak::Identity::length)
  //
  //     .def_property_readonly("ref", &ak::Identity::ref)
  //     .def_property_readonly("fieldloc", &ak::Identity::fieldloc)
  //     .def_property_readonly("chunkdepth", &ak::Identity::chunkdepth)
  //     .def_property_readonly("indexdepth", &ak::Identity::indexdepth)
  //     .def_property_readonly("keydepth", [](ak::Identity& self) { return self.keydepth(); })
  //     .def_property_readonly("array", [](py::buffer& self) -> py::array {
  //       return py::array(self);
  //     })
  //
  // ;
  //
  // /////////////////////////////////////////////////////////////// Iterator
  //
  // py::class_<ak::Iterator>(m, "Iterator")
  //     .def(py::init([](ak::NumpyArray& content) -> ak::Iterator {
  //       return ak::Iterator(std::shared_ptr<ak::Content>(new ak::NumpyArray(content)));
  //     }))
  //
  //     .def(py::init([](ak::ListOffsetArray& content) -> ak::Iterator {
  //       return ak::Iterator(std::shared_ptr<ak::Content>(new ak::ListOffsetArray(content)));
  //     }))
  //
  //     .def("__next__", [](ak::Iterator& iterator) -> py::object {
  //       if (iterator.isdone()) {
  //         throw py::stop_iteration();
  //       }
  //       return unwrap(iterator.next());
  //     })
  //
  //     .def("next", [](ak::Iterator& iterator) -> py::object {
  //       if (iterator.isdone()) {
  //         throw py::stop_iteration();
  //       }
  //       return unwrap(iterator.next());
  //     })
  //
  //     .def("__repr__", [](ak::Iterator& self) -> const std::string {
  //       return self.repr("", "", "");
  //     })
  //
  // ;
  //
  // /////////////////////////////////////////////////////////////// NumpyArray
  // py::class_<ak::NumpyArray>(m, "NumpyArray", py::buffer_protocol())
  //     .def_buffer([](ak::NumpyArray& self) -> py::buffer_info {
  //       return py::buffer_info(
  //         self.byteptr(),
  //         self.itemsize(),
  //         self.format(),
  //         self.ndim(),
  //         self.shape(),
  //         self.strides()
  //       );
  //     })
  //
  //     .def(py::init([](py::array array, ak::Identity* id) -> ak::NumpyArray {
  //       py::buffer_info info = array.request();
  //       if (info.ndim == 0) {
  //         throw std::invalid_argument("NumpyArray must not be scalar; try array.reshape(1)");
  //       }
  //       if (info.shape.size() != info.ndim  ||  info.strides.size() != info.ndim) {
  //         throw std::invalid_argument("len(shape) != ndim or len(strides) != ndim");
  //       }
  //       ak::NumpyArray out = ak::NumpyArray(std::shared_ptr<ak::Identity>(nullptr), std::shared_ptr<byte>(
  //         reinterpret_cast<byte*>(info.ptr), pyobject_deleter<byte>(array.ptr())),
  //         info.shape,
  //         info.strides,
  //         0,
  //         info.itemsize,
  //         info.format);
  //       setid(&out, id);
  //       return out;
  //     }), py::arg("array"), py::arg("id") = py::none())
  //
  //     .def_property("id", [](ak::NumpyArray& self) -> ak::Identity* {
  //       return self.id().get();
  //     }, [](ak::NumpyArray& self, ak::Identity* id) -> void {
  //       setid(&self, id);
  //     }, py::return_value_policy::copy)
  //     .def("setid", [](ak::NumpyArray& self) -> void {
  //       self.setid();
  //     })
  //
  //     .def("__repr__", [](ak::NumpyArray& self) -> const std::string {
  //       return self.repr("", "", "");
  //     })
  //
  //     .def_property_readonly("shape", &ak::NumpyArray::shape)
  //     .def_property_readonly("strides", &ak::NumpyArray::strides)
  //     .def_property_readonly("itemsize", &ak::NumpyArray::itemsize)
  //     .def_property_readonly("format", &ak::NumpyArray::format)
  //     .def_property_readonly("ndim", &ak::NumpyArray::ndim)
  //     .def_property_readonly("isscalar", &ak::NumpyArray::isscalar)
  //     .def_property_readonly("isempty", &ak::NumpyArray::isempty)
  //     .def_property_readonly("iscompact", &ak::NumpyArray::iscompact)
  //
  //     .def("__len__", &ak::NumpyArray::length)
  //
  //     .def("__getitem__", [](ak::NumpyArray& self, IndexType at) -> py::object {
  //       return unwrap(self.get(at));
  //     })
  //
  //     .def("__getitem__", [](ak::NumpyArray& self, py::slice slice) -> py::object {
  //       size_t start, stop, step, length;
  //       if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
  //         throw py::error_already_set();
  //       }
  //       return unwrap(self.slice((IndexType)start, (IndexType)stop));
  //     })
  //
  //     .def("__iter__", [](ak::NumpyArray& self) -> ak::Iterator {
  //       return ak::Iterator(std::shared_ptr<ak::Content>(new ak::NumpyArray(self)));
  //     })
  //
  // ;
  //
  // /////////////////////////////////////////////////////////////// ListOffsetArray
  // py::class_<ak::ListOffsetArray>(m, "ListOffsetArray")
  //
  //     .def(py::init([](ak::Index& offsets, ak::NumpyArray* content, ak::Identity* id) -> ak::ListOffsetArray {
  //       ak::ListOffsetArray out = ak::ListOffsetArray(std::shared_ptr<ak::Identity>(nullptr), offsets, std::shared_ptr<ak::Content>(content->shallow_copy()));
  //       setid(&out, id);
  //       return out;
  //     }), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none())
  //
  //     .def(py::init([](ak::Index& offsets, ak::ListOffsetArray* content, ak::Identity* id) -> ak::ListOffsetArray {
  //       ak::ListOffsetArray out = ak::ListOffsetArray(std::shared_ptr<ak::Identity>(nullptr), offsets, std::shared_ptr<ak::Content>(content->shallow_copy()));
  //       setid(&out, id);
  //       return out;
  //     }), py::arg("offsets"), py::arg("content"), py::arg("id") = py::none())
  //
  //     .def_property_readonly("offsets", &ak::ListOffsetArray::offsets)
  //
  //     .def_property_readonly("content", [](ak::ListOffsetArray& self) -> py::object {
  //       return unwrap(self.content());
  //     })
  //
  //     .def_property("id", [](ak::ListOffsetArray& self) -> ak::Identity* {
  //       return self.id().get();
  //     }, [](ak::ListOffsetArray& self, ak::Identity* id) -> void {
  //       setid(&self, id);
  //     }, py::return_value_policy::copy)
  //     .def("setid", [](ak::ListOffsetArray& self) -> void {
  //       self.setid();
  //     })
  //
  //     .def("__repr__", [](ak::ListOffsetArray& self) -> const std::string {
  //       return self.repr("", "", "");
  //     })
  //
  //     .def("__len__", &ak::ListOffsetArray::length)
  //
  //     .def("__getitem__", [](ak::ListOffsetArray& self, IndexType at) -> py::object {
  //       return unwrap(self.get(at));
  //     })
  //
  //     .def("__getitem__", [](ak::ListOffsetArray& self, py::slice slice) -> py::object {
  //       size_t start, stop, step, length;
  //       if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
  //         throw py::error_already_set();
  //       }
  //       return unwrap(self.slice((IndexType)start, (IndexType)stop));
  //     })
  //
  //     .def("__iter__", [](ak::ListOffsetArray& self) -> ak::Iterator {
  //       return ak::Iterator(std::shared_ptr<ak::Content>(new ak::ListOffsetArray(self)));
  //     })
  //
  // ;
}
