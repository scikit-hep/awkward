// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "awkward/cpu-kernels/dummy1.h"
#include "awkward/dummy2.h"

#include "awkward/Index.h"
#include "awkward/Content.h"
#include "awkward/NumpyArray.h"
#include "awkward/ListOffsetArray.h"

namespace py = pybind11;
namespace ak = awkward;

int dummy3(int x) {
  return dummy2(x);
}

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
  else if (ak::ListOffsetArray* x = dynamic_cast<ak::ListOffsetArray*>(content.get())) {
    return py::cast(*x);
  }
  else {
    assert(false  &&  "missing unwrapper for Content subtype");
    return py::none();
  }
}

PYBIND11_MODULE(layout, m) {
  m.def("dummy3", &dummy3);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  /////////////////////////////////////////////////////////////// Index
  py::class_<ak::Index>(m, "Index", py::buffer_protocol())
      .def_buffer([](ak::Index& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<ak::IndexType*>(reinterpret_cast<ssize_t>(self.ptr().get()) +
                                           self.offset()*sizeof(ak::IndexType)),
          sizeof(ak::IndexType),
          py::format_descriptor<ak::IndexType>::format(),
          1,
          { self.length() },
          { sizeof(ak::IndexType) }
        );
      })

      .def(py::init([](py::array_t<ak::IndexType, py::array::c_style | py::array::forcecast> array) -> ak::Index {
        py::buffer_info info = array.request();
        if (info.ndim != 1) {
          throw std::invalid_argument("Index must be built from a one-dimensional array; try array.ravel()");
        }
        if (info.strides[0] != sizeof(ak::IndexType)) {
          throw std::invalid_argument("Index must be built from a compact array (array.strides == (array.itemsize,)); try array.copy()");
        }
        return ak::Index(std::shared_ptr<ak::IndexType>(
          reinterpret_cast<ak::IndexType*>(info.ptr),
          pyobject_deleter<ak::IndexType>(array.ptr())),
          0,
          reinterpret_cast<ak::IndexType>(info.shape[0]));
      }))

      .def("__repr__", [](ak::Index& self) -> const std::string {
        return self.repr("", "", "");
      })

      .def("__getitem__", &ak::Index::get)

      .def("__getitem__", [](ak::Index& self, py::slice slice) -> ak::Index {
        size_t start, stop, step, length;
        if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
          throw py::error_already_set();
        }
        return self.slice(reinterpret_cast<ak::AtType>(start), reinterpret_cast<ak::AtType>(stop));
      })

  ;

  /////////////////////////////////////////////////////////////// NumpyArray
  py::class_<ak::NumpyArray>(m, "NumpyArray", py::buffer_protocol())
      .def_buffer([](ak::NumpyArray& self) -> py::buffer_info {
        return py::buffer_info(
          self.byteptr(),
          self.itemsize(),
          self.format(),
          self.ndim(),
          self.shape(),
          self.strides()
        );
      })

      .def(py::init([](py::array array) -> ak::NumpyArray {
        py::buffer_info info = array.request();
        if (info.ndim == 0) {
          throw std::invalid_argument("NumpyArray must not be scalar; try array.reshape(1)");
        }
        if (info.shape.size() != info.ndim  ||  info.strides.size() != info.ndim) {
          throw std::invalid_argument("len(shape) != ndim or len(strides) != ndim");
        }
        return ak::NumpyArray(std::shared_ptr<ak::byte>(
          reinterpret_cast<ak::byte*>(info.ptr), pyobject_deleter<ak::byte>(array.ptr())),
          info.shape,
          info.strides,
          0,
          info.itemsize,
          info.format);
      }))

      .def("__repr__", [](ak::NumpyArray& self) -> const std::string {
        return self.repr("", "", "");
      })

      .def("shape", &ak::NumpyArray::shape)
      .def("strides", &ak::NumpyArray::strides)
      .def("itemsize", &ak::NumpyArray::itemsize)
      .def("format", &ak::NumpyArray::format)
      .def("ndim", &ak::NumpyArray::ndim)
      .def("isscalar", &ak::NumpyArray::isscalar)
      .def("isempty", &ak::NumpyArray::isempty)
      .def("iscompact", &ak::NumpyArray::iscompact)

      .def("__getitem__", [](ak::NumpyArray& self, ak::AtType at) -> py::object {
        return unwrap(self.get(at));
      })

      .def("__getitem__", [](ak::NumpyArray& self, py::slice slice) -> py::object {
        size_t start, stop, step, length;
        if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
          throw py::error_already_set();
        }
        return unwrap(self.slice(reinterpret_cast<AtType>(start), reinterpret_cast<AtType>(stop)));
      })

  ;

  /////////////////////////////////////////////////////////////// ListOffsetArray
  py::class_<ak::ListOffsetArray>(m, "ListOffsetArray")

      .def(py::init([](ak::Index& offsets, ak::NumpyArray& content) -> ak::ListOffsetArray {
        return ak::ListOffsetArray(offsets, std::shared_ptr<ak::Content>(new ak::NumpyArray(content)));
      }))

      .def(py::init([](ak::Index& offsets, ak::ListOffsetArray& content) -> ak::ListOffsetArray {
        return ak::ListOffsetArray(offsets, std::shared_ptr<ak::Content>(new ak::ListOffsetArray(content)));
      }))

      .def ("offsets", &ak::ListOffsetArray::offsets)

      .def ("content", [](ak::ListOffsetArray& self) -> py::object {
        return unwrap(self.content());
      })

      .def("__repr__", [](ak::ListOffsetArray& self) -> const std::string {
        return self.repr("", "", "");
      })

      .def("__getitem__", [](ak::ListOffsetArray& self, ak::AtType at) -> py::object {
        return unwrap(self.get(at));
      })

      .def("__getitem__", [](ak::ListOffsetArray& self, py::slice slice) -> py::object {
        size_t start, stop, step, length;
        if (!slice.compute(self.length(), &start, &stop, &step, &length)) {
          throw py::error_already_set();
        }
        return unwrap(self.slice(reinterpret_cast<AtType>(start), reinterpret_cast<AtType>(stop)));
      })

  ;
}
