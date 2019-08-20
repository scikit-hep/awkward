// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "awkward/cpu-kernels/dummy1.h"
#include "awkward/dummy2.h"
#include "awkward/Index.h"

namespace py = pybind11;
namespace ak = awkward;

int dummy3(int x) {
  return dummy2(x);
}

class NumpyArray {
public:
private:
};

PYBIND11_MODULE(layout, m) {
  m.def("dummy3", &dummy3);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<ak::Index>(m, "Index")
      .def(py::init([](py::array_t<ak::IndexType, py::array::c_style | py::array::forcecast> array) {
        py::buffer_info info = array.request();
        if (info.ndim != 1) {
          throw std::invalid_argument("Index must be built from a one-dimensional array; try array.ravel()");
        }
        if (info.strides[0] != sizeof(ak::IndexType)) {
          throw std::invalid_argument("Index must be built from a compact array (array.strides == (array.itemsize,)); try array.copy()");
        }
        return ak::Index(reinterpret_cast<ak::IndexType*>(info.ptr), info.shape[0]);
      }), py::keep_alive<1, 2>())

      .def("__repr__", &ak::Index::repr)

      .def("__getitem__", &ak::Index::get)

      .def("__getitem__", [](ak::Index self, py::slice slice) {
        size_t start, stop, step, length;
        if (!slice.compute(self.len(), &start, &stop, &step, &length)) {
          throw py::error_already_set();
        }
        return self.slice(start, stop);
      })

  ;
}
