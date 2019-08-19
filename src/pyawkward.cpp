#include <pybind11/pybind11.h>

#include "awkward/cpu-kernels/dummy1.h"
#include "awkward/dummy2.h"
#include "awkward/Index.h"

namespace py = pybind11;
namespace ak = awkward;

int dummy3(int x) {
  return dummy2(x);
}

PYBIND11_MODULE(layout, m) {
  m.def("dummy3", &dummy3);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  // py::class_<ak::Index>(m, "Index")
  //     .def(py::init([](const Index &self,
  //                         py::init<py::array_t<ak::INDEXTYPE, py::array::c_style | py::array::forcecast> array) {
  //       // HERE
  //     }))
  //     .def(>())
  //     .def("__getitem__", &ak::Index::GetItem);
}
