#include <pybind11/pybind11.h>

#include "awkward/cpu-kernels/dummy1.h"
#include "awkward/dummy2.h"

namespace py = pybind11;

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
}
