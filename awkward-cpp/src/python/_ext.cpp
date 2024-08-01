// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#include <string>

#include <pybind11/pybind11.h>

#include "awkward/python/forth.h"

namespace py = pybind11;
PYBIND11_MODULE(_ext, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  ////////// forth.h

  make_ForthMachineOf<int32_t, int32_t>(m, "ForthMachine32");
  make_ForthMachineOf<int64_t, int32_t>(m, "ForthMachine64");

}
