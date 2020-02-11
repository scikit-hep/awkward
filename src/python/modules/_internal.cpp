// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/pybind11.h>

#include "awkward/python/slice.h"
#include "awkward/python/io.h"

namespace py = pybind11;
PYBIND11_MODULE(_internal, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  /////////////////////////////////////////////////////////////// slice.h

  make_slice_tostring(m, "slice_tostring");

  /////////////////////////////////////////////////////////////// io.h

  make_fromjson(m, "fromjson");
  make_fromroot_nestedvector(m, "fromroot_nestedvector");

}
