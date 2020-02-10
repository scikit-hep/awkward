// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/pybind11.h>

#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/cpu-kernels/getitem.h"

namespace py = pybind11;
namespace ak = awkward;

int64_t producer() {
  ak::FillableArray builder(ak::FillableOptions(1024, 2.0));

  builder.real(1.1);
  builder.real(2.2);
  builder.real(3.3);

  return 12;
}

PYBIND11_MODULE(dependent, m) {
  m.def("producer", &producer);
}
