// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iostream>

#include <pybind11/pybind11.h>

#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/cpu-kernels/getitem.h"

namespace py = pybind11;
namespace ak = awkward;

std::shared_ptr<ak::Content> producer() {
  ak::FillableArray builder(ak::FillableOptions(1024, 2.0));

  builder.real(1.1);
  builder.real(2.2);
  builder.real(3.3);

  return builder.snapshot();
}

std::string consumer(const std::shared_ptr<ak::Content>& array) {
  return array.get()->tojson(false, 10);
}

PYBIND11_MODULE(dependent, m) {
  // Ensure dependencies are loaded.
  py::module::import("awkward1");

  m.def("producer", &producer);
  m.def("consumer", &consumer);
}
