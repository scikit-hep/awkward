// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_KERNEL_UTILS_H
#define AWKWARD_KERNEL_UTILS_H

#include <string>

#include <pybind11/pybind11.h>

#include "awkward/kernel-dispatch.h"

namespace py = pybind11;
namespace ak = awkward;

py::enum_<ak::kernel::lib>
  make_lib_enum(const py::handle& m, const std::string& name);


#endif //AWKWARD_KERNEL_UTILS_H
