// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARDPY_FORTH_H_
#define AWKWARDPY_FORTH_H_

#include <pybind11/pybind11.h>

#include "awkward/forth/ForthMachine.h"

namespace py = pybind11;
namespace ak = awkward;

template <typename T, typename I>
py::class_<ak::ForthMachineOf<T, I>, std::shared_ptr<ak::ForthMachineOf<T, I>>>
make_ForthMachineOf(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_FORTH_H_
