// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_INDEX_H_
#define AWKWARDPY_INDEX_H_

#include <string>

#include <pybind11/pybind11.h>

#include "awkward/Index.h"

namespace py = pybind11;
namespace ak = awkward;

template <typename T>
py::class_<ak::IndexOf<T>>
  make_IndexOf(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_INDEX_H_
