// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_INDEX_H_
#define AWKWARDPY_INDEX_H_

#include <string>

#include <pybind11/pybind11.h>

#include "awkward/Index.h"

namespace py = pybind11;
namespace ak = awkward;

/// @brief Makes Index32, IndexU32, Index64 classes in Python that mirror
/// IndexOf in C++.
template <typename T>
py::class_<ak::IndexOf<T>>
  make_IndexOf(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_INDEX_H_
