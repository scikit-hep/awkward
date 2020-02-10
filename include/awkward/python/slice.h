// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_SLICE_H_
#define AWKWARDPY_SLICE_H_

#include <pybind11/pybind11.h>

#include "awkward/Slice.h"

namespace py = pybind11;
namespace ak = awkward;

PYBIND11_EXPORT int64_t testy(int64_t x, int64_t y);

ak::Slice toslice(py::object obj);

void make_slice_tostring(py::module& m, const std::string& name);

#endif // AWKWARDPY_SLICE_H_
