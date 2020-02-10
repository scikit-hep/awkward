// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_FILLABLE_H_
#define AWKWARDPY_FILLABLE_H_

#include <pybind11/pybind11.h>

#include "awkward/fillable/FillableArray.h"

namespace py = pybind11;
namespace ak = awkward;

py::class_<ak::FillableArray> make_FillableArray(py::handle m, std::string name);

#endif // AWKWARDPY_FILLABLE_H_
