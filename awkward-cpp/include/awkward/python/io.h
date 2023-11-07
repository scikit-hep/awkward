// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARDPY_IO_H_
#define AWKWARDPY_IO_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void
make_fromjsonobj(py::module& m, const std::string& name);

void
make_fromjsonobj_schema(py::module& m, const std::string& name);

#endif // AWKWARDPY_IO_H_
