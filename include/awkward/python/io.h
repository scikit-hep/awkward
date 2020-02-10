// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_IO_H_
#define AWKWARDPY_IO_H_

#include <string>

#include <pybind11/pybind11.h>

#include "awkward/io/json.h"
#include "awkward/io/root.h"

namespace py = pybind11;

void make_fromjson(py::module m, std::string name);
void make_fromroot_nestedvector(py::module m, std::string name);

#endif // AWKWARDPY_IO_H_
