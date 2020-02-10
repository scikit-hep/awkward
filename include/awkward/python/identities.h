// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_IDENTITIES_H_
#define AWKWARDPY_IDENTITIES_H_

#include <pybind11/pybind11.h>
#include "awkward/Identities.h"

namespace py = pybind11;
namespace ak = awkward;

template <typename T>
py::tuple identity(const T& self);

template <typename T>
py::class_<ak::IdentitiesOf<T>> make_IdentitiesOf(py::handle m, std::string name);

#endif // AWKWARDPY_IDENTITIES_H_
