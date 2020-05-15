// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_IDENTITIES_H_
#define AWKWARDPY_IDENTITIES_H_

#include <pybind11/pybind11.h>
#include "awkward/Identities.h"

namespace py = pybind11;
namespace ak = awkward;

/// @brief Creates a single identity as a Python tuple of integers and strings.
template <typename T>
py::tuple
  identity(const T& self);

/// @brief Makes Identities32 and Identities64 classes in Python that mirror
/// IdentitiesOf in C++.
template <typename T>
py::class_<ak::IdentitiesOf<T>>
  make_IdentitiesOf(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_IDENTITIES_H_
