// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_BOXING_H_
#define AWKWARDPY_BOXING_H_

#include <pybind11/pybind11.h>

#include "awkward/Content.h"
#include "awkward/Identities.h"
#include "awkward/type/Type.h"

namespace py = pybind11;
namespace ak = awkward;

py::object box(std::shared_ptr<ak::Content> content);
py::object box(std::shared_ptr<ak::Identities> identities);
py::object box(std::shared_ptr<ak::Type> t);
std::shared_ptr<ak::Content> unbox_content(py::handle obj);
std::shared_ptr<ak::Identities> unbox_identities_none(py::handle obj);
std::shared_ptr<ak::Identities> unbox_identities(py::handle obj);
std::shared_ptr<ak::Type> unbox_type(py::handle obj);

#endif // AWKWARDPY_BOXING_H_
