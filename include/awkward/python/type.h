// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_TYPE_H_
#define AWKWARDPY_TYPE_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "awkward/type/Type.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/ListType.h"
#include "awkward/type/OptionType.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/RegularType.h"
#include "awkward/type/UnionType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/python/util.h"

namespace py = pybind11;
namespace ak = awkward;

py::class_<ak::Type, std::shared_ptr<ak::Type>> make_Type(py::handle m, std::string name);
py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type> make_ArrayType(py::handle m, std::string name);
py::class_<ak::ListType, std::shared_ptr<ak::ListType>, ak::Type> make_ListType(py::handle m, std::string name);
py::class_<ak::OptionType, std::shared_ptr<ak::OptionType>, ak::Type> make_OptionType(py::handle m, std::string name);
py::class_<ak::PrimitiveType, std::shared_ptr<ak::PrimitiveType>, ak::Type> make_PrimitiveType(py::handle m, std::string name);
py::class_<ak::RecordType, std::shared_ptr<ak::RecordType>, ak::Type> make_RecordType(py::handle m, std::string name);
py::class_<ak::RegularType, std::shared_ptr<ak::RegularType>, ak::Type> make_RegularType(py::handle m, std::string name);
py::class_<ak::UnionType, std::shared_ptr<ak::UnionType>, ak::Type> make_UnionType(py::handle m, std::string name);
py::class_<ak::UnknownType, std::shared_ptr<ak::UnknownType>, ak::Type> make_UnknownType(py::handle m, std::string name);

#endif // AWKWARDPY_TYPE_H_
