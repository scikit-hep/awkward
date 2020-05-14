// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_FORMS_H_
#define AWKWARDPY_FORMS_H_

#include <pybind11/pybind11.h>

#include "awkward/Content.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/VirtualArray.h"

namespace py = pybind11;
namespace ak = awkward;

py::class_<ak::Form, std::shared_ptr<ak::Form>>
make_Form(const py::handle& m, const std::string& name);

py::class_<ak::BitMaskedForm, std::shared_ptr<ak::BitMaskedForm>, ak::Form>
make_BitMaskedForm(const py::handle& m, const std::string& name);

py::class_<ak::ByteMaskedForm, std::shared_ptr<ak::ByteMaskedForm>, ak::Form>
make_ByteMaskedForm(const py::handle& m, const std::string& name);

py::class_<ak::EmptyForm, std::shared_ptr<ak::EmptyForm>, ak::Form>
make_EmptyForm(const py::handle& m, const std::string& name);

py::class_<ak::IndexedForm, std::shared_ptr<ak::IndexedForm>, ak::Form>
make_IndexedForm(const py::handle& m, const std::string& name);

py::class_<ak::IndexedOptionForm, std::shared_ptr<ak::IndexedOptionForm>, ak::Form>
make_IndexedOptionForm(const py::handle& m, const std::string& name);

py::class_<ak::ListForm, std::shared_ptr<ak::ListForm>, ak::Form>
make_ListForm(const py::handle& m, const std::string& name);

py::class_<ak::ListOffsetForm, std::shared_ptr<ak::ListOffsetForm>, ak::Form>
make_ListOffsetForm(const py::handle& m, const std::string& name);

py::class_<ak::NumpyForm, std::shared_ptr<ak::NumpyForm>, ak::Form>
make_NumpyForm(const py::handle& m, const std::string& name);

py::class_<ak::RecordForm, std::shared_ptr<ak::RecordForm>, ak::Form>
make_RecordForm(const py::handle& m, const std::string& name);

py::class_<ak::RegularForm, std::shared_ptr<ak::RegularForm>, ak::Form>
make_RegularForm(const py::handle& m, const std::string& name);

py::class_<ak::UnionForm, std::shared_ptr<ak::UnionForm>, ak::Form>
make_UnionForm(const py::handle& m, const std::string& name);

py::class_<ak::UnmaskedForm, std::shared_ptr<ak::UnmaskedForm>, ak::Form>
make_UnmaskedForm(const py::handle& m, const std::string& name);

py::class_<ak::VirtualForm, std::shared_ptr<ak::VirtualForm>, ak::Form>
make_VirtualForm(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_FORMS_H_
