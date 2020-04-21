// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "awkward/python/content.h"

#include "awkward/python/forms.h"

py::class_<ak::Form, std::shared_ptr<ak::Form>>
make_Form(const py::handle& m, const std::string& name) {
  return (py::class_<ak::Form, std::shared_ptr<ak::Form>>(m, name.c_str())
      .def("__eq__", [](const std::shared_ptr<ak::Form>& self,
                        const std::shared_ptr<ak::Form>& other) -> bool {
        return self.get()->equal(other);
      })
      .def("__ne__", [](const std::shared_ptr<ak::Form>& self,
                        const std::shared_ptr<ak::Form>& other) -> bool {
        return !self.get()->equal(other);
      })
  );
}

template <typename T>
py::dict
getparameters(const T& self) {
  return parameters2dict(self.parameters());
}

template <typename T>
py::object
parameter(const T& self, const std::string& key) {
  std::string cppvalue = self.parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(),
                                       cppvalue.length(),
                                       "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
py::class_<T, ak::Form>
form_methods(py::class_<T, std::shared_ptr<T>, ak::Form>& x) {
  return x.def("__repr__", &T::tostring)
          .def_property_readonly("has_identities", &T::has_identities)
          .def_property_readonly("parameters", &getparameters<T>)
          .def("parameter", &parameter<T>)
          .def("type",
               [](const T& self,
                  const std::map<std::string, std::string>& typestrs)
               -> std::shared_ptr<ak::Type> {
            return self.type(typestrs);
          })
          .def("tojson", &T::tojson)
  ;
}

py::class_<ak::BitMaskedForm, std::shared_ptr<ak::BitMaskedForm>, ak::Form>
make_BitMaskedForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::BitMaskedForm,
                      std::shared_ptr<ak::BitMaskedForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::string& mask,
                       const std::shared_ptr<ak::Form>& content,
                       bool valid_when,
                       bool lsb_order,
                       bool has_identities,
                       const py::object& parameters) -> ak::BitMaskedForm {
        return ak::BitMaskedForm(has_identities,
                                 dict2parameters(parameters),
                                 ak::Index::str2form(mask),
                                 content,
                                 valid_when,
                                 lsb_order);
      }),
           py::arg("mask"),
           py::arg("content"),
           py::arg("valid_when"),
           py::arg("lsb_order"),
           py::arg("has_identities") = false,
           py::arg("parameters") = py::none())
      .def_property_readonly("mask", [](const ak::BitMaskedForm& self)
                                     -> std::string {
        return ak::Index::form2str(self.mask());
      })
      .def_property_readonly("content", &ak::BitMaskedForm::content)
      .def_property_readonly("valid_when", &ak::BitMaskedForm::valid_when)
      .def_property_readonly("lsb_order", &ak::BitMaskedForm::lsb_order)
  );
}

// py::class_<ak::ByteMaskedForm, std::shared_ptr<ak::ByteMaskedForm>, ak::Form>
// make_ByteMaskedForm(const py::handle& m, const std::string& name) {

// }

// py::class_<ak::EmptyForm, std::shared_ptr<ak::EmptyForm>, ak::Form>
// make_EmptyForm(const py::handle& m, const std::string& name) {

// }

// py::class_<ak::IndexedForm, std::shared_ptr<ak::IndexedForm>, ak::Form>
// make_IndexedForm(const py::handle& m, const std::string& name) {

// }

// py::class_<ak::IndexedOptionForm,
//            std::shared_ptr<ak::IndexedOptionForm>,
//            ak::Form>
// make_IndexedOptionForm(const py::handle& m, const std::string& name) {

// }

// py::class_<ak::ListForm, std::shared_ptr<ak::ListForm>, ak::Form>
// make_ListForm(const py::handle& m, const std::string& name) {

// }

// py::class_<ak::ListOffsetForm, std::shared_ptr<ak::ListOffsetForm>, ak::Form>
// make_ListOffsetForm(const py::handle& m, const std::string& name) {

// }

py::class_<ak::NumpyForm, std::shared_ptr<ak::NumpyForm>, ak::Form>
make_NumpyForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::NumpyForm,
                      std::shared_ptr<ak::NumpyForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::vector<int64_t>& inner_shape,
                       int64_t itemsize,
                       const std::string& format,
                       bool has_identities,
                       const py::object& parameters) -> ak::NumpyForm {
        return ak::NumpyForm(has_identities,
                             dict2parameters(parameters),
                             inner_shape,
                             itemsize,
                             format);
      }),
           py::arg("inner_shape"),
           py::arg("itemsize"),
           py::arg("format"),
           py::arg("has_identities") = false,
           py::arg("parameters") = py::none())
      .def_property_readonly("inner_shape", &ak::NumpyForm::inner_shape)
      .def_property_readonly("itemsize", &ak::NumpyForm::itemsize)
      .def_property_readonly("format", &ak::NumpyForm::format)
  );
}

// py::class_<ak::RecordForm, std::shared_ptr<ak::RecordForm>, ak::Form>
// make_RecordForm(const py::handle& m, const std::string& name) {

// }

// py::class_<ak::RegularForm, std::shared_ptr<ak::RegularForm>, ak::Form>
// make_RegularForm(const py::handle& m, const std::string& name) {

// }

// py::class_<ak::UnionForm, std::shared_ptr<ak::UnionForm>, ak::Form>
// make_UnionForm(const py::handle& m, const std::string& name) {

// }

// py::class_<ak::UnmaskedForm, std::shared_ptr<ak::UnmaskedForm>, ak::Form>
// make_UnmaskedForm(const py::handle& m, const std::string& name) {

// }
