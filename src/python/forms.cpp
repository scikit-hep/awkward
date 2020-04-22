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
        return self.get()->equal(other, true, true);
      })
      .def("__ne__", [](const std::shared_ptr<ak::Form>& self,
                        const std::shared_ptr<ak::Form>& other) -> bool {
        return !self.get()->equal(other, true, true);
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
          .def("tojson", &T::tojson,
                         py::arg("pretty") = false,
                         py::arg("verbose") = true)
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
      }), py::arg("mask"),
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

py::class_<ak::ByteMaskedForm, std::shared_ptr<ak::ByteMaskedForm>, ak::Form>
make_ByteMaskedForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::ByteMaskedForm,
                      std::shared_ptr<ak::ByteMaskedForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::string& mask,
                       const std::shared_ptr<ak::Form>& content,
                       bool valid_when,
                       bool has_identities,
                       const py::object& parameters) -> ak::ByteMaskedForm {
        return ak::ByteMaskedForm(has_identities,
                                 dict2parameters(parameters),
                                 ak::Index::str2form(mask),
                                 content,
                                 valid_when);
      }), py::arg("mask"),
          py::arg("content"),
          py::arg("valid_when"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("mask", [](const ak::ByteMaskedForm& self)
                                     -> std::string {
        return ak::Index::form2str(self.mask());
      })
      .def_property_readonly("content", &ak::ByteMaskedForm::content)
      .def_property_readonly("valid_when", &ak::ByteMaskedForm::valid_when)
  );
}

py::class_<ak::EmptyForm, std::shared_ptr<ak::EmptyForm>, ak::Form>
make_EmptyForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::EmptyForm,
                      std::shared_ptr<ak::EmptyForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](bool has_identities,
                       const py::object& parameters) -> ak::EmptyForm {
        return ak::EmptyForm(has_identities,
                             dict2parameters(parameters));
      }), py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
  );
}

py::class_<ak::IndexedForm, std::shared_ptr<ak::IndexedForm>, ak::Form>
make_IndexedForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::IndexedForm,
                      std::shared_ptr<ak::IndexedForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::string& index,
                       const std::shared_ptr<ak::Form>& content,
                       bool has_identities,
                       const py::object& parameters) -> ak::IndexedForm {
        return ak::IndexedForm(has_identities,
                               dict2parameters(parameters),
                               ak::Index::str2form(index),
                               content);
      }), py::arg("index"),
          py::arg("content"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("index", [](const ak::IndexedForm& self)
                                     -> std::string {
        return ak::Index::form2str(self.index());
      })
      .def_property_readonly("content", &ak::IndexedForm::content)
  );
}

py::class_<ak::IndexedOptionForm,
           std::shared_ptr<ak::IndexedOptionForm>,
           ak::Form>
make_IndexedOptionForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::IndexedOptionForm,
                      std::shared_ptr<ak::IndexedOptionForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::string& index,
                       const std::shared_ptr<ak::Form>& content,
                       bool has_identities,
                       const py::object& parameters) -> ak::IndexedOptionForm {
        return ak::IndexedOptionForm(has_identities,
                                     dict2parameters(parameters),
                                     ak::Index::str2form(index),
                                     content);
      }), py::arg("index"),
          py::arg("content"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("index", [](const ak::IndexedOptionForm& self)
                                     -> std::string {
        return ak::Index::form2str(self.index());
      })
      .def_property_readonly("content", &ak::IndexedOptionForm::content)
  );
}

py::class_<ak::ListForm, std::shared_ptr<ak::ListForm>, ak::Form>
make_ListForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::ListForm,
                      std::shared_ptr<ak::ListForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::string& starts,
                       const std::string& stops,
                       const std::shared_ptr<ak::Form>& content,
                       bool has_identities,
                       const py::object& parameters) -> ak::ListForm {
        return ak::ListForm(has_identities,
                            dict2parameters(parameters),
                            ak::Index::str2form(starts),
                            ak::Index::str2form(stops),
                            content);
      }), py::arg("starts"),
          py::arg("stops"),
          py::arg("content"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("starts", [](const ak::ListForm& self)
                                     -> std::string {
        return ak::Index::form2str(self.starts());
      })
      .def_property_readonly("stops", [](const ak::ListForm& self)
                                     -> std::string {
        return ak::Index::form2str(self.stops());
      })
      .def_property_readonly("content", &ak::ListForm::content)
  );
}

py::class_<ak::ListOffsetForm, std::shared_ptr<ak::ListOffsetForm>, ak::Form>
make_ListOffsetForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::ListOffsetForm,
                      std::shared_ptr<ak::ListOffsetForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::string& offsets,
                       const std::shared_ptr<ak::Form>& content,
                       bool has_identities,
                       const py::object& parameters) -> ak::ListOffsetForm {
        return ak::ListOffsetForm(has_identities,
                                  dict2parameters(parameters),
                                  ak::Index::str2form(offsets),
                                  content);
      }), py::arg("offsets"),
          py::arg("content"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("offsets", [](const ak::ListOffsetForm& self)
                                     -> std::string {
        return ak::Index::form2str(self.offsets());
      })
      .def_property_readonly("content", &ak::ListOffsetForm::content)
  );
}

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
      }), py::arg("inner_shape"),
          py::arg("itemsize"),
          py::arg("format"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("inner_shape", &ak::NumpyForm::inner_shape)
      .def_property_readonly("itemsize", &ak::NumpyForm::itemsize)
      .def_property_readonly("format", &ak::NumpyForm::format)
      .def_property_readonly("primitive", &ak::NumpyForm::primitive)
  );
}

py::class_<ak::RecordForm, std::shared_ptr<ak::RecordForm>, ak::Form>
make_RecordForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::RecordForm,
                      std::shared_ptr<ak::RecordForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::vector<std::shared_ptr<ak::Form>>& contents,
                       bool has_identities,
                       const py::object& parameters) -> ak::RecordForm {
        ak::util::RecordLookupPtr recordlookup(nullptr);
        return ak::RecordForm(has_identities,
                              dict2parameters(parameters),
                              recordlookup,
                              contents);
      }), py::arg("contents"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def(py::init([](const std::map<std::string,
                                      std::shared_ptr<ak::Form>>& contents,
                       bool has_identities,
                       const py::object& parameters) -> ak::RecordForm {
        std::shared_ptr<ak::util::RecordLookup> recordlookup =
          std::make_shared<ak::util::RecordLookup>();
        std::vector<std::shared_ptr<ak::Form>> contentvec;
        for (auto pair : contents) {
          recordlookup.get()->push_back(pair.first);
          contentvec.push_back(pair.second);
        }
        return ak::RecordForm(has_identities,
                              dict2parameters(parameters),
                              recordlookup,
                              contentvec);
      }), py::arg("contents"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("contents", [](const ak::RecordForm& self)
           -> std::map<std::string, std::shared_ptr<ak::Form>> {
        std::map<std::string, std::shared_ptr<ak::Form>> out;
        for (int64_t i = 0;  i < self.numfields();  i++) {
          out[self.key(i)] = self.content(i);
        }
        return out;
      })
      .def_property_readonly("istuple", &ak::RecordForm::istuple)
      .def_property_readonly("numfields", &ak::RecordForm::numfields)
      .def("fieldindex", &ak::RecordForm::fieldindex)
      .def("key", &ak::RecordForm::key)
      .def("haskey", &ak::RecordForm::haskey)
      .def("keys", &ak::RecordForm::keys)
      .def("content", [](const ak::RecordForm& self,
                         int64_t fieldindex) -> ak::FormPtr {
        return self.content(fieldindex);
      })
      .def("content", [](const ak::RecordForm& self,
                         const std::string& fieldindex) -> ak::FormPtr {
        return self.content(fieldindex);
      })
      .def("items", &ak::RecordForm::items)
      .def("values", &ak::RecordForm::contents)
  );
}

py::class_<ak::RegularForm, std::shared_ptr<ak::RegularForm>, ak::Form>
make_RegularForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::RegularForm,
                      std::shared_ptr<ak::RegularForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::shared_ptr<ak::Form>& content,
                       int64_t size,
                       bool has_identities,
                       const py::object& parameters) -> ak::RegularForm {
        return ak::RegularForm(has_identities,
                               dict2parameters(parameters),
                               content,
                               size);
      }), py::arg("content"),
          py::arg("size"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("content", &ak::RegularForm::content)
      .def_property_readonly("size", &ak::RegularForm::size)
  );
}

py::class_<ak::UnionForm, std::shared_ptr<ak::UnionForm>, ak::Form>
make_UnionForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::UnionForm,
                      std::shared_ptr<ak::UnionForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::string& tags,
                       const std::string& index,
                       const std::vector<std::shared_ptr<ak::Form>>& contents,
                       bool has_identities,
                       const py::object& parameters) -> ak::UnionForm {
        return ak::UnionForm(has_identities,
                             dict2parameters(parameters),
                             ak::Index::str2form(tags),
                             ak::Index::str2form(index),
                             contents);
      }), py::arg("tags"),
          py::arg("index"),
          py::arg("contents"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("tags", [](const ak::UnionForm& self)
                                     -> std::string {
        return ak::Index::form2str(self.tags());
      })
      .def_property_readonly("index", [](const ak::UnionForm& self)
                                     -> std::string {
        return ak::Index::form2str(self.index());
      })
      .def_property_readonly("contents", &ak::UnionForm::contents)
      .def_property_readonly("numcontents", &ak::UnionForm::numcontents)
      .def("content", &ak::UnionForm::content)
  );
}

py::class_<ak::UnmaskedForm, std::shared_ptr<ak::UnmaskedForm>, ak::Form>
make_UnmaskedForm(const py::handle& m, const std::string& name) {
  return form_methods(py::class_<ak::UnmaskedForm,
                      std::shared_ptr<ak::UnmaskedForm>,
                      ak::Form>(m, name.c_str())
      .def(py::init([](const std::shared_ptr<ak::Form>& content,
                       bool has_identities,
                       const py::object& parameters) -> ak::UnmaskedForm {
        return ak::UnmaskedForm(has_identities,
                                dict2parameters(parameters),
                                content);
      }), py::arg("content"),
          py::arg("has_identities") = false,
          py::arg("parameters") = py::none())
      .def_property_readonly("content", &ak::UnmaskedForm::content)
  );
}
