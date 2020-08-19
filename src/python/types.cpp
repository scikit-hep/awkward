// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/types.cpp", line)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "awkward/python/content.h"

#include "awkward/python/types.h"

////////// boxing

py::object
box(const std::shared_ptr<ak::Type>& t) {
  if (ak::ArrayType* raw =
      dynamic_cast<ak::ArrayType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListType* raw =
           dynamic_cast<ak::ListType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::OptionType* raw =
           dynamic_cast<ak::OptionType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::PrimitiveType* raw =
           dynamic_cast<ak::PrimitiveType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::RecordType* raw =
           dynamic_cast<ak::RecordType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::RegularType* raw =
           dynamic_cast<ak::RegularType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionType* raw =
           dynamic_cast<ak::UnionType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnknownType* raw =
           dynamic_cast<ak::UnknownType*>(t.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error(
      std::string("missing boxer for Type subtype") + FILENAME(__LINE__));
  }
}

std::shared_ptr<ak::Type>
unbox_type(const py::handle& obj) {
  try {
    return obj.cast<ak::ArrayType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::OptionType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::PrimitiveType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RecordType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RegularType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnionType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnknownType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument(
    std::string("argument must be a Type subtype") + FILENAME(__LINE__));
}

////////// Type

py::class_<ak::Type, std::shared_ptr<ak::Type>>
make_Type(const py::handle& m, const std::string& name) {
  return (py::class_<ak::Type, std::shared_ptr<ak::Type>>(m, name.c_str())
      .def("__eq__", [](const std::shared_ptr<ak::Type>& self,
                        const std::shared_ptr<ak::Type>& other) -> bool {
        return self.get()->equal(other, true);
      })
      .def("__ne__", [](const std::shared_ptr<ak::Type>& self,
                        const std::shared_ptr<ak::Type>& other) -> bool {
        return !self.get()->equal(other, true);
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
py::object
purelist_parameter(const T& self, const std::string& key) {
  std::string cppvalue = self.purelist_parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(),
                                       cppvalue.length(),
                                       "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
void
setparameters(T& self, const py::object& parameters) {
  self.setparameters(dict2parameters(parameters));
}

template <typename T>
void
setparameter(T& self, const std::string& key, const py::object& value) {
  py::object valuestr = py::module::import("json").attr("dumps")(value);
  self.setparameter(key, valuestr.cast<std::string>());
}

std::string
typestr2str(const py::object& in) {
  if (in.is(py::none())) {
    return std::string();
  }
  else {
    return in.cast<std::string>();
  }
}

py::object
str2typestr(const std::string& in) {
  if (in.empty()) {
    return py::none();
  }
  else {
    py::str pyvalue(PyUnicode_DecodeUTF8(in.data(),
                                         in.length(),
                                         "surrogateescape"));
    return pyvalue;
  }
}

template <typename T>
py::class_<T, ak::Type>
type_methods(py::class_<T, std::shared_ptr<T>, ak::Type>& x) {
  return x.def("__repr__", &T::tostring)
          .def_property("parameters", &getparameters<T>, &setparameters<T>)
          .def("setparameter", &setparameter<T>)
          .def_property_readonly("typestr",
                                 [](const ak::Type& self) -> py::object {
            return str2typestr(self.typestr());
          })
          .def_property_readonly("numfields", &T::numfields)
          .def("fieldindex", &T::fieldindex)
          .def("key", &T::key)
          .def("haskey", &T::haskey)
          .def("keys", &T::keys)
          .def("empty", &T::empty)
  ;
}

////////// ArrayType

py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type>
make_ArrayType(const py::handle& m, const std::string& name) {
  return type_methods(py::class_<ak::ArrayType,
                      std::shared_ptr<ak::ArrayType>,
                      ak::Type>(m, name.c_str())
      .def(py::init([](const std::shared_ptr<ak::Type>& type,
                       int64_t length,
                       const py::object& parameters,
                       const py::object& typestr) -> ak::ArrayType {
        return ak::ArrayType(dict2parameters(parameters),
                             typestr2str(typestr),
                             type,
                             length);
      }), py::arg("type"),
           py::arg("length"),
           py::arg("parameters") = py::none(),
           py::arg("typestr") = py::none())
      .def_property_readonly("type", &ak::ArrayType::type)
      .def_property_readonly("length", &ak::ArrayType::length)
      .def(py::pickle([](const ak::ArrayType& self) {
        return py::make_tuple(parameters2dict(self.parameters()),
                              str2typestr(self.typestr()),
                              box(self.type()),
                              py::cast(self.length()));
      }, [](const py::tuple& state) {
        return ak::ArrayType(dict2parameters(state[0]),
                             typestr2str(state[1]),
                             unbox_type(state[2]),
                             state[3].cast<int64_t>());
      }))
  );
}

////////// ListType

py::class_<ak::ListType, std::shared_ptr<ak::ListType>, ak::Type>
make_ListType(const py::handle& m, const std::string& name) {
  return type_methods(py::class_<ak::ListType,
                      std::shared_ptr<ak::ListType>,
                      ak::Type>(m, name.c_str())
      .def(py::init([](const std::shared_ptr<ak::Type>& type,
                       const py::object& parameters,
                       const py::object& typestr) -> ak::ListType {
        return ak::ListType(dict2parameters(parameters),
                            typestr2str(typestr),
                            type);
      }),
           py::arg("type"),
           py::arg("parameters") = py::none(),
           py::arg("typestr") = py::none())
      .def_property_readonly("type", &ak::ListType::type)
      .def(py::pickle([](const ak::ListType& self) {
        return py::make_tuple(parameters2dict(self.parameters()),
                              str2typestr(self.typestr()),
                              box(self.type()));
      }, [](const py::tuple& state) {
        return ak::ListType(dict2parameters(state[0]),
                            typestr2str(state[1]),
                            unbox_type(state[2]));
      }))
  );
}

////////// OptionType

py::class_<ak::OptionType, std::shared_ptr<ak::OptionType>, ak::Type>
make_OptionType(const py::handle& m, const std::string& name) {
  return type_methods(py::class_<ak::OptionType,
                      std::shared_ptr<ak::OptionType>,
                      ak::Type>(m, name.c_str())
      .def(py::init([](const std::shared_ptr<ak::Type>& type,
                       const py::object& parameters,
                       const py::object& typestr) -> ak::OptionType {
        return ak::OptionType(dict2parameters(parameters),
                              typestr2str(typestr),
                              type);
      }), py::arg("type"),
          py::arg("parameters") = py::none(),
          py::arg("typestr") = py::none())
      .def_property_readonly("type", &ak::OptionType::type)
      .def(py::pickle([](const ak::OptionType& self) {
        return py::make_tuple(parameters2dict(self.parameters()),
                              str2typestr(self.typestr()),
                              box(self.type()));
      }, [](const py::tuple& state) {
        return ak::OptionType(dict2parameters(state[0]),
                              typestr2str(state[1]),
                              unbox_type(state[2]));
      }))
  );
}

////////// PrimitiveType

py::class_<ak::PrimitiveType, std::shared_ptr<ak::PrimitiveType>, ak::Type>
make_PrimitiveType(const py::handle& m, const std::string& name) {
  return type_methods(py::class_<ak::PrimitiveType,
                      std::shared_ptr<ak::PrimitiveType>,
                      ak::Type>(m, name.c_str())
      .def(py::init([](const std::string& dtype,
                       const py::object& parameters,
                       const py::object& typestr) -> ak::PrimitiveType {
        ak::util::dtype dt = ak::util::name_to_dtype(dtype);
        if (dt == ak::util::dtype::NOT_PRIMITIVE) {
          throw std::invalid_argument(
            std::string("unrecognized primitive type: ") + dtype
            + FILENAME(__LINE__));
        }
        return ak::PrimitiveType(dict2parameters(parameters),
                                 typestr2str(typestr),
                                 dt);
      }), py::arg("dtype"),
          py::arg("parameters") = py::none(),
          py::arg("typestr") = py::none())
      .def_property_readonly("dtype",
                             [](const ak::PrimitiveType& self) -> std::string {
        return ak::util::dtype_to_name(self.dtype());
      })
      .def(py::pickle([](const ak::PrimitiveType& self) {
        return py::make_tuple(parameters2dict(self.parameters()),
                              str2typestr(self.typestr()),
                              py::cast((int64_t)self.dtype()));
      }, [](const py::tuple& state) {
        return ak::PrimitiveType(
          dict2parameters(state[0]),
          typestr2str(state[1]),
          (ak::util::dtype)state[2].cast<int64_t>());
      }))
  );
}

////////// RecordType

ak::RecordType
iterable_to_RecordType(const py::iterable& types,
                       const py::object& keys,
                       const py::object& parameters,
                       const py::object& typestr) {
  std::vector<std::shared_ptr<ak::Type>> out;
  for (auto x : types) {
    out.push_back(unbox_type(x));
  }
  if (keys.is(py::none())) {
    return ak::RecordType(dict2parameters(parameters),
                          typestr2str(typestr),
                          out,
                          std::shared_ptr<ak::util::RecordLookup>(nullptr));
  }
  else {
    std::shared_ptr<ak::util::RecordLookup> recordlookup =
      std::make_shared<ak::util::RecordLookup>();
    for (auto x : keys.cast<py::iterable>()) {
      recordlookup.get()->push_back(x.cast<std::string>());
    }
    if (out.size() != recordlookup.get()->size()) {
      throw std::invalid_argument(
        std::string("if provided, 'keys' must have the same length as 'types'")
        + FILENAME(__LINE__));
    }
    return ak::RecordType(dict2parameters(parameters),
                          typestr2str(typestr),
                          out,
                          recordlookup);
  }
}

py::class_<ak::RecordType, std::shared_ptr<ak::RecordType>, ak::Type>
make_RecordType(const py::handle& m, const std::string& name) {
  return type_methods(py::class_<ak::RecordType,
                      std::shared_ptr<ak::RecordType>,
                      ak::Type>(m, name.c_str())
      .def(py::init([](const py::dict& types,
                       const py::object& parameters,
                       const py::object& typestr) -> ak::RecordType {
        std::shared_ptr<ak::util::RecordLookup> recordlookup =
          std::make_shared<ak::util::RecordLookup>();
        std::vector<std::shared_ptr<ak::Type>> out;
        for (auto x : types) {
          std::string key = x.first.cast<std::string>();
          recordlookup.get()->push_back(key);
          out.push_back(unbox_type(x.second));
        }
        return ak::RecordType(dict2parameters(parameters),
                              typestr2str(typestr),
                              out, recordlookup);
      }), py::arg("types"),
          py::arg("parameters") = py::none(),
          py::arg("typestr") = py::none())
      .def(py::init(&iterable_to_RecordType),
           py::arg("types"),
           py::arg("keys") = py::none(),
           py::arg("parameters") = py::none(),
           py::arg("typestr") = py::none())
      .def("__getitem__",
           [](const ak::RecordType& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("__getitem__",
           [](const ak::RecordType& self, const std::string& key)
           -> py::object {
        return box(self.field(key));
      })
      .def_property_readonly("istuple", &ak::RecordType::istuple)
      .def_property_readonly("types",
                             [](const ak::RecordType& self) -> py::object {
        std::vector<std::shared_ptr<ak::Type>> types = self.types();
        py::tuple pytypes(types.size());
        for (size_t i = 0;  i < types.size();  i++) {
          pytypes[i] = box(types[i]);
        }
        return pytypes;
      })
      .def("field",
           [](const ak::RecordType& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("field",
           [](const ak::RecordType& self, const std::string& key)
           -> py::object {
        return box(self.field(key));
      })
      .def("fields", [](const ak::RecordType& self) -> py::object {
        py::list out;
        for (auto item : self.fields()) {
          out.append(box(item));
        }
        return out;
      })
      .def("fielditems", [](const ak::RecordType& self) -> py::object {
        py::list out;
        for (auto item : self.fielditems()) {
          py::str key(item.first);
          py::object val(box(item.second));
          py::tuple pair(2);
          pair[0] = key;
          pair[1] = val;
          out.append(pair);
        }
        return out;
      })
      .def(py::pickle([](const ak::RecordType& self) {
        py::tuple pytypes((size_t)self.numfields());
        for (int64_t i = 0;  i < self.numfields();  i++) {
          pytypes[(size_t)i] = box(self.field(i));
        }
        std::shared_ptr<ak::util::RecordLookup> recordlookup =
          self.recordlookup();
        if (recordlookup.get() == nullptr) {
          return py::make_tuple(pytypes,
                                py::none(),
                                parameters2dict(self.parameters()),
                                str2typestr(self.typestr()));
        }
        else {
          py::tuple pyrecordlookup((size_t)self.numfields());
          for (size_t i = 0;  i < (size_t)self.numfields();  i++) {
            pyrecordlookup[i] = py::cast(recordlookup.get()->at(i));
          }
          return py::make_tuple(pytypes,
                                pyrecordlookup,
                                parameters2dict(self.parameters()),
                                str2typestr(self.typestr()));
        }
      }, [](const py::tuple& state) {
        return iterable_to_RecordType(state[0].cast<py::iterable>(),
                                      state[1],
                                      state[2],
                                      state[3]);
      }))
  );
}

////////// RegularType

py::class_<ak::RegularType, std::shared_ptr<ak::RegularType>, ak::Type>
make_RegularType(const py::handle& m, const std::string& name) {
  return type_methods(py::class_<ak::RegularType,
                      std::shared_ptr<ak::RegularType>,
                      ak::Type>(m, name.c_str())
      .def(py::init([](const std::shared_ptr<ak::Type>& type,
                       int64_t size,
                       const py::object& parameters,
                       const py::object& typestr) -> ak::RegularType {
        return ak::RegularType(dict2parameters(parameters),
                               typestr2str(typestr),
                               type,
                               size);
      }), py::arg("type"),
          py::arg("size"),
          py::arg("parameters") = py::none(),
          py::arg("typestr") = py::none())
      .def_property_readonly("type", &ak::RegularType::type)
      .def_property_readonly("size", &ak::RegularType::size)
      .def(py::pickle([](const ak::RegularType& self) {
        return py::make_tuple(parameters2dict(self.parameters()),
                              str2typestr(self.typestr()),
                              box(self.type()),
                              py::cast(self.size()));
      }, [](const py::tuple& state) {
        return ak::RegularType(dict2parameters(state[0]),
                               typestr2str(state[1]),
                               unbox_type(state[2]),
                               state[3].cast<int64_t>());
      }))
  );
}

////////// UnionType

py::class_<ak::UnionType, std::shared_ptr<ak::UnionType>, ak::Type>
make_UnionType(const py::handle& m, const std::string& name) {
  return type_methods(py::class_<ak::UnionType,
                      std::shared_ptr<ak::UnionType>,
                      ak::Type>(m, name.c_str())
      .def(py::init([](const py::iterable& types,
                       const py::object& parameters,
                       const py::object& typestr) -> ak::UnionType {
        std::vector<std::shared_ptr<ak::Type>> out;
        for (auto x : types) {
          out.push_back(unbox_type(x));
        }
        return ak::UnionType(dict2parameters(parameters),
                             typestr2str(typestr),
                             out);
      }), py::arg("types"),
          py::arg("parameters") = py::none(),
          py::arg("typestr") = py::none())
      .def_property_readonly("numtypes", &ak::UnionType::numtypes)
      .def_property_readonly("types",
                             [](const ak::UnionType& self) -> py::tuple {
        py::tuple types((size_t)self.numtypes());
        for (int64_t i = 0;  i < self.numtypes();  i++) {
          types[(size_t)i] = box(self.type(i));
        }
        return types;
      })
      .def("type", &ak::UnionType::type)
      .def(py::pickle([](const ak::UnionType& self) {
        py::tuple types((size_t)self.numtypes());
        for (int64_t i = 0;  i < self.numtypes();  i++) {
          types[(size_t)i] = box(self.type(i));
        }
        return py::make_tuple(parameters2dict(self.parameters()),
                              str2typestr(self.typestr()),
                              types);
      }, [](const py::tuple& state) {
        std::vector<std::shared_ptr<ak::Type>> types;
        for (auto x : state[2]) {
          types.push_back(unbox_type(x));
        }
        return ak::UnionType(dict2parameters(state[0]),
                             typestr2str(state[1]),
                             types);
      }))
  );
}

////////// UnknownType

py::class_<ak::UnknownType, std::shared_ptr<ak::UnknownType>, ak::Type>
make_UnknownType(const py::handle& m, const std::string& name) {
  return type_methods(py::class_<ak::UnknownType,
                      std::shared_ptr<ak::UnknownType>,
                      ak::Type>(m, name.c_str())
      .def(py::init([](const py::object& parameters,
                       const py::object& typestr) -> ak::UnknownType {
        return ak::UnknownType(dict2parameters(parameters),
                               typestr2str(typestr));
      }), py::arg("parameters") = py::none(), py::arg("typestr") = py::none())
      .def(py::pickle([](const ak::UnknownType& self) {
        return py::make_tuple(parameters2dict(self.parameters()),
                              str2typestr(self.typestr()));
      }, [](const py::tuple& state) {
        return ak::UnknownType(dict2parameters(state[0]),
                               typestr2str(state[1]));
      }))
  );
}
