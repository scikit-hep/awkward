// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/python/boxing.h"

#include "awkward/python/type.h"

py::class_<ak::Type, std::shared_ptr<ak::Type>> make_Type(py::handle m, std::string name) {
  return (py::class_<ak::Type, std::shared_ptr<ak::Type>>(m, name.c_str())
      .def("__eq__", [](std::shared_ptr<ak::Type> self, std::shared_ptr<ak::Type> other) -> bool {
        return self.get()->equal(other, true);
      })
      .def("__ne__", [](std::shared_ptr<ak::Type> self, std::shared_ptr<ak::Type> other) -> bool {
        return !self.get()->equal(other, true);
      })
  );
}

template <typename T>
py::dict getparameters(T& self) {
  return parameters2dict(self.parameters());
}

template <typename T>
py::object parameter(T& self, std::string& key) {
  std::string cppvalue = self.parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
py::object purelist_parameter(T& self, std::string& key) {
  std::string cppvalue = self.purelist_parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
void setparameters(T& self, py::object parameters) {
  self.setparameters(dict2parameters(parameters));
}

template <typename T>
void setparameter(T& self, std::string& key, py::object value) {
  py::object valuestr = py::module::import("json").attr("dumps")(value);
  self.setparameter(key, valuestr.cast<std::string>());
}

template <typename T>
py::class_<T, ak::Type> type_methods(py::class_<T, std::shared_ptr<T>, ak::Type>& x) {
  return x.def("__repr__", &T::tostring)
          .def_property("parameters", &getparameters<T>, &setparameters<T>)
          .def("setparameter", &setparameter<T>)
          .def_property_readonly("numfields", &T::numfields)
          .def("fieldindex", &T::fieldindex)
          .def("key", &T::key)
          .def("haskey", &T::haskey)
          .def("keys", &T::keys)
          .def("empty", &T::empty)
  ;
}

/////////////////////////////////////////////////////////////// ArrayType

py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type> make_ArrayType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::ArrayType, std::shared_ptr<ak::ArrayType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::Type> type, int64_t length, py::object parameters) -> ak::ArrayType {
        return ak::ArrayType(dict2parameters(parameters), type, length);
      }), py::arg("type"), py::arg("length"), py::arg("parameters") = py::none())
      .def_property_readonly("type", &ak::ArrayType::type)
      .def_property_readonly("length", &ak::ArrayType::length)
      .def(py::pickle([](const ak::ArrayType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), box(self.type()), py::cast(self.length()));
      }, [](py::tuple state) {
        return ak::ArrayType(dict2parameters(state[0]), unbox_type(state[1]), state[2].cast<int64_t>());
      }))
  );
}

/////////////////////////////////////////////////////////////// ListType

py::class_<ak::ListType, std::shared_ptr<ak::ListType>, ak::Type> make_ListType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::ListType, std::shared_ptr<ak::ListType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::Type> type, py::object parameters) -> ak::ListType {
        return ak::ListType(dict2parameters(parameters), type);
      }), py::arg("type"), py::arg("parameters") = py::none())
      .def_property_readonly("type", &ak::ListType::type)
      .def(py::pickle([](const ak::ListType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), box(self.type()));
      }, [](py::tuple state) {
        return ak::ListType(dict2parameters(state[0]), unbox_type(state[1]));
      }))
  );
}

/////////////////////////////////////////////////////////////// OptionType

py::class_<ak::OptionType, std::shared_ptr<ak::OptionType>, ak::Type> make_OptionType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::OptionType, std::shared_ptr<ak::OptionType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::Type> type, py::object parameters) -> ak::OptionType {
        return ak::OptionType(dict2parameters(parameters), type);
      }), py::arg("type"), py::arg("parameters") = py::none())
      .def_property_readonly("type", &ak::OptionType::type)
      .def(py::pickle([](const ak::OptionType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), box(self.type()));
      }, [](py::tuple state) {
        return ak::OptionType(dict2parameters(state[0]), unbox_type(state[1]));
      }))
  );
}

/////////////////////////////////////////////////////////////// PrimitiveType

py::class_<ak::PrimitiveType, std::shared_ptr<ak::PrimitiveType>, ak::Type> make_PrimitiveType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::PrimitiveType, std::shared_ptr<ak::PrimitiveType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::string dtype, py::object parameters) -> ak::PrimitiveType {
        if (dtype == std::string("bool")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::boolean);
        }
        else if (dtype == std::string("int8")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::int8);
        }
        else if (dtype == std::string("int16")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::int16);
        }
        else if (dtype == std::string("int32")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::int32);
        }
        else if (dtype == std::string("int64")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::int64);
        }
        else if (dtype == std::string("uint8")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::uint8);
        }
        else if (dtype == std::string("uint16")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::uint16);
        }
        else if (dtype == std::string("uint32")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::uint32);
        }
        else if (dtype == std::string("uint64")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::uint64);
        }
        else if (dtype == std::string("float32")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::float32);
        }
        else if (dtype == std::string("float64")) {
          return ak::PrimitiveType(dict2parameters(parameters), ak::PrimitiveType::float64);
        }
        else {
          throw std::invalid_argument(std::string("unrecognized primitive type: ") + dtype);
        }
      }), py::arg("dtype"), py::arg("parameters") = py::none())
      .def_property_readonly("dtype", [](ak::PrimitiveType& self) -> std::string {
        switch (self.dtype()) {
          case ak::PrimitiveType::boolean: return std::string("bool");
          case ak::PrimitiveType::int8: return std::string("int8");
          case ak::PrimitiveType::int16: return std::string("int16");
          case ak::PrimitiveType::int32: return std::string("int32");
          case ak::PrimitiveType::int64: return std::string("int64");
          case ak::PrimitiveType::uint8: return std::string("uint8");
          case ak::PrimitiveType::uint16: return std::string("uint16");
          case ak::PrimitiveType::uint32: return std::string("uint32");
          case ak::PrimitiveType::uint64: return std::string("uint64");
          case ak::PrimitiveType::float32: return std::string("float32");
          case ak::PrimitiveType::float64: return std::string("float64");
          default:
          throw std::invalid_argument(std::string("unrecognized primitive type: ") + std::to_string(self.dtype()));
        }
      })
      .def(py::pickle([](const ak::PrimitiveType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), py::cast((int64_t)self.dtype()));
      }, [](py::tuple state) {
        return ak::PrimitiveType(dict2parameters(state[0]), (ak::PrimitiveType::DType)state[1].cast<int64_t>());
      }))
  );
}

/////////////////////////////////////////////////////////////// RecordType

ak::RecordType iterable_to_RecordType(py::iterable types, py::object keys, py::object parameters) {
  std::vector<std::shared_ptr<ak::Type>> out;
  for (auto x : types) {
    out.push_back(unbox_type(x));
  }
  if (keys.is(py::none())) {
    return ak::RecordType(dict2parameters(parameters), out, std::shared_ptr<ak::util::RecordLookup>(nullptr));
  }
  else {
    std::shared_ptr<ak::util::RecordLookup> recordlookup = std::make_shared<ak::util::RecordLookup>();
    for (auto x : keys.cast<py::iterable>()) {
      recordlookup.get()->push_back(x.cast<std::string>());
    }
    if (out.size() != recordlookup.get()->size()) {
      throw std::invalid_argument("if provided, 'keys' must have the same length as 'types'");
    }
    return ak::RecordType(dict2parameters(parameters), out, recordlookup);
  }
}

py::class_<ak::RecordType, std::shared_ptr<ak::RecordType>, ak::Type> make_RecordType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::RecordType, std::shared_ptr<ak::RecordType>, ak::Type>(m, name.c_str())
      .def(py::init([](py::dict types, py::object parameters) -> ak::RecordType {
        std::shared_ptr<ak::util::RecordLookup> recordlookup = std::make_shared<ak::util::RecordLookup>();
        std::vector<std::shared_ptr<ak::Type>> out;
        for (auto x : types) {
          std::string key = x.first.cast<std::string>();
          recordlookup.get()->push_back(key);
          out.push_back(unbox_type(x.second));
        }
        return ak::RecordType(dict2parameters(parameters), out, recordlookup);
      }), py::arg("types"), py::arg("parameters") = py::none())
      .def(py::init(&iterable_to_RecordType), py::arg("types"), py::arg("keys") = py::none(), py::arg("parameters") = py::none())
      .def("__getitem__", [](ak::RecordType& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("__getitem__", [](ak::RecordType& self, std::string key) -> py::object {
        return box(self.field(key));
      })
      .def_property_readonly("istuple", &ak::RecordType::istuple)
      .def_property_readonly("types", [](ak::RecordType& self) -> py::object {
        std::vector<std::shared_ptr<ak::Type>> types = self.types();
        py::tuple pytypes(types.size());
        for (size_t i = 0;  i < types.size();  i++) {
          pytypes[i] = box(types[i]);
        }
        return pytypes;
      })
      .def("field", [](ak::RecordType& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("field", [](ak::RecordType& self, std::string key) -> py::object {
        return box(self.field(key));
      })
      .def("fields", [](ak::RecordType& self) -> py::object {
        py::list out;
        for (auto item : self.fields()) {
          out.append(box(item));
        }
        return out;
      })
      .def("fielditems", [](ak::RecordType& self) -> py::object {
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
        std::shared_ptr<ak::util::RecordLookup> recordlookup = self.recordlookup();
        if (recordlookup.get() == nullptr) {
          return py::make_tuple(pytypes, py::none(), parameters2dict(self.parameters()));
        }
        else {
          py::tuple pyrecordlookup((size_t)self.numfields());
          for (size_t i = 0;  i < (size_t)self.numfields();  i++) {
            pyrecordlookup[i] = py::cast(recordlookup.get()->at(i));
          }
          return py::make_tuple(pytypes, pyrecordlookup, parameters2dict(self.parameters()));
        }
      }, [](py::tuple state) {
        return iterable_to_RecordType(state[0].cast<py::iterable>(), state[1], state[2]);
      }))
  );
}

/////////////////////////////////////////////////////////////// RegularType

py::class_<ak::RegularType, std::shared_ptr<ak::RegularType>, ak::Type> make_RegularType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::RegularType, std::shared_ptr<ak::RegularType>, ak::Type>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::Type> type, int64_t size, py::object parameters) -> ak::RegularType {
        return ak::RegularType(dict2parameters(parameters), type, size);
      }), py::arg("type"), py::arg("size"), py::arg("parameters") = py::none())
      .def_property_readonly("type", &ak::RegularType::type)
      .def_property_readonly("size", &ak::RegularType::size)
      .def(py::pickle([](const ak::RegularType& self) {
        return py::make_tuple(parameters2dict(self.parameters()), box(self.type()), py::cast(self.size()));
      }, [](py::tuple state) {
        return ak::RegularType(dict2parameters(state[0]), unbox_type(state[1]), state[2].cast<int64_t>());
      }))
  );
}

/////////////////////////////////////////////////////////////// UnionType

py::class_<ak::UnionType, std::shared_ptr<ak::UnionType>, ak::Type> make_UnionType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::UnionType, std::shared_ptr<ak::UnionType>, ak::Type>(m, name.c_str())
      .def(py::init([](py::iterable types, py::object parameters) -> ak::UnionType {
        std::vector<std::shared_ptr<ak::Type>> out;
        for (auto x : types) {
          out.push_back(unbox_type(x));
        }
        return ak::UnionType(dict2parameters(parameters), out);
      }), py::arg("types"), py::arg("parameters") = py::none())
      .def_property_readonly("numtypes", &ak::UnionType::numtypes)
      .def_property_readonly("types", [](ak::UnionType& self) -> py::tuple {
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
        return py::make_tuple(parameters2dict(self.parameters()), types);
      }, [](py::tuple state) {
        std::vector<std::shared_ptr<ak::Type>> types;
        for (auto x : state[1]) {
          types.push_back(unbox_type(x));
        }
        return ak::UnionType(dict2parameters(state[0]), types);
      }))
  );
}

/////////////////////////////////////////////////////////////// UnknownType

py::class_<ak::UnknownType, std::shared_ptr<ak::UnknownType>, ak::Type> make_UnknownType(py::handle m, std::string name) {
  return type_methods(py::class_<ak::UnknownType, std::shared_ptr<ak::UnknownType>, ak::Type>(m, name.c_str())
      .def(py::init([](py::object parameters) -> ak::UnknownType {
        return ak::UnknownType(dict2parameters(parameters));
      }), py::arg("parameters") = py::none())
      .def(py::pickle([](const ak::UnknownType& self) {
        return py::make_tuple(parameters2dict(self.parameters()));
      }, [](py::tuple state) {
        return ak::UnknownType(dict2parameters(state[0]));
      }))
  );
}
