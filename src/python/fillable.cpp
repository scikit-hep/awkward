// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Iterator.h"
#include "awkward/fillable/FillableOptions.h"

#include "awkward/python/boxing.h"
#include "awkward/python/util.h"

#include "awkward/python/fillable.h"

void fillable_fill(ak::FillableArray& self, py::handle obj) {
  if (obj.is(py::none())) {
    self.null();
  }
  else if (py::isinstance<py::bool_>(obj)) {
    self.boolean(obj.cast<bool>());
  }
  else if (py::isinstance<py::int_>(obj)) {
    self.integer(obj.cast<int64_t>());
  }
  else if (py::isinstance<py::float_>(obj)) {
    self.real(obj.cast<double>());
  }
  else if (py::isinstance<py::bytes>(obj)) {
    self.bytestring(obj.cast<std::string>());
  }
  else if (py::isinstance<py::str>(obj)) {
    self.string(obj.cast<std::string>());
  }
  else if (py::isinstance<py::tuple>(obj)) {
    py::tuple tup = obj.cast<py::tuple>();
    self.begintuple(tup.size());
    for (size_t i = 0;  i < tup.size();  i++) {
      self.index((int64_t)i);
      fillable_fill(self, tup[i]);
    }
    self.endtuple();
  }
  else if (py::isinstance<py::dict>(obj)) {
    py::dict dict = obj.cast<py::dict>();
    self.beginrecord();
    for (auto pair : dict) {
      if (!py::isinstance<py::str>(pair.first)) {
        throw std::invalid_argument("keys of dicts in 'fromiter' must all be strings");
      }
      std::string key = pair.first.cast<std::string>();
      self.field_check(key.c_str());
      fillable_fill(self, pair.second);
    }
    self.endrecord();
  }
  else if (py::isinstance<py::iterable>(obj)) {
    py::iterable seq = obj.cast<py::iterable>();
    self.beginlist();
    for (auto x : seq) {
      fillable_fill(self, x);
    }
    self.endlist();
  }
  else {
    throw std::invalid_argument(std::string("cannot convert ") + obj.attr("__repr__")().cast<std::string>() + std::string(" to an array element"));
  }
}

py::class_<ak::FillableArray> make_FillableArray(py::handle m, std::string name) {
  return (py::class_<ak::FillableArray>(m, name.c_str())
      .def(py::init([](int64_t initial, double resize) -> ak::FillableArray {
        return ak::FillableArray(ak::FillableOptions(initial, resize));
      }), py::arg("initial") = 1024, py::arg("resize") = 2.0)
      .def_property_readonly("_ptr", [](ak::FillableArray* self) -> size_t { return reinterpret_cast<size_t>(self); })
      .def("__repr__", &ak::FillableArray::tostring)
      .def("__len__", &ak::FillableArray::length)
      .def("clear", &ak::FillableArray::clear)
      .def_property_readonly("type", &ak::FillableArray::type)
      .def("snapshot", [](ak::FillableArray& self) -> py::object {
        return box(self.snapshot());
      })
      .def("__getitem__", &getitem<ak::FillableArray>)
      .def("__iter__", [](ak::FillableArray& self) -> ak::Iterator {
        return ak::Iterator(self.snapshot());
      })
      .def("null", &ak::FillableArray::null)
      .def("boolean", &ak::FillableArray::boolean)
      .def("integer", &ak::FillableArray::integer)
      .def("real", &ak::FillableArray::real)
      .def("bytestring", [](ak::FillableArray& self, py::bytes x) -> void {
        self.bytestring(x.cast<std::string>());
      })
      .def("string", [](ak::FillableArray& self, py::str x) -> void {
        self.string(x.cast<std::string>());
      })
      .def("beginlist", &ak::FillableArray::beginlist)
      .def("endlist", &ak::FillableArray::endlist)
      .def("begintuple", &ak::FillableArray::begintuple)
      .def("index", &ak::FillableArray::index)
      .def("endtuple", &ak::FillableArray::endtuple)
      .def("beginrecord", [](ak::FillableArray& self, py::object name) -> void {
        if (name.is(py::none())) {
          self.beginrecord();
        }
        else {
          std::string cppname = name.cast<std::string>();
          self.beginrecord_check(cppname.c_str());
        }
      }, py::arg("name") = py::none())
      .def("field", [](ak::FillableArray& self, const std::string& x) -> void {
        self.field_check(x);
      })
      .def("endrecord", &ak::FillableArray::endrecord)
      .def("fill", &fillable_fill)
  );
}
