// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/pybind11.h>

#include "awkward/python/content.h"

#include "awkward/python/virtual.h"

PyArrayGenerator::PyArrayGenerator(const ak::FormPtr& form,
                                   const py::object& callable,
                                   const py::tuple& args,
                                   const py::dict& kwargs)
    : ArrayGenerator(form)
    , callable_(callable)
    , args_(args)
    , kwargs_(kwargs) { }

const ak::ContentPtr
PyArrayGenerator::generate() const {
  return unbox_content(callable_(*args_, **kwargs_));
}

py::class_<PyArrayGenerator, std::shared_ptr<PyArrayGenerator>>
make_PyArrayGenerator(const py::handle& m, const std::string& name) {
  return (py::class_<PyArrayGenerator,
                     std::shared_ptr<PyArrayGenerator>>(m, name.c_str())
      .def(py::init([](const py::object& callable,
                       const py::tuple& args,
                       const py::dict& kwargs,
                       const py::object& form) -> PyArrayGenerator {
        ak::FormPtr cppform(nullptr);
        if (!form.is(py::none())) {
          try {
            cppform = form.cast<ak::Form*>()->shallow_copy();
          }
          catch (py::cast_error err) {
            throw std::invalid_argument(
                "PyArrayGenerator 'form' must be an ak.forms.Form or None");
          }
        }
        return PyArrayGenerator(cppform, callable, args, kwargs);
      }), py::arg("callable")
        , py::arg("args") = py::tuple(0)
        , py::arg("kwargs") = py::dict()
        , py::arg("form") = py::none())
      .def("form", [](const PyArrayGenerator& self) -> py::object {
        ak::FormPtr form = self.form();
        if (form.get() == nullptr) {
          return py::none();
        }
        else {
          py::cast(form.get());
        }
      })
      .def("__call__", [](const PyArrayGenerator& self) -> py::object {
        return box(self.generate_and_check());
      })
  );
}

PyArrayCache::PyArrayCache(const py::object& mutablemapping)
    : mutablemapping_(mutablemapping) { }

ak::ContentPtr
PyArrayCache::get(const std::string& key) const {
  py::str pykey(PyUnicode_DecodeUTF8(key.data(),
                                     key.length(),
                                     "surrogateescape"));
  try {
    return unbox_content(mutablemapping_.attr("__getitem__")(pykey));
  }
  catch (py::key_error err) {
    return ak::ContentPtr(nullptr);
  }
}

void
PyArrayCache::set(const std::string& key, const ak::ContentPtr& value) {
  py::str pykey(PyUnicode_DecodeUTF8(key.data(),
                                     key.length(),
                                     "surrogateescape"));
  mutablemapping_.attr("__setitem__")(pykey, box(value));
}

py::class_<PyArrayCache, std::shared_ptr<PyArrayCache>>
make_PyArrayCache(const py::handle& m, const std::string& name) {
  return (py::class_<PyArrayCache,
                     std::shared_ptr<PyArrayCache>>(m, name.c_str())
      .def(py::init<const py::object&>(),
           py::arg("mutablemapping"))
  );
}
