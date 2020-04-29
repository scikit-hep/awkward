// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include <pybind11/pybind11.h>

#include "awkward/python/content.h"

#include "awkward/python/virtual.h"

////////// PyArrayGenerator

PyArrayGenerator::PyArrayGenerator(const ak::FormPtr& form,
                                   int64_t length,
                                   const py::object& callable,
                                   const py::tuple& args,
                                   const py::dict& kwargs)
    : ArrayGenerator(form, length)
    , callable_(callable)
    , args_(args)
    , kwargs_(kwargs) { }

const py::object
PyArrayGenerator::callable() const {
  return callable_;
}

const py::tuple
PyArrayGenerator::args() const {
  return args_;
}

const py::kwargs
PyArrayGenerator::kwargs() const {
  return kwargs_;
}

const ak::ContentPtr
PyArrayGenerator::generate() const {
  py::object out = callable_(*args_, **kwargs_);
  py::object layout = py::module::import("awkward1").attr("to_layout")(
                                        out, py::cast(false), py::cast(false));
  return unbox_content(layout);
}

const std::string
PyArrayGenerator::tostring_part(const std::string& indent,
                                const std::string& pre,
                                const std::string& post) const {
  std::stringstream out;
  out << indent << pre << "<ArrayGenerator f=\"";
  out << callable_.attr("__repr__")().cast<std::string>() << "\"";
  if (!args_.empty()) {
    out << " args=\"" << args_.attr("__repr__")().cast<std::string>() << "\"";
  }
  if (!kwargs_.empty()) {
    out << " kwargs=\"" << kwargs_.attr("__repr__")().cast<std::string>()
        << "\"";
  }
  if (form_.get() == nullptr  &&  length_ < 0) {
    out << "/>";
  }
  else {
    out << ">\n";
    if (length_ >= 0) {
      out << indent << "    <length>" << length_ << "</length>\n";
    }
    if (form_.get() != nullptr) {
      std::string formstr = form_.get()->tojson(true, false);
      std::string replace = std::string("\n")
                                + indent + std::string("        ");
      size_t pos = 0;
      while ((pos = formstr.find("\n", pos)) != std::string::npos) {
        formstr.replace(pos, 1, replace);
        pos += replace.length();
      }
      out << indent << "    <form>\n" << indent << "        " << formstr
          << "\n" << indent << "    </form>\n";
    }
    out << indent << "</ArrayGenerator>";
  }
  out << post;
  return out.str();
}

py::class_<PyArrayGenerator, std::shared_ptr<PyArrayGenerator>>
make_PyArrayGenerator(const py::handle& m, const std::string& name) {
  return (py::class_<PyArrayGenerator,
                     std::shared_ptr<PyArrayGenerator>>(m, name.c_str())
      .def(py::init([](const py::object& callable,
                       const py::tuple& args,
                       const py::dict& kwargs,
                       const py::object& form,
                       const py::object& length) -> PyArrayGenerator {
        ak::FormPtr cppform(nullptr);
        if (!form.is(py::none())) {
          try {
            cppform = form.cast<ak::Form*>()->shallow_copy();
          }
          catch (py::cast_error err) {
            throw std::invalid_argument(
                "ArrayGenerator 'form' must be an ak.forms.Form or None");
          }
        }
        int64_t cpplength = -1;
        if (!length.is(py::none())) {
          try {
            cpplength = length.cast<int64_t>();
          }
          catch (py::cast_error err) {
            throw std::invalid_argument(
                "ArrayGenerator 'length' must be an int or None");
          }
        }
        return PyArrayGenerator(cppform, cpplength, callable, args, kwargs);
      }), py::arg("callable")
        , py::arg("args") = py::tuple(0)
        , py::arg("kwargs") = py::dict()
        , py::arg("form") = py::none()
        , py::arg("length") = py::none())
      .def_property_readonly("form", [](const PyArrayGenerator& self)
                                     -> py::object {
        ak::FormPtr form = self.form();
        if (form.get() == nullptr) {
          return py::none();
        }
        else {
          return py::cast(form.get());
        }
      })
      .def_property_readonly("length", [](const PyArrayGenerator& self)
                                       -> py::object {
        int64_t length = self.length();
        if (length < 0) {
          return py::none();
        }
        else {
          return py::cast(length);
        }
      })
      .def("__call__", [](const PyArrayGenerator& self) -> py::object {
        return box(self.generate_and_check());
      })
      .def("__repr__", [](const PyArrayGenerator& self) -> std::string {
        return self.tostring_part("", "", "");
      })
  );
}

////////// SliceGenerator

py::class_<ak::SliceGenerator, std::shared_ptr<ak::SliceGenerator>>
make_SliceGenerator(const py::handle& m, const std::string& name) {
  return (py::class_<ak::SliceGenerator,
                     std::shared_ptr<ak::SliceGenerator>>(m, name.c_str())
      .def(py::init([](const py::object& generator,
                       const py::object& slice,
                       const py::object& form,
                       const py::object& length) -> ak::SliceGenerator {
        ak::FormPtr cppform(nullptr);
        if (!form.is(py::none())) {
          try {
            cppform = form.cast<ak::Form*>()->shallow_copy();
          }
          catch (py::cast_error err) {
            throw std::invalid_argument(
                "SliceGenerator 'form' must be an ak.forms.Form or None");
          }
        }
        int64_t cpplength = -1;
        if (!length.is(py::none())) {
          try {
            cpplength = length.cast<int64_t>();
          }
          catch (py::cast_error err) {
            throw std::invalid_argument(
                "SliceGenerator 'length' must be an int or None");
          }
        }
        std::shared_ptr<ak::ArrayGenerator> cppgenerator(nullptr);
        try {
          cppgenerator = generator.cast<std::shared_ptr<PyArrayGenerator>>();
        }
        catch (py::cast_error err) {
          try {
            cppgenerator = generator.cast<std::shared_ptr<ak::SliceGenerator>>();
          }
          catch (py::cast_error err) {
            throw std::invalid_argument(
              "generator must be an ArrayGenerator or another SliceGenerator");
          }
        }
        ak::Slice cppslice = toslice(slice);
        return ak::SliceGenerator(cppform, cpplength, cppgenerator, cppslice);
      }), py::arg("generator")
        , py::arg("slice")
        , py::arg("form") = py::none()
        , py::arg("length") = py::none())
      .def_property_readonly("form",
                             [](const ak::SliceGenerator& self) -> py::object {
        ak::FormPtr form = self.form();
        if (form.get() == nullptr) {
          return py::none();
        }
        else {
          return py::cast(form.get());
        }
      })
      .def_property_readonly("length",
                             [](const ak::SliceGenerator& self) -> py::object {
        int64_t length = self.length();
        if (length < 0) {
          return py::none();
        }
        else {
          return py::cast(length);
        }
      })
      .def_property_readonly("generator",
                             [](const ak::SliceGenerator& self) -> py::object {
        std::shared_ptr<ak::ArrayGenerator> out = self.generator();
        if (std::shared_ptr<ak::SliceGenerator> ptr =
            std::dynamic_pointer_cast<ak::SliceGenerator>(out)) {
          return py::cast(ptr);
        }
        else {
          return py::cast(out);
        }
      })
      .def("__call__", [](const ak::SliceGenerator& self) -> py::object {
        return box(self.generate_and_check());
      })
      .def("__repr__", [](const ak::SliceGenerator& self) -> std::string {
        return self.tostring_part("", "", "");
      })
  );
}

////////// PyArrayCache

PyArrayCache::PyArrayCache(const py::object& mutablemapping)
    : mutablemapping_(mutablemapping) { }

const py::object
PyArrayCache::mutablemapping() const {
  return mutablemapping_;
}

ak::ContentPtr
PyArrayCache::get(const std::string& key) const {
  py::str pykey(PyUnicode_DecodeUTF8(key.data(),
                                     key.length(),
                                     "surrogateescape"));
  py::object out;
  try {
    out = mutablemapping_.attr("__getitem__")(pykey);
  }
  catch (py::error_already_set err) {
    return ak::ContentPtr(nullptr);
  }
  return unbox_content(out);
}

void
PyArrayCache::set(const std::string& key, const ak::ContentPtr& value) {
  py::str pykey(PyUnicode_DecodeUTF8(key.data(),
                                     key.length(),
                                     "surrogateescape"));
  mutablemapping_.attr("__setitem__")(pykey, box(value));
}

const std::string
PyArrayCache::tostring_part(const std::string& indent,
                            const std::string& pre,
                            const std::string& post) const {
  std::string mutablemapping =
    mutablemapping_.attr("__repr__")().cast<std::string>();
  if (mutablemapping.length() > 50) {
    mutablemapping = mutablemapping.substr(0, 47) + std::string("...");
  }
  std::stringstream out;
  out << indent << pre << "<ArrayCache mapping=\""
      << mutablemapping << "\"/>"
      << post;
  return out.str();
}

py::class_<PyArrayCache, std::shared_ptr<PyArrayCache>>
make_PyArrayCache(const py::handle& m, const std::string& name) {
  return (py::class_<PyArrayCache,
                     std::shared_ptr<PyArrayCache>>(m, name.c_str())
      .def(py::init<const py::object&>(),
           py::arg("mutablemapping"))
      .def_property_readonly("mutablemapping", &PyArrayCache::mutablemapping)
      .def("__repr__", [](const PyArrayCache& self) -> std::string {
        return self.tostring_part("", "", "");
      })
      .def("__getitem__", [](const PyArrayCache& self,
                             const std::string& key) -> py::object {
        return box(self.get(key));
      })
      .def("__setitem__", [](PyArrayCache& self,
                             const std::string& key,
                             const py::object& value) -> void {
        self.set(key, unbox_content(value));
      })
      .def("__delitem__", [](PyArrayCache& self,
                             const std::string& key) -> py::object {
        return self.mutablemapping().attr("__delitem__")(py::cast(key));
      })
      .def("__iter__", [](const PyArrayCache& self) -> py::object {
        return self.mutablemapping().attr("__iter__")();
      })
      .def("__len__", [](const PyArrayCache& self) -> py::object {
        return self.mutablemapping().attr("__len__")();
      })

  );
}
