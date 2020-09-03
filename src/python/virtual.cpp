// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/virtual.cpp", line)

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

const py::dict
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

const std::shared_ptr<ak::ArrayGenerator>
PyArrayGenerator::shallow_copy() const {
  return std::make_shared<PyArrayGenerator>(form_,
                                            length_,
                                            callable_,
                                            args_,
                                            kwargs_);
}

const std::shared_ptr<ak::ArrayGenerator>
PyArrayGenerator::with_form(const std::shared_ptr<ak::Form>& form) const {
  return std::make_shared<PyArrayGenerator>(form,
                                            length_,
                                            callable_,
                                            args_,
                                            kwargs_);
}

const std::shared_ptr<ak::ArrayGenerator>
PyArrayGenerator::with_length(int64_t length) const {
  return std::make_shared<PyArrayGenerator>(form_,
                                            length,
                                            callable_,
                                            args_,
                                            kwargs_);
}


const std::shared_ptr<ak::ArrayGenerator>
PyArrayGenerator::with_callable(const py::object& callable) const {
  return std::make_shared<PyArrayGenerator>(form_,
                                            length_,
                                            callable,
                                            args_,
                                            kwargs_);
}

const std::shared_ptr<ak::ArrayGenerator>
PyArrayGenerator::with_args(const py::tuple& args) const {
  return std::make_shared<PyArrayGenerator>(form_,
                                            length_,
                                            callable_,
                                            args,
                                            kwargs_);
}

const std::shared_ptr<ak::ArrayGenerator>
PyArrayGenerator::with_kwargs(const py::dict& kwargs) const {
  return std::make_shared<PyArrayGenerator>(form_,
                                            length_,
                                            callable_,
                                            args_,
                                            kwargs);
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
              std::string("ArrayGenerator 'form' must be an ak.forms.Form or None")
              + FILENAME(__LINE__));
          }
        }
        int64_t cpplength = -1;
        if (!length.is(py::none())) {
          try {
            cpplength = length.cast<int64_t>();
          }
          catch (py::cast_error err) {
            throw std::invalid_argument(
              std::string("ArrayGenerator 'length' must be an int or None")
              + FILENAME(__LINE__));
          }
        }
        return PyArrayGenerator(cppform, cpplength, callable, args, kwargs);
      }), py::arg("callable")
        , py::arg("args") = py::tuple(0)
        , py::arg("kwargs") = py::dict()
        , py::arg("form") = py::none()
        , py::arg("length") = py::none())
      .def_property_readonly("callable", &PyArrayGenerator::callable)
      .def_property_readonly("args", &PyArrayGenerator::args)
      .def_property_readonly("kwargs", &PyArrayGenerator::kwargs)
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
      .def("__call__", [](PyArrayGenerator& self) -> py::object {
        return box(self.generate_and_check());
      })
      .def("__repr__", [](const PyArrayGenerator& self) -> std::string {
        return self.tostring_part("", "", "");
      })
      .def("with_form", [](const PyArrayGenerator& self,
                           const std::shared_ptr<ak::Form>& form) -> py::object {
        std::shared_ptr<ak::ArrayGenerator> out = self.with_form(form);
        return py::cast(out);
      })
      .def("with_length", [](const PyArrayGenerator& self,
                             int64_t length) -> py::object {
        std::shared_ptr<ak::ArrayGenerator> out = self.with_length(length);
        return py::cast(out);
      })
      .def("with_callable", [](const PyArrayGenerator& self,
                               const py::object& callable) -> py::object {
        std::shared_ptr<ak::ArrayGenerator> out = self.with_callable(callable);
        return py::cast(out);
      })
      .def("with_args", [](const PyArrayGenerator& self,
                           const py::tuple& args) -> py::object {
        std::shared_ptr<ak::ArrayGenerator> out = self.with_args(args);
        return py::cast(out);
      })
      .def("with_kwargs", [](const PyArrayGenerator& self,
                             const py::dict& kwargs) -> py::object {
        std::shared_ptr<ak::ArrayGenerator> out = self.with_kwargs(kwargs);
        return py::cast(out);
      })
  );
}

////////// SliceGenerator

py::class_<ak::SliceGenerator, std::shared_ptr<ak::SliceGenerator>>
make_SliceGenerator(const py::handle& m, const std::string& name) {
  return (py::class_<ak::SliceGenerator,
                     std::shared_ptr<ak::SliceGenerator>>(m, name.c_str())
      .def(py::init([](const py::object& content,
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
              std::string("SliceGenerator 'form' must be an ak.forms.Form or None")
              + FILENAME(__LINE__));
          }
        }
        int64_t cpplength = -1;
        if (!length.is(py::none())) {
          try {
            cpplength = length.cast<int64_t>();
          }
          catch (py::cast_error err) {
            throw std::invalid_argument(
              std::string("SliceGenerator 'length' must be an int or None")
              + FILENAME(__LINE__));
          }
        }
        ak::Slice cppslice = toslice(slice);
        return ak::SliceGenerator(cppform,
                                  cpplength,
                                  unbox_content(content),
                                  cppslice);
      }), py::arg("content")
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
      .def_property_readonly("content", &ak::SliceGenerator::content)
      .def("__call__", [](ak::SliceGenerator& self) -> py::object {
        return box(self.generate_and_check());
      })
      .def("__repr__", [](const ak::SliceGenerator& self) -> std::string {
        return self.tostring_part("", "", "");
      })
      .def("with_form", [](const ak::SliceGenerator& self,
                           const std::shared_ptr<ak::Form>& form) -> py::object {
        std::shared_ptr<ak::ArrayGenerator> out = self.with_form(form);
        return py::cast(out);
      })
      .def("with_length", [](const ak::SliceGenerator& self,
                             int64_t length) -> py::object {
        std::shared_ptr<ak::ArrayGenerator> out = self.with_length(length);
        return py::cast(out);
      })
  );
}

////////// PyArrayCache

PyArrayCache::PyArrayCache(const py::object& mutablemapping)
    : mutablemapping_(mutablemapping.is(py::none())
                          ? mutablemapping
                          : py::weakref((const py::handle&) mutablemapping)) {}

const py::object
PyArrayCache::mutablemapping() const {
  if ( mutablemapping_.is(py::none()) ) {
    return mutablemapping_;
  }
  const py::object out = mutablemapping_();
  if ( out.is(py::none()) ) {
    throw std::runtime_error(
      std::string("PyArrayCache has lost its weak reference to mapping")
      + FILENAME(__LINE__));
  }
  return out;
}

ak::ContentPtr
PyArrayCache::get(const std::string& key) const {
  py::str pykey(PyUnicode_DecodeUTF8(key.data(),
                                     key.length(),
                                     "surrogateescape"));
  py::object out;
  try {
    out = mutablemapping().attr("__getitem__")(pykey);
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
  const py::object mapping = mutablemapping();
  if ( ! mapping.is(py::none()) ) {
    mapping.attr("__setitem__")(pykey, box(value));
  }
}

const std::string
PyArrayCache::tostring_part(const std::string& indent,
                            const std::string& pre,
                            const std::string& post) const {
  std::string repr =
    mutablemapping().attr("__repr__")().cast<std::string>();
  if (repr.length() > 50) {
    repr = repr.substr(0, 47) + std::string("...");
  }
  std::stringstream out;
  out << indent << pre << "<ArrayCache mapping=\""
      << repr << "\"/>"
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
                             const std::string& key) -> void {
        self.mutablemapping().attr("__delitem__")(py::cast(key));
      })
      .def("__iter__", [](const PyArrayCache& self) -> py::object {
        return self.mutablemapping().attr("__iter__")();
      })
      .def("__len__", [](const PyArrayCache& self) -> py::object {
        return self.mutablemapping().attr("__len__")();
      })

  );
}
