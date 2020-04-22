// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/pybind11.h>

#include "awkward/python/content.h"

#include "awkward/python/virtual.h"

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
      .def(py::init<const py::object&>(), py::arg("mutablemapping"))
  );
}
