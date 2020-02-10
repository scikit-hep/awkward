// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_UTIL_H_
#define AWKWARDPY_UTIL_H_

#include <pybind11/pybind11.h>

#include "awkward/util.h"

namespace py = pybind11;
namespace ak = awkward;

template<typename T>
class pyobject_deleter {
public:
  pyobject_deleter(PyObject *pyobj): pyobj_(pyobj) {
    Py_INCREF(pyobj_);
  }
  void operator()(T const *p) {
    Py_DECREF(pyobj_);
  }
private:
  PyObject* pyobj_;
};

template <typename T>
py::object getitem(const T& self, const py::object& obj);

ak::util::Parameters dict2parameters(const py::object& in);
py::dict parameters2dict(const ak::util::Parameters& in);

#endif // AWKWARDPY_UTIL_H_
