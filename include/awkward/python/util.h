// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_UTIL_H_
#define AWKWARDPY_UTIL_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "awkward/Content.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"

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

#endif // AWKWARDPY_UTIL_H_
