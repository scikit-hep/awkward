// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARDPY_UTIL_H_
#define AWKWARDPY_UTIL_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

/// @class pyobject_deleter
///
/// @brief Used as a `std::shared_ptr` deleter (second argument) to
/// overload `delete ptr` with `Py_DECREF(ptr)`.
///
/// This allows data to be shared between C++ and Python that will be
/// deleted when _both_ reference counts reach zero.
///
/// The effective reference count is a lexicographical tuple of
/// (Python reference count, C++ reference count), where the Python
/// reference count is the most significant digit. A single Python
/// reference count is held for all C++ references to the object.
///
/// See also
///   - array_deleter, which frees array buffers, rather
///     than objects.
template<typename T>
class pyobject_deleter {
public:
  /// @brief Creates a pyobject_deleter and calls `Py_INCREF(ptr)`.
  pyobject_deleter(PyObject *pyobj): pyobj_(pyobj) {
    Py_INCREF(pyobj_);
  }
  /// @brief Called by `std::shared_ptr` when its reference count reaches
  /// zero.
  void operator()(T const * /* p */) {
    Py_DECREF(pyobj_);
  }
private:
  /// @brief The Python object that we hold a reference to.
  PyObject* pyobj_;
};

#endif // AWKWARDPY_UTIL_H_
