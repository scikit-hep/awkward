// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/numpy.h>

#include "awkward/python/util.h"

#include "awkward/python/index.h"

template <typename T>
py::class_<ak::IndexOf<T>> make_IndexOf(const py::handle& m, const std::string& name) {
  return (py::class_<ak::IndexOf<T>>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](const ak::IndexOf<T>& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<void*>(reinterpret_cast<ssize_t>(self.ptr().get()) + self.offset()*sizeof(T)),
          sizeof(T),
          py::format_descriptor<T>::format(),
          1,
          { (ssize_t)self.length() },
          { (ssize_t)sizeof(T) });
        })

      .def(py::init([name](py::array_t<T, py::array::c_style | py::array::forcecast> array) -> ak::IndexOf<T> {
        py::buffer_info info = array.request();
        if (info.ndim != 1) {
          throw std::invalid_argument(name + std::string(" must be built from a one-dimensional array; try array.ravel()"));
        }
        if (info.strides[0] != sizeof(T)) {
          throw std::invalid_argument(name + std::string(" must be built from a contiguous array (array.strides == (array.itemsize,)); try array.copy()"));
        }
        return ak::IndexOf<T>(
          std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr), pyobject_deleter<T>(array.ptr())),
          0,
          (int64_t)info.shape[0]);
      }))

      .def("__repr__", &ak::IndexOf<T>::tostring)
      .def("__len__", &ak::IndexOf<T>::length)
      .def("__getitem__", &ak::IndexOf<T>::getitem_at)
      .def("__getitem__", &ak::IndexOf<T>::getitem_range)

  );
}

template py::class_<ak::Index8> make_IndexOf<int8_t>(const py::handle& m, const std::string& name);
template py::class_<ak::IndexU8> make_IndexOf<uint8_t>(const py::handle& m, const std::string& name);
template py::class_<ak::Index32> make_IndexOf<int32_t>(const py::handle& m, const std::string& name);
template py::class_<ak::IndexU32> make_IndexOf<uint32_t>(const py::handle& m, const std::string& name);
template py::class_<ak::Index64> make_IndexOf<int64_t>(const py::handle& m, const std::string& name);
