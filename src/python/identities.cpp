// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "awkward/python/util.h"

#include "awkward/python/identities.h"

template <typename T>
py::class_<ak::IdentitiesOf<T>> make_IdentitiesOf(const py::handle& m, const std::string& name) {
  return (py::class_<ak::IdentitiesOf<T>>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](const ak::IdentitiesOf<T>& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<void*>(reinterpret_cast<ssize_t>(self.ptr().get()) + self.offset()*sizeof(T)),
          sizeof(T),
          py::format_descriptor<T>::format(),
          2,
          { (ssize_t)self.length(), (ssize_t)self.width() },
          { (ssize_t)(sizeof(T)*self.width()), (ssize_t)sizeof(T) });
        })

      .def_static("newref", &ak::Identities::newref)

      .def(py::init([](ak::Identities::Ref ref, const ak::Identities::FieldLoc& fieldloc, int64_t width, int64_t length) {
        return ak::IdentitiesOf<T>(ref, fieldloc, width, length);
      }))

      .def(py::init([name](ak::Identities::Ref ref, ak::Identities::FieldLoc fieldloc, py::array_t<T, py::array::c_style | py::array::forcecast> array) {
        py::buffer_info info = array.request();
        if (info.ndim != 2) {
          throw std::invalid_argument(name + std::string(" must be built from a two-dimensional array"));
        }
        if (info.strides[0] != sizeof(T)*info.shape[1]  ||  info.strides[1] != sizeof(T)) {
          throw std::invalid_argument(name + std::string(" must be built from a contiguous array (array.stries == (array.shape[1]*array.itemsize, array.itemsize)); try array.copy()"));
        }
        return ak::IdentitiesOf<T>(ref, fieldloc, 0, info.shape[1], info.shape[0],
            std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr), pyobject_deleter<T>(array.ptr())));
      }))

      .def("__repr__", &ak::IdentitiesOf<T>::tostring)
      .def("__len__", &ak::IdentitiesOf<T>::length)
      .def("__getitem__", &ak::IdentitiesOf<T>::getitem_at)
      .def("__getitem__", &ak::IdentitiesOf<T>::getitem_range)

      .def_property_readonly("ref", &ak::IdentitiesOf<T>::ref)
      .def_property_readonly("fieldloc", &ak::IdentitiesOf<T>::fieldloc)
      .def_property_readonly("width", &ak::IdentitiesOf<T>::width)
      .def_property_readonly("length", &ak::IdentitiesOf<T>::length)
      .def_property_readonly("array", [](const py::buffer& self) -> py::array {
        return py::array(self);
      })
      .def("identity_at_str", &ak::IdentitiesOf<T>::identity_at)
      .def("identity_at", [](const ak::Identities& self, int64_t at) -> py::tuple {
        ak::Identities::FieldLoc fieldloc = self.fieldloc();
        py::tuple out((size_t)self.width() + fieldloc.size());
        size_t j = 0;
        for (int64_t i = 0;  i < self.width();  i++) {
          out[j] = py::cast(self.value(at, i));
          j++;
          for (auto pair : fieldloc) {
            if (pair.first == i) {
              out[j] = py::cast(pair.second);
              j++;
            }
          }
        }
        return out;
      })

  );
}

template py::class_<ak::Identities32> make_IdentitiesOf(const py::handle& m, const std::string& name);
template py::class_<ak::Identities64> make_IdentitiesOf(const py::handle& m, const std::string& name);
