// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/identities.cpp", line)

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "awkward/python/util.h"

#include "awkward/python/identities.h"

template <typename T>
ak::IdentitiesOf<T>
Identities_from_cupy(const std::string& name,
                     ak::Identities::Ref ref,
                     const ak::Identities::FieldLoc& fieldloc,
                     const py::object& array) {
  if (py::isinstance(array, py::module::import("cupy").attr("ndarray"))) {
    void* ptr = reinterpret_cast<void *>(py::cast<ssize_t>
                                         (array.attr("data").attr("ptr")));

    if (py::cast<int64_t>(array.attr("ndim")) != 2) {
      throw std::invalid_argument(
        name + std::string(" must be built from a two-dimensional array")
        + FILENAME(__LINE__));
    }
    std::vector<int64_t> shape, strides;

    shape = array.attr("shape").cast<std::vector<int64_t>>();
    strides = array.attr("strides").cast<std::vector<int64_t>>();

    if (strides[0] != sizeof(T)*shape[1]  ||
        strides[1] != sizeof(T)) {
      throw std::invalid_argument(
        name + std::string(" must be built from a contiguous array (array"
                           ".stries == (array.shape[1]*array.itemsize, "
                           "array.itemsize)); try array.copy()")
        + FILENAME(__LINE__));
    }
    return ak::IdentitiesOf<T>(ref,
                               fieldloc,
                               0,
                               shape[1],
                               shape[0],
                               std::shared_ptr<T>(reinterpret_cast<T*>(ptr),
                                                  pyobject_deleter<T>(array.ptr())),
                               ak::kernel::lib::cuda);
  }
  else {
    throw std::invalid_argument(
      name + std::string(".from_cupy() can only accept CuPy Arrays!")
      + FILENAME(__LINE__));
  }
}

template <typename T>
py::class_<ak::IdentitiesOf<T>>
make_IdentitiesOf(const py::handle& m, const std::string& name) {
  return (py::class_<ak::IdentitiesOf<T>>(m,
                                          name.c_str(),
                                          py::buffer_protocol())
      .def_buffer([](const ak::IdentitiesOf<T>& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<void*>(
            reinterpret_cast<ssize_t>(
              self.ptr().get()) + self.offset()*sizeof(T)),
          sizeof(T),
          py::format_descriptor<T>::format(),
          2,
          { (ssize_t)self.length(), (ssize_t)self.width() },
          { (ssize_t)(sizeof(T)*self.width()), (ssize_t)sizeof(T) });
        })

      .def_static("newref", &ak::Identities::newref)

      .def(py::init([](ak::Identities::Ref ref,
                       const ak::Identities::FieldLoc& fieldloc,
                       int64_t width,
                       int64_t length) {
        return ak::IdentitiesOf<T>(ref, fieldloc, width, length);
      }))

      .def(py::init([name](ak::Identities::Ref ref,
                           ak::Identities::FieldLoc fieldloc,
                           const py::object& anyarray) -> ak::IdentitiesOf<T> {
        std::string module = anyarray.get_type().attr("__module__").cast<std::string>();
        if (module.rfind("cupy.", 0) == 0) {
          return Identities_from_cupy<T>(name, ref, fieldloc, anyarray);
        }

        py::array_t<T> array = anyarray.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();

        py::buffer_info info = array.request();
        if (info.ndim != 2) {
          throw std::invalid_argument(
            name + std::string(" must be built from a two-dimensional array")
            + FILENAME(__LINE__));
        }
        if (info.strides[0] != sizeof(T)*info.shape[1]  ||
            info.strides[1] != sizeof(T)) {
          throw std::invalid_argument(
            name + std::string(" must be built from a contiguous array (array"
                               ".stries == (array.shape[1]*array.itemsize, "
                               "array.itemsize)); try array.copy()")
            + FILENAME(__LINE__));
        }
        return ak::IdentitiesOf<T>(ref,
                                   fieldloc,
                                   0,
                                   info.shape[1],
                                   info.shape[0],
            std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr),
                               pyobject_deleter<T>(array.ptr())));
      }))

      .def_property_readonly("ptr_lib", [](const ak::IdentitiesOf<T>& self) {
        if (self.ptr_lib() == ak::kernel::lib::cpu) {
          return py::cast("cpu");
        }
        else if (self.ptr_lib() == ak::kernel::lib::cuda) {
          return py::cast("cuda");
        }
        else {
          throw std::runtime_error(
            std::string("unrecognized ptr_lib") + FILENAME(__LINE__));
        }
      })
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
      .def("identity_at",
           [](const ak::IdentitiesOf<T>& self, int64_t at) -> py::tuple {
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

      .def("copy_to",
           [name](const ak::IdentitiesOf<T>& self, const std::string& ptr_lib) -> py::object {
             if (ptr_lib == "cpu") {
               std::shared_ptr<ak::Identities> cpu_identities =
                   self.copy_to(ak::kernel::lib::cpu) ;
               return py::cast(*cpu_identities);
             }
             else if (ptr_lib == "cuda") {
               std::shared_ptr<ak::Identities> cuda_identities =
                   self.copy_to(ak::kernel::lib::cuda);
               return py::cast(*cuda_identities);
             }
             else {
               throw std::invalid_argument(
                 std::string("specify 'cpu' or 'cuda'")
                 + FILENAME(__LINE__));
             }
           })
      .def_static("from_cupy", [name](ak::Identities::Ref ref,
                                      const ak::Identities::FieldLoc& fieldloc,
                                      const py::object& array) -> ak::IdentitiesOf<T> {
        return Identities_from_cupy<T>(name, ref, fieldloc, array);
      })
      .def("to_cupy", [name](const ak::IdentitiesOf<T>& self) -> py::object {
        if (self.ptr_lib() != ak::kernel::lib::cuda) {
          throw std::invalid_argument(
            name
            + std::string(" resides in main memory, must be converted to NumPy, not CuPy")
            + FILENAME(__LINE__));
        }

        std::shared_ptr<ak::Identities> cuda_identities =
            self.copy_to(ak::kernel::lib::cuda);

        ak::IdentitiesOf<T>* cuda_identities_of =
            dynamic_cast<ak::IdentitiesOf<T>*>(cuda_identities.get());

        py::object cupy_unowned_mem =
            py::module::import("cupy").attr("cuda").attr("UnownedMemory")(
                reinterpret_cast<ssize_t>(cuda_identities_of->ptr().get()),
                cuda_identities_of->length() * sizeof(T),
                cuda_identities_of);

        py::object cupy_memoryptr =
            py::module::import("cupy").attr("cuda").attr("MemoryPointer")(
                cupy_unowned_mem,
                0);

        return py::module::import("cupy").attr("ndarray")(
                pybind11::make_tuple(py::cast<ssize_t>(cuda_identities->length())),
                py::format_descriptor<T>::format(),
                cupy_memoryptr,
                pybind11::make_tuple(py::cast<ssize_t>(sizeof(T))));
      })

  );
}

template py::class_<ak::Identities32>
make_IdentitiesOf(const py::handle& m, const std::string& name);

template py::class_<ak::Identities64>
make_IdentitiesOf(const py::handle& m, const std::string& name);
