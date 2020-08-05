// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <pybind11/numpy.h>

#include "awkward/python/util.h"

#include "awkward/python/index.h"


template <typename T>
py::class_<ak::IndexOf<T>>
make_IndexOf(const py::handle& m, const std::string& name) {
  return (py::class_<ak::IndexOf<T>>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](const ak::IndexOf<T>& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<void*>(reinterpret_cast<ssize_t>(self.ptr().get())
                                  + self.offset()*sizeof(T)),
          sizeof(T),
          py::format_descriptor<T>::format(),
          1,
          { (ssize_t)self.length() },
          { (ssize_t)sizeof(T) });
        })

      .def(py::init([name](py::array_t<T,
                           py::array::c_style |
                           py::array::forcecast> array) -> ak::IndexOf<T> {
        py::buffer_info info = array.request();
        if (info.ndim != 1) {
          throw std::invalid_argument(name + std::string(
            " must be built from a one-dimensional array; try array.ravel()"));
        }
        if (info.strides[0] != sizeof(T)) {
          throw std::invalid_argument(name + std::string(
            " must be built from a contiguous array (array.strides == "
            "(array.itemsize,)); try array.copy()"));
        }
        return ak::IndexOf<T>(
          std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr),
                             pyobject_deleter<T>(array.ptr())),
          0,
          (int64_t)info.shape[0]);
      }))
      .def("__repr__", &ak::IndexOf<T>::tostring)
      .def("__len__", &ak::IndexOf<T>::length)
      .def("__getitem__", [](const ak::IndexOf<T>& self, py::object& obj) {
        if (py::isinstance<py::int_>(obj)) {
          T out = self.getitem_at(obj.cast<int64_t>());
          return py::cast(out);
        }
        else if (py::isinstance<py::slice>(obj)) {
          py::object pystep = obj.attr("step");
          if ((py::isinstance<py::int_>(pystep)  &&  pystep.cast<int64_t>() == 1)  ||
              pystep.is(py::none())) {
            int64_t start = ak::Slice::none();
            int64_t stop = ak::Slice::none();
            py::object pystart = obj.attr("start");
            py::object pystop = obj.attr("stop");
            if (!pystart.is(py::none())) {
              start = pystart.cast<int64_t>();
            }
            if (!pystop.is(py::none())) {
              stop = pystop.cast<int64_t>();
            }
            ak::IndexOf<T> out = self.getitem_range(start, stop);
            return py::cast(out);
          }
          else {
            throw std::invalid_argument("Index slices cannot contain step != 1");
          }
        }
        else {
          throw std::invalid_argument(
            "Index can only be sliced by an integer or start:stop slice");
        }
      })
      .def_static("from_cupy", [name](py::object array) -> ak::IndexOf<T> {
          if(py::isinstance(array, py::module::import("cupy").attr("ndarray"))) {
            if(!py::dtype(array.attr("dtype")).equal(py::dtype::of<T>())) {
                throw std::invalid_argument(name + std::string(
                  " arg0: must be a ") +
                  py::cast<std::string>(py::str(py::dtype::of<T>())) +
                  std::string(" array"));
            }

            if (py::cast<int64_t>(array.attr("ndim")) != 1) {
              throw std::invalid_argument(name + std::string(
                " must be built from a one-dimensional array; try array.ravel()"));
            }

            auto strides = pytuples_to_vector<int64_t>(array.attr("strides"));

            if (strides[0] != sizeof(T)) {
              throw std::invalid_argument(name + std::string(
                " must be built from a contiguous array (array.strides == "
                "(array.itemsize,)); try array.copy()"));
            }

            void* ptr = reinterpret_cast<void*>(py::cast<ssize_t>(array.attr("data").attr("ptr")));

            std::vector<int64_t> shape = pytuples_to_vector<int64_t>(array.attr("shape"));

            return ak::IndexOf<T>(std::shared_ptr<T>(reinterpret_cast<T*>(ptr),
                                                           pyobject_deleter<T>(array.ptr())),
                                  0,
                                  (int64_t)shape[0],
                                  kernel::Lib::cuda_kernels);
          }
          else {
            throw std::invalid_argument(name + std::string(
              ".from_cupy() can only accept CuPy Arrays!"));
          }
      })
      .def("copy_to", [name](const ak::IndexOf<T>& self, std::string& ptr_lib) {
        if(ptr_lib == "cuda") {
          auto cuda_index = self.copy_to(kernel::Lib::cuda_kernels);

          auto cupy_unowned_mem = py::module::import("cupy").attr("cuda").attr("UnownedMemory")(
            reinterpret_cast<ssize_t>(cuda_index.ptr().get()),
            cuda_index.length() * sizeof(T),
            cuda_index);
          
          auto cupy_memoryptr = py::module::import("cupy").attr("cuda").attr("MemoryPointer")(
            cupy_unowned_mem,
            0);
          auto cuda_array = py::module::import("cupy").attr("ndarray")(
            pybind11::make_tuple(py::cast<ssize_t>(cuda_index.length())),
            py::format_descriptor<T>::format(),
            cupy_memoryptr,
          pybind11::make_tuple(py::cast<ssize_t>(sizeof(T))));

          return py::module::import("awkward1").attr("layout").attr(name.c_str()).attr("from_cupy")
            (py::module::import("cupy").attr("ndarray")(
                pybind11::make_tuple(py::cast<ssize_t>(cuda_index.length())),
                py::format_descriptor<T>::format(),
                cupy_memoryptr,
                pybind11::make_tuple(py::cast<ssize_t>(sizeof(T)))));
        }
        else if(ptr_lib == "cpu") {
          ak::IndexOf<T> cuda_arr = self.copy_to(kernel::Lib::cpu_kernels) ;
          return py::cast<ak::IndexOf<T>>(cuda_arr);
        }
        else {
          throw std::invalid_argument("Invalid kernel specified, valid kernels are cpu and cuda");
        }
      })
  );
}

template py::class_<ak::Index8>
make_IndexOf<int8_t>(const py::handle& m, const std::string& name);

template py::class_<ak::IndexU8>
make_IndexOf<uint8_t>(const py::handle& m, const std::string& name);

template py::class_<ak::Index32>
make_IndexOf<int32_t>(const py::handle& m, const std::string& name);

template py::class_<ak::IndexU32>
make_IndexOf<uint32_t>(const py::handle& m, const std::string& name);

template py::class_<ak::Index64>
make_IndexOf<int64_t>(const py::handle& m, const std::string& name);
