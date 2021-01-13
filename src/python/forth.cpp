// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/forth.cpp", line)

#include <pybind11/pybind11.h>

#include "awkward/python/forth.h"

template <typename T, typename I>
py::class_<ak::ForthMachineOf<T, I>, std::shared_ptr<ak::ForthMachineOf<T, I>>>
make_ForthMachineOf(const py::handle& m, const std::string& name) {
  return (py::class_<ak::ForthMachineOf<T, I>,
          std::shared_ptr<ak::ForthMachineOf<T, I>>>(m, name.c_str())
          .def(py::init([](const std::string& source,
                           int64_t stack_size,
                           int64_t recursion_depth,
                           int64_t output_initial_size,
                           double output_resize_factor) -> ak::ForthMachineOf<T, I> {
            return ak::ForthMachineOf<T, I>(source,
                                            stack_size,
                                            recursion_depth,
                                            output_initial_size,
                                            output_resize_factor);
          }),
               py::arg("source"),
               py::arg("stack_size") = 1024,
               py::arg("recursion_depth") = 1024,
               py::arg("output_initial_size") = 1024,
               py::arg("output_resize_factor") = 1.5)

         );
}

template py::class_<ak::ForthMachine32, std::shared_ptr<ak::ForthMachine32>>
make_ForthMachineOf(const py::handle& m, const std::string& name);

template py::class_<ak::ForthMachine64, std::shared_ptr<ak::ForthMachine64>>
make_ForthMachineOf(const py::handle& m, const std::string& name);
