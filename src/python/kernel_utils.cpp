// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/python/kernel_utils.h"

py::enum_<kernel::Lib>
make_Libenum(const py::handle& m, const std::string& name) {
  return (py::enum_<kernel::Lib>(m, name.c_str())
    .value("cpu", kernel::Lib::cpu_kernels)
    .value("cuda", kernel::Lib::cuda_kernels)
    .export_values());
}
