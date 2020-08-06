// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/python/kernel_utils.h"

namespace ak = awkward;

py::enum_<ak::kernel::lib>
make_lib_enum(const py::handle& m, const std::string& name) {
  return (py::enum_<ak::kernel::lib>(m, name.c_str())
    .value("cpu", ak::kernel::lib::cpu)
    .value("cuda", ak::kernel::lib::cuda)
    .export_values());
}
