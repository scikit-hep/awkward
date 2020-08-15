// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_STARTUP_H_
#define AWKWARDPY_STARTUP_H_

#include <pybind11/pybind11.h>

#include "awkward/kernel-dispatch.h"

namespace py = pybind11;
namespace ak = awkward;

class StartupLibraryPathCallback : public ak::kernel::LibraryPathCallback {
public:
  StartupLibraryPathCallback() = default;

  std::string library_path() override;
private:
    std::string library_path_;
};

void
make_startup(py::module& m, const std::string& name);

#endif // AWKWARDPY_STARTUP_H_
