// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <iostream>
#include <string>

#include "awkward/python/startup.h"

 namespace ak = awkward;

std::string StartupLibraryPathCallback::library_path() {
  // Fetches Path Eagerly
  std::string eager_path;

  try {
    py::object awkward1_cuda_kernels = py::module::import(
      "awkward1_cuda_kernels");

    if (py::hasattr(awkward1_cuda_kernels, "shared_library_path")) {
      py::object library_path_pyobj = py::getattr(awkward1_cuda_kernels,
                                                  "shared_library_path");
      eager_path = library_path_pyobj.cast<std::string>();
    }
  }
  catch (...) {
    // If the Python Extension fails to import the module, fall back to the
    // previous fetched path
    eager_path = library_path_;
  }

  // Keep the Library Patch as a cache, this will fetch paths even when the shared
  // library path is changed but in case of a failure it will revert to the last
  // successful library fetch. In case the shared Library ceases to exist, it will
  // be handled by the dlopen and dlsym in the C++ layer.
  library_path_ = eager_path;

  return library_path_;
}

void
make_startup(py::module& m, const std::string& name) {
  m.def(name.c_str(), []() -> void {
    ak::kernel::lib_callback->add_library_path_callback(
      ak::kernel::lib::cuda, std::make_shared<StartupLibraryPathCallback>());
  });
}
