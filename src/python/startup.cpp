// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <iostream>
#include <string>

#include "awkward/python/startup.h"

// namespace ak = awkward;

const std::string StartupLibraryPathCallback::library_path() const {
  std::string library_path = ("/");
  try {
    py::object awkward1_cuda_kernels = py::module::import("awkward1_cuda_kernels");

    if (py::hasattr(awkward1_cuda_kernels, "shared_library_path")) {
      py::object library_path_pyobj = py::getattr(awkward1_cuda_kernels, "shared_library_path");

      library_path = library_path_pyobj.cast<std::string>();

      std::cout << "awkward-cuda-kernels library path: " << library_path << std::endl;
    }
  }
  catch (...) {
    // do nothing
  }
  return library_path;
}

void
make_startup(py::module& m, const std::string& name) {
  m.def(name.c_str(), []() -> void {
    kernel::lib_callback->add_library_path_callback(kernel::Lib::cuda_kernels,
                                                    std::make_shared<StartupLibraryPathCallback>());
    try {
    py::object awkward1_cuda_kernels = py::module::import("awkward1_cuda_kernels");
    }
    catch (...) {
      // do nothing
    }
  });
}
