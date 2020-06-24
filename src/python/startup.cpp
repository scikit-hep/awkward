// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <iostream>
#include <string>

#include "awkward/python/startup.h"

// namespace ak = awkward;

void
make_startup(py::module& m, const std::string& name) {
  m.def(name.c_str(), []() -> void {

      // instead of the following, create a kernel::LibraryPathCallback
      // that runs this when the callback is called
      if (false) {
        try {
          py::object awkward1_cuda_kernels = py::module::import("awkward1_cuda_kernels");

          if (py::hasattr(awkward1_cuda_kernels, "shared_library_path")) {
            py::object library_path_pyobj = py::getattr(awkward1_cuda_kernels, "shared_library_path");

            std::string library_path = library_path_pyobj.cast<std::string>();

            std::cout << "awkward-cuda-kernels library path: " << library_path << std::endl;
          }
        }
        catch (...) {
          // do nothing
        }
      }

  });
}
