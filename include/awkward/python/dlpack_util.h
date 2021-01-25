// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARDPY_DLPACK_UTIL_H_
#define AWKWARDPY_DLPACK_UTIL_H_

#include "awkward/util.h"

#include "dlpack/dlpack.h"

namespace py = pybind11;
namespace ak = awkward;

namespace awkward {
  namespace dlpack {
    DLDataType
    data_type_dispatch(ak::util::dtype dt);

    DLContext
    device_context_dispatch(ak::kernel::lib ptr_lib, void* ptr);

    void
    dlpack_deleter(DLManagedTensor* tensor);

    void
    pycapsule_deleter(PyObject* dltensor);
  }
}

#endif
