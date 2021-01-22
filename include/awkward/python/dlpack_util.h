// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARDPY_DLPACK_UTIL_H_
#define AWKWARDPY_DLPACK_UTIL_H_

#include "dlpack/dlpack.h"
#include "awkward/util.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/python/util.h"
#include "awkward/kernel-dispatch.h"

namespace awkward {
  namespace dlpack {
    inline DLDataType data_type_dispatch(ak::util::dtype dt) {
      switch (dt) {
      // No support for boolean types in dlpack?
      case ak::util::dtype::int8:
        return {kDLInt, 8, 1};
      case ak::util::dtype::int16:
        return {kDLInt, 16, 1};
      case ak::util::dtype::int32:
        return {kDLInt, 32, 1};
      case ak::util::dtype::int64:
        return {kDLInt, 64, 1};
      case ak::util::dtype::uint8:
        return {kDLUInt, 8, 1};
      case ak::util::dtype::uint16:
        return {kDLUInt, 16, 1};
      case ak::util::dtype::uint32:
        return {kDLUInt, 32, 1};
      case ak::util::dtype::uint64:
        return {kDLUInt, 64, 1};
      case ak::util::dtype::float16:
        return {kDLFloat, 16, 1};
      case ak::util::dtype::float32:
        return {kDLFloat, 32, 1};
      case ak::util::dtype::float64:
        return {kDLFloat, 64, 1};
      case ak::util::dtype::float128:
        return {kDLFloat, 128, 1};
      // case ak::util::dtype::datetime64:
      //   return 8;
      // case ak::util::dtype::timedelta64:
      //   return 8;
      }
    }

    inline DLContext device_context_dispatch(ak::kernel::lib ptr_lib, void* ptr) {
      if (ptr_lib == ak::kernel::lib::cpu) {
        return DLContext{DLDeviceType::kDLCPU, 0};
      }
      else if (ptr_lib == ak::kernel::lib::cuda) {
        return DLContext{DLDeviceType::kDLGPU, ak::kernel::lib_device_num(ak::kernel::lib::cuda,
                                                                          ptr)};
      }
    }

    inline void deleter(DLManagedTensor* tensor) {
      if(tensor->manager_ctx == nullptr) 
        return;
      Py_DECREF(reinterpret_cast<PyObject*>(tensor->manager_ctx));
      tensor->manager_ctx = nullptr;
    }

    inline void pycapsule_deleter(PyObject* dltensor) {
      DLManagedTensor* dlm_tensor;
      if(PyCapsule_IsValid(dltensor, "dltensor")) {
        dlm_tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(
              dltensor, "dltensor"));
        dlm_tensor->deleter(dlm_tensor);
      }
    }
  }
}

#endif
