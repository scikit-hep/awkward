# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._backends.backend import Backend, KernelKeyType
from awkward._backends.dispatch import register_backend
from awkward._kernels import CudaComputeKernel, CupyKernel, NumpyKernel
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Final

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@register_backend(Cupy)  # type: ignore[type-abstract]
class CupyBackend(Backend):
    name: Final[str] = "cuda"

    _cupy: Cupy

    @property
    def nplike(self) -> Cupy:
        return self._cupy

    def __init__(self):
        self._cupy = Cupy.instance()

    def __getitem__(
        self, index: KernelKeyType
    ) -> CudaComputeKernel | CupyKernel | NumpyKernel:
        from awkward._connect import cuda

        kernel_name = index[0] if index else ""
        print("Calling kernel:", kernel_name)

        cupy = cuda.import_cupy("Awkward Arrays with CUDA")
        _cuda_kernels = cuda.initialize_cuda_kernels(cupy)
        func = _cuda_kernels[index]

        compute_impl = self._get_cuda_compute_impl(kernel_name)
        if compute_impl is not None:
            print("||  ", compute_impl)
            for idx in index:
                print("||||", idx)
            return CudaComputeKernel(compute_impl, index)

        if func is not None:
            # Return CudaComputeKernel for supported operations
            return CupyKernel(func, index)
        else:
            raise AssertionError(f"CuPyKernel not found: {index!r}")

    def _get_cuda_compute_impl(self, kernel_name: str):
        """
        Get the cuda.compute implementation for a kernel operation.
        Args:
            kernel_name: Name of the kernel operation (e.g., "awkward_sort")
        Returns:
            Callable implementing the operation, or None if not supported
        """
        from awkward._connect.cuda import cccl_kernels

        if getattr(cccl_kernels, kernel_name, False):
            return getattr(cccl_kernels, kernel_name)

        return None
