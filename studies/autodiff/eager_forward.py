# Eager, forward-mode autodiff (autograd)
# Backpropagation will probably require collecting a DAG with a typetracer or Dask
#
# Presented at https://indico.cern.ch/event/1387764/
#
# The following are good references:
#
# https://www.hedonisticlearning.com/posts/complex-step-differentiation.html
# https://researchrepository.wvu.edu/faculty_publications/426/

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

class diffarray(NDArrayOperatorsMixin):
    __slots__ = ("_array",)

    @classmethod
    def _build(cls, complex_array):
        "Manual constructor from a `complex_array`."
        self = cls.__new__(cls)
        self._array = complex_array
        return self

    def __init__(self, primal, tangent=None, *, dtype=None):
        "Constructor for floating-point `primal` and (optional) `tangent`."
        if dtype is None:
            dtype = primal.dtype.type
        elif isinstance(dtype, np.dtype):
            dtype = dtype.type

        if issubclass(dtype, np.float32):
            self._array = primal.astype(np.complex64)
        elif issubclass(dtype, np.float64):
            self._array = primal.astype(np.complex128)
        else:
            raise TypeError("only float32 or float64 arrays can be differentiated")

        self._array += (1 if tangent is None else tangent) * 1j * self._step_scale

    @property
    def _step_scale(self):
        "Size of the complex step; half precision of 1.0."
        return 1e-4 if issubclass(self._array.dtype.type, np.complex128) else 1e-8

    @property
    def primal(self):
        "Array of primary values."
        return np.real(self._array)

    @property
    def tangent(self):
        "Array of derivatives."
        return np.imag(self._array) / self._step_scale

    def __str__(self):
        primal = str(self.primal).replace("\n", "\n         ")
        tangent = str(self.tangent).replace("\n", "\n         ")
        return f"primal:  {primal}\ntangent: {tangent}"

    def __repr__(self):
        primal = str(self.primal).replace("\n", "\n          ")
        tangent = str(self.tangent).replace("\n", "\n          ")
        dtype = ""
        if issubclass(self._array.dtype.type, np.complex64):
            dtype = ",\n          dtype=np.float32"
        return f"diffarray({primal},\n          {tangent}{dtype})"

    def _prepare(self, args, kwargs):
        "Used in NEP-13 and NEP-18 overrides."
        cls = type(self)
        args = [x._array if isinstance(x, cls) else x for x in args]
        kwargs = {k: v._array if isinstance(x, cls) else v for k, v in kwargs.items()}
        return cls, args, kwargs

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        "https://numpy.org/neps/nep-0013-ufunc-overrides.html"
        if ufunc.__name__ == "absolute":
            # interpret `absolute` only on the primal
            if len(kwargs) != 0:
                raise NotImplementedError("kwargs in np.absolute")
            arg = args[0]._array
            out = arg.copy()
            out[arg.real < 0] *= -1
            return type(self)._build(out)

        if ufunc.__name__ in (
            "less", "less_equal", "equal", "not_equal", "greater", "greater_equal"
        ):
            # do comparisons only on the primal
            cls = type(self)
            args = [x._array.real if isinstance(x, cls) else x for x in args]
            return getattr(ufunc, method)(*args, **kwargs)

        cls, prepared_args, prepared_kwargs = self._prepare(args, kwargs)
        out = getattr(ufunc, method)(*prepared_args, **prepared_kwargs)
        if issubclass(out.dtype.type, np.complexfloating):
            return cls._build(out)
        else:
            return out

    def __array_function__(self, func, types, args, kwargs):
        "https://numpy.org/neps/nep-0018-array-function-protocol.html"
        if func.__name__ == "real":
            # interpret `real` only on the primal
            return type(self)._build(args[0]._array)
        if func.__name__ == "imag":
            # interpret `imag` only on the primal
            return type(self)._build(args[0]._array * 0)

        cls, prepared_args, prepared_kwargs = self._prepare(args, kwargs)
        out = func(*prepared_args, **prepared_kwargs)
        if issubclass(out.dtype.type, np.complexfloating):
            return cls._build(out)
        else:
            return out

    def __getitem__(self, where):
        out = self._array[where]
        if isinstance(out, np.complexfloating):
            # NumPy returns a scalar; CuPy and Array API return an array
            # we return an array to keep derivatives
            return type(self)._build(np.asarray(out))
        return out

# >>> x = np.linspace(-20, 20, 10000)
# >>> da_x = diffarray(x)
# >>> da_y = np.sin(da_x) / da_x
# >>> da_x
# diffarray([-20.        -19.9959996 -19.9919992 ...  19.9919992  19.9959996
#             20.       ],
#           [1. 1. 1. ... 1. 1. 1.])
# >>> da_y
# diffarray([0.04564726 0.04557439 0.04550076 ... 0.04550076 0.04557439 0.04564726],
#           [-0.01812174 -0.01831149 -0.01850102 ...  0.01850102  0.01831149
#             0.01812174])
# >>> abs(da_y.tangent - ((x*np.cos(x) - np.sin(x)) / x**2)).max()
# 3.9683650809863025e-10
# >>> import matplotlib.pyplot as plt
# >>> plt.plot(x, da_y.tangent)
# >>> plt.plot(x, (x*np.cos(x) - np.sin(x)) / x**2, ls="--")
#
# See https://gist.github.com/jpivarski/8dc48a87bae7a856848f87e36b9d244d for the plot
