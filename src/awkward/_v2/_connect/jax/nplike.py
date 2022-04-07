# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

from collections.abc import Iterable


class Jax(ak.nplike.NumpyLike):
    def to_rectilinear(self, array, *args, **kwargs):
        if isinstance(array, self._module.DeviceArray):
            return array

        elif isinstance(
            array,
            (
                ak.Array,
                ak.Record,
                ak.ArrayBuilder,
                ak.layout.Content,
                ak.layout.Record,
                ak.layout.ArrayBuilder,
                ak.layout.LayoutBuilder32,
                ak.layout.LayoutBuilder64,
            ),
        ):
            return ak.operations.convert.to_jax(array, *args, **kwargs)

        elif isinstance(array, Iterable):
            return [self.to_rectilinear(x, *args, **kwargs) for x in array]

        else:
            ak._v2._util.error(ValueError("to_rectilinear argument must be iterable"))

    def __getitem__(self, name_and_types):
        ak._v2._util.error(
            ValueError("__getitem__ for JAX Kernels is not implemented yet")
        )

    def __init__(self):
        from awkward._v2._connect.jax import import_jax  # noqa: F401

        self._module = import_jax().numpy

    @property
    def ma(self):
        ak._v2._util.error(
            ValueError(
                "JAX arrays cannot have missing values until JAX implements "
                "numpy.ma.MaskedArray" + ak._util.exception_suffix(__file__)
            )
        )

    @property
    def char(self):
        ak._v2._util.error(
            ValueError(
                "JAX arrays cannot do string manipulations until JAX implements "
                "numpy.char"
            )
        )

    @property
    def ndarray(self):
        return self._module.ndarray

    def asarray(self, array, dtype=None, order=None):
        return self._module.asarray(array, dtype=dtype, order="K")

    def raw(self, array, nplike):
        if isinstance(nplike, Jax):
            return array
        elif isinstance(nplike, ak.nplike.Cupy):
            cupy = ak.nplike.Cupy.instance()
            return cupy.asarray(array)
        elif isinstance(nplike, ak.nplike.Numpy):
            numpy = ak.nplike.Numpy.instance()
            return numpy.asarray(array)
        elif isinstance(nplike, ak._v2._typetracer.TypeTracer):
            return ak._v2._typetracer.TypeTracerArray(
                dtype=array.dtype, shape=array.shape
            )
        else:
            ak._v2._util.error(
                TypeError(
                    "Invalid nplike, choose between nplike.Numpy, nplike.Cupy, Typetracer or Jax",
                )
            )

    def ascontiguousarray(self, array, dtype=None):
        ak._v2._util.error(NotImplementedError)

    def zeros(self, *args, **kwargs):
        return self._module.zeros(*args, **kwargs)

    def frombuffer(self, *args, **kwargs):
        import numpy

        np_array = numpy.frombuffer(*args, **kwargs)
        return self._module.asarray(np_array)

    def array_equal(self, array1, array2):
        return self._module.array_equal(array1, array2)

    def repeat(self, array, repeats, axis=None):
        return self._module.repeat(array, repeats, axis)

    def nan_to_num(self, *args, **kwargs):
        return self._module.nan_to_num(*args, **kwargs)

    # For all reducers: JAX returns zero-dimensional arrays like CuPy

    def all(self, *args, **kwargs):
        out = self._module.all(*args, **kwargs)
        if out.shape == ():
            return out.item()
        return out

    def any(self, *args, **kwargs):
        out = self._module.any(*args, **kwargs)
        if out.shape == ():
            return out.item()
        return out

    def count_nonzero(self, *args, **kwargs):
        out = self._module.count_nonzero(*args, **kwargs)
        if out.shape == ():
            return out.item()
        return out

    def sum(self, *args, **kwargs):
        out = self._module.sum(*args, **kwargs)
        if out.shape == ():
            return out.item()
        return out

    def prod(self, *args, **kwargs):
        out = self._module.prod(*args, **kwargs)
        if out.shape == ():
            return out.item()
        return out

    def min(self, *args, **kwargs):
        out = self._module.min(*args, **kwargs)
        if out.shape == ():
            return out.item()
        return out

    def max(self, *args, **kwargs):
        out = self._module.max(*args, **kwargs)
        if out.shape == ():
            return out.item()
        return out

    def argmin(self, *args, **kwargs):
        out = self._module.argmin(*args, **kwargs)
        if out.shape == ():
            return out.item()
        return out

    def argmax(self, *args, **kwargs):
        out = self._module.argmax(*args, **kwargs)
        if out.shape == ():
            return out.item()
        return out

    def array_str(
        self, array, max_line_width=None, precision=None, suppress_small=None
    ):
        # array, max_line_width, precision=None, suppress_small=None
        return self._module.array_str(array, max_line_width, precision, suppress_small)
