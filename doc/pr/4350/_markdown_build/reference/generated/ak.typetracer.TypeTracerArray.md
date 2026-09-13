# ak.typetracer.TypeTracerArray

Defined in [awkward._nplikes.typetracer](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/_nplikes/typetracer.py) on [line 239](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/_nplikes/typetracer.py#L239).

#### *class* ak.typetracer.TypeTracerArray

Mixin defining all operator special methods using \_\_array_ufunc_\_.

This class implements the special methods for almost all of Python’s
builtin operators defined in the `operator` module, including comparisons
(`==`, `>`, etc.) and arithmetic (`+`, `*`, `-`, etc.), by
deferring to the `__array_ufunc__` method, which subclasses must
implement.

It is useful for writing classes that do not inherit from `numpy.ndarray`,
but that should support arithmetic and numpy universal functions like
arrays as described in 

```
``
```

A Mechanism for Overriding Ufuncs
<[https://numpy.org/neps/nep-0013-ufunc-overrides.html](https://numpy.org/neps/nep-0013-ufunc-overrides.html)>\`\`_.

As an trivial example, consider this implementation of an `ArrayLike`
class that simply wraps a NumPy array and ensures that the result of any
arithmetic operation is also an `ArrayLike` object:

```default
class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, value):
        self.value = np.asarray(value)

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (ArrayLike,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, ArrayLike) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.value if isinstance(x, ArrayLike) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.value)
```

In interactions between `ArrayLike` objects and numbers or numpy arrays,
the result is always another `ArrayLike`:

```pycon
>>> x = ArrayLike([1, 2, 3])
>>> x - 1
ArrayLike(array([0, 1, 2]))
>>> 1 - x
ArrayLike(array([ 0, -1, -2]))
>>> np.arange(3) - x
ArrayLike(array([-1, -1, -1]))
>>> x - np.arange(3)
ArrayLike(array([1, 1, 1]))
```

Note that unlike `numpy.ndarray`, `ArrayLike` does not allow operations
with arbitrary, unrecognized types. This ensures that interactions with
ArrayLike preserve a well-defined casting hierarchy.

#### Versionadded
Added in version 1.13.

#### \_dtype *: [numpy.dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)*

#### \_shape *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...]*

#### runtime_typechecks *= True*

#### \_\_reduce_\_()

#### *classmethod* \_new(dtype: awkward._typing.DType, shape: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...], form_key: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, report: [TypeTracerReport](sphinx-llm:62560cc96b5946c48e279ed5d9c8eb04#ak.typetracer.TypeTracerReport) | [None](https://docs.python.org/3/library/constants.html#None) = None)

#### \_\_repr_\_()

#### \_\_str_\_()

#### *property* T *: awkward._typing.Self*

#### *property* dtype *: awkward._typing.DType*

#### *property* size *: awkward._nplikes.shape.ShapeItem*

#### *property* shape *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...]*

#### *property* strides

#### *property* inner_shape *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...]*

#### *property* form_key *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### *property* report *: [TypeTracerReport](sphinx-llm:62560cc96b5946c48e279ed5d9c8eb04#ak.typetracer.TypeTracerReport) | [None](https://docs.python.org/3/library/constants.html#None)*

#### touch_shape()

#### touch_data()

#### *property* nplike *: TypeTracer*

#### *property* ndim *: [int](https://docs.python.org/3/library/functions.html#int)*

#### *property* nbytes *: awkward._nplikes.shape.ShapeItem*

#### view(dtype: numpy.typing.DTypeLike) → awkward._typing.Self

#### forget_length() → awkward._typing.Self

#### \_\_iter_\_()

#### \_\_array_\_(dtype=None, copy=None)

#### *property* ctypes

#### \_\_len_\_()

#### \_\_getitem_\_(key: awkward._typing.SupportsIndex | [slice](https://docs.python.org/3/library/functions.html#slice) | awkward._typing.EllipsisType | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._typing.SupportsIndex | [slice](https://docs.python.org/3/library/functions.html#slice) | awkward._typing.EllipsisType | awkward._nplikes.numpy_like.ArrayLike, ...] | awkward._nplikes.numpy_like.ArrayLike) → awkward._typing.Self

#### \_\_setitem_\_(key: awkward._typing.SupportsIndex | [slice](https://docs.python.org/3/library/functions.html#slice) | awkward._typing.EllipsisType | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._typing.SupportsIndex | [slice](https://docs.python.org/3/library/functions.html#slice) | awkward._typing.EllipsisType | awkward._nplikes.numpy_like.ArrayLike, ...] | awkward._nplikes.numpy_like.ArrayLike, value: [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [bool](https://docs.python.org/3/library/functions.html#bool) | [complex](https://docs.python.org/3/library/functions.html#complex) | awkward._nplikes.numpy_like.ArrayLike)

#### copy()

#### \_\_array_ufunc_\_(ufunc, method, \*inputs, \*\*kwargs)

#### tolist() → [list](https://docs.python.org/3/library/stdtypes.html#list)

#### \_\_bool_\_() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_\_int_\_() → [int](https://docs.python.org/3/library/functions.html#int)

#### \_\_index_\_() → [int](https://docs.python.org/3/library/functions.html#int)

#### \_\_dlpack_device_\_() → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]

#### \_\_dlpack_\_(stream: awkward._typing.Any = None) → awkward._typing.Any

## Classes

| `_CTypes`   |    |
|-------------|----|
