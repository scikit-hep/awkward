# ak.typetracer.PlaceholderArray

Defined in [awkward._nplikes.placeholder](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/_nplikes/placeholder.py) on [line 16](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/_nplikes/placeholder.py#L16).

#### *class* ak.typetracer.PlaceholderArray(nplike: awkward._nplikes.numpy_like.NumpyLike, shape: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...], dtype: awkward._typing.DType, buffer_key: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Base class for protocol classes.

Protocol classes are defined as:

```default
class Proto(Protocol):
    def meth(self) -> int:
        ...
```

Such classes are primarily used with static type checkers that recognize
structural subtyping (static duck-typing), for example:

```default
class C:
    def meth(self) -> int:
        return 0

def func(x: Proto) -> int:
    return x.meth()

func(C())  # Passes static type check
```

See PEP 544 for details. Protocol classes decorated with
@typing.runtime_checkable act as simple-minded runtime protocols that check
only the presence of given attributes, ignoring their type signatures.
Protocol classes can be generic, they are defined as:

```default
class GenProto(Protocol[T]):
    def meth(self) -> T:
        ...
```

#### \_nplike

#### \_shape

#### \_dtype

#### \_buffer_key *= None*

#### *property* buffer_key *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### *property* dtype *: awkward._typing.DType*

#### *property* shape *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...]*

#### *property* ndim *: [int](https://docs.python.org/3/library/functions.html#int)*

#### *property* size *: awkward._nplikes.shape.ShapeItem*

#### *property* nbytes *: [int](https://docs.python.org/3/library/functions.html#int)*

#### materialize()

#### *property* strides *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...]*

#### *property* T

#### view(dtype: numpy.typing.DTypeLike) → awkward._typing.Self

#### \_\_repr_\_()

#### \_\_getitem_\_(index)

#### tolist()

#### \_\_setitem_\_(key, value)

#### \_\_bool_\_()

#### \_\_int_\_()

#### \_\_index_\_()

#### \_\_len_\_() → [int](https://docs.python.org/3/library/functions.html#int)

#### \_\_add_\_(other)

#### \_\_and_\_(other)

#### \_\_eq_\_(other)

#### \_\_floordiv_\_(other)

#### \_\_ge_\_(other)

#### \_\_gt_\_(other)

#### \_\_invert_\_()

#### \_\_le_\_(other)

#### \_\_lt_\_(other)

#### \_\_mul_\_(other)

#### \_\_or_\_(other)

#### \_\_sub_\_(other)

#### \_\_truediv_\_(other)

#### \_\_iter_\_()

#### \_\_array_\_(dtype=None, copy=None)

#### \_\_cupy_get_ndarray_\_()

#### \_\_jax_array_\_()

#### *property* \_\_array_interface_\_

#### *property* \_\_cuda_array_interface_\_

#### \_\_dlpack_device_\_()

#### \_\_dlpack_\_(stream: awkward._typing.Any = None)
