# ak.index.Index

Defined in [awkward.index](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/index.py) on [line 44](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/index.py#L44).

#### *class* ak.index.Index(data, \*, metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None, nplike: awkward._nplikes.numpy_like.NumpyLike | [None](https://docs.python.org/3/library/constants.html#None) = None)

#### \_expected_dtype *: awkward._typing.DType | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### \_metadata *= None*

#### \_data

#### *classmethod* zeros(length: awkward._nplikes.shape.ShapeItem, nplike: awkward._nplikes.numpy_like.NumpyLike, dtype: awkward._typing.DType | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Index](sphinx-llm:207c886cfac7437cbe58f19bd52e4a87)

#### *classmethod* empty(length: awkward._nplikes.shape.ShapeItem, nplike: awkward._nplikes.numpy_like.NumpyLike, dtype: awkward._typing.DType | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Index](sphinx-llm:207c886cfac7437cbe58f19bd52e4a87)

#### *property* data *: awkward._nplikes.array_like.ArrayLike*

#### *property* nplike *: awkward._nplikes.numpy_like.NumpyLike*

#### *property* dtype *: awkward._typing.DType*

#### *property* metadata *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)*

#### *property* ptr

#### *property* length *: awkward._nplikes.shape.ShapeItem*

#### forget_length() → awkward._typing.Self

#### raw(nplike: awkward._nplikes.numpy_like.NumpyLike) → awkward._nplikes.array_like.ArrayLike

#### materialize(type_) → [Index](sphinx-llm:207c886cfac7437cbe58f19bd52e4a87)

#### *property* is_all_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* is_any_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### \_\_len_\_() → [int](https://docs.python.org/3/library/functions.html#int)

#### *property* \_\_cuda_array_interface_\_

#### *property* \_\_array_interface_\_

#### \_\_dlpack_device_\_() → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]

#### \_\_dlpack_\_(stream: awkward._typing.Any = None) → awkward._typing.Any

#### \_\_repr_\_() → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### \_repr(indent: [str](https://docs.python.org/3/library/stdtypes.html#str), pre: [str](https://docs.python.org/3/library/stdtypes.html#str), post: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### *property* form *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### \_\_getitem_\_(where)

#### \_\_setitem_\_(where, what)

#### to64() → [Index](sphinx-llm:207c886cfac7437cbe58f19bd52e4a87)

#### \_\_copy_\_() → awkward._typing.Self

#### \_\_deepcopy_\_(memo: [dict](https://docs.python.org/3/library/stdtypes.html#dict)) → awkward._typing.Self

#### \_nbytes_part() → awkward._nplikes.shape.ShapeItem

#### to_nplike(nplike: awkward._nplikes.numpy_like.NumpyLike) → awkward._typing.Self

#### is_equal_to(other: awkward._typing.Any, index_dtype: [bool](https://docs.python.org/3/library/functions.html#bool) = True, numpyarray: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_touch_data()

#### \_touch_shape()
