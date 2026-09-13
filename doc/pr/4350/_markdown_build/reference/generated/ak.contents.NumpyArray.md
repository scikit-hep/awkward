# ak.contents.NumpyArray

Defined in [awkward.contents.numpyarray](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/numpyarray.py) on [line 63](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/numpyarray.py#L63).

#### *class* ak.contents.NumpyArray(data: awkward._nplikes.array_like.ArrayLike, \*, parameters=None, backend=None)

A NumpyArray describes 1-dimensional or rectilinear data using a NumPy
`np.ndarray`, a CuPy `cp.ndarray`, etc., depending on the backend.

This class is aware of the rectilinear array’s `shape` and `strides`, and
allows for arbitrary `strides`, such as Fortran-ordered data. However, many
operations require C-contiguous data, so derivatives of Fortran-ordered
arrays may not be Fortran-ordered.

Only a subset of `dtype` values are allowed, and only for your system’s
native endianness:

* `bool`: boolean, like NumPy’s `np.bool_` (considered distinct from integers)
* `int8`: signed 8-bit
* `uint8`: unsigned 8-bit
* `int16`: signed 16-bit
* `uint16`: unsigned 16-bit
* `int32`: signed 32-bit
* `uint32`: unsigned 32-bit
* `int64`: signed 64-bit
* `uint64`: unsigned 64-bit
* `float16`: floating point 16-bit, if your system’s NumPy supports it
* `float32`: floating point 32-bit
* `float64`: floating point 64-bit
* `float128`: floating point 128-bit, if your system’s NumPy supports it
* `complex64`: floating complex numbers composed of 32-bit real/imag parts
* `complex128`: floating complex numbers composed of 64-bit real/imag parts
* `complex256`: floating complex numbers composed of 128-bit real/imag parts, if your system’s NumPy supports it
* `datetime64`: date/time, origin is midnight on January 1, 1970, in any units NumPy supports
* `timedelta64`: time difference, in any units NumPy supports

If the `shape` is one-dimensional, a NumpyArray corresponds to an Apache
Arrow [Primitive array](https://arrow.apache.org/docs/format/Columnar.html#fixed-size-primitive-layout).

To illustrate how the constructor arguments are interpreted, the following is a
simplified implementation of `__init__`, `__len__`, and `__getitem__`:

```default
class NumpyArray(Content):
    def __init__(self, data):
        assert isinstance(data, numpy_like_array)
        assert data.dtype in allowed_dtypes
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, where):
        result = self.data[where]
        if isinstance(result, numpy_like_array):
            return NumpyArray(result)
        else:
            return result
```

#### \_data

#### *property* data *: awkward._nplikes.array_like.ArrayLike*

#### form_cls *: awkward._typing.Final*

#### copy(data=UNSET, \*, parameters=UNSET, backend=UNSET)

#### \_\_copy_\_()

#### \_\_deepcopy_\_(memo)

#### *classmethod* simplified(data, \*, parameters=None, backend=None)

#### *property* shape *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...]*

#### *property* inner_shape *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...]*

#### *property* strides *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._nplikes.shape.ShapeItem, ...]*

#### *property* dtype *: np*

#### \_raw(nplike=None)

#### \_form_with_key(getkey: awkward._typing.Callable[[awkward.contents.content.Content], [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)]) → awkward.forms.numpyform.NumpyForm

#### \_form_with_key_path(path: awkward.forms.form.FormKeyPathT) → awkward.forms.numpyform.NumpyForm

#### \_to_buffers(form: awkward.forms.form.Form, getkey: awkward._typing.Callable[[awkward.contents.content.Content, awkward.forms.form.Form, [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], container: [collections.abc.MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._nplikes.array_like.ArrayLike], backend: awkward._backends.backend.Backend, byteorder: [str](https://docs.python.org/3/library/stdtypes.html#str))

#### \_to_typetracer(forget_length: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self

#### \_touch_data(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### \_touch_shape(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### *property* length *: awkward._nplikes.shape.ShapeItem*

#### \_\_repr_\_()

#### \_repr(indent, pre, post)

#### to_RegularArray()

#### maybe_to_NumpyArray() → awkward._typing.Self

#### \_\_iter_\_()

#### \_getitem_nothing()

#### \_is_getitem_at_placeholder() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_is_getitem_at_virtual() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_getitem_at(where: awkward._nplikes.numpy_like.IndexType)

#### \_getitem_range(start: awkward._nplikes.numpy_like.IndexType, stop: awkward._nplikes.numpy_like.IndexType) → awkward.contents.content.Content

#### \_getitem_field(where: [str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._typing.SupportsIndex, only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → awkward.contents.content.Content

#### \_getitem_fields(where: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._typing.SupportsIndex], only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → awkward.contents.content.Content

#### \_carry(carry: awkward.index.Index, allow_lazy: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward.contents.content.Content

#### \_getitem_next_jagged(slicestarts: awkward.index.Index, slicestops: awkward.index.Index, slicecontent: awkward.contents.content.Content, tail) → awkward.contents.content.Content

#### \_getitem_next(head: awkward._slicing.SliceItem | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple), tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None)) → awkward.contents.content.Content

#### \_offsets_and_flattened(axis: [int](https://docs.python.org/3/library/functions.html#int), depth: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward.index.Index, awkward.contents.content.Content]

#### \_mergeable_next(other: awkward.contents.content.Content, mergebool: [bool](https://docs.python.org/3/library/functions.html#bool), mergecastable: awkward._typing.Literal[same_kind, equiv, NumpyArray._mergeable_next.family]) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_mergemany(others: [collections.abc.Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[awkward.contents.content.Content]) → awkward.contents.content.Content

#### \_fill_none(value: awkward.contents.content.Content) → awkward.contents.content.Content

#### \_local_index(axis, depth)

#### to_contiguous() → awkward._typing.Self

#### *property* is_contiguous *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### \_subranges_equal(starts, stops, length, sorted=True)

#### \_as_unique_strings(offsets)

#### \_numbers_to_type(name, including_unknown)

#### \_is_unique(negaxis, starts, offsets, outlength)

#### \_unique(negaxis, starts, offsets, outlength)

#### \_argsort_next(negaxis, starts, shifts, offsets, outlength, ascending, stable)

#### \_sort_next(negaxis, starts, offsets, outlength, ascending, stable)

#### \_combinations(n, replacement, recordlookup, parameters, axis, depth)

#### \_reduce_next(reducer, negaxis, starts, shifts, offsets, outlength, mask, keepdims, behavior)

#### \_validity_error(path)

#### \_pad_none(target, axis, depth, clip)

#### \_nbytes_part()

#### \_to_arrow(pyarrow: awkward._typing.Any, mask_node: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), validbytes: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int), options: awkward.contents.content.ToArrowOptions)

#### \_to_cudf(cudf: awkward._typing.Any, mask: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int))

#### \_to_backend_array(allow_missing, backend)

#### \_remove_structure(backend: awkward._backends.backend.Backend, options: awkward.contents.content.RemoveStructureOptions) → [list](https://docs.python.org/3/library/stdtypes.html#list)[awkward.contents.content.Content]

#### \_recursively_apply(action: awkward.contents.content.ImplementsApplyAction, depth: [int](https://docs.python.org/3/library/functions.html#int), depth_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), lateral_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), options: awkward.contents.content.ApplyActionOptions) → awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None)

#### \_to_packed(recursive: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → awkward._typing.Self

#### \_to_list(behavior, json_conversions)

#### \_to_backend(backend: awkward._backends.backend.Backend) → awkward._typing.Self

#### \_materialize(type_) → awkward._typing.Self

#### *property* \_is_all_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* \_is_any_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### \_is_equal_to(other: awkward._typing.Self, index_dtype: [bool](https://docs.python.org/3/library/functions.html#bool), numpyarray: [bool](https://docs.python.org/3/library/functions.html#bool), all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_to_regular_primitive() → awkward.contents.RegularArray
