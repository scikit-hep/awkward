# ak.contents.RegularArray

Defined in [awkward.contents.regulararray](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/regulararray.py) on [line 74](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/regulararray.py#L74).

#### *class* ak.contents.RegularArray(content, size, zeros_length=0, zeros_length_generator=None, \*, parameters=None)

RegularArray describes lists that all have the same length, the single
integer `size`. Its underlying `content` is a flattened view of the data;
that is, each list is not stored separately in memory, but is inferred as a
subinterval of the underlying data.

If the `content` length is not an integer multiple of `size`, then the length
of the RegularArray is truncated to the largest integer multiple.

An extra field `zeros_length` is ignored unless the `size` is zero. This sets the
length of the RegularArray in only those cases, so that it is possible for an
array to contain a non-zero number of zero-length lists with regular type.

A multidimensional [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) is equivalent to a one-dimensional
`ak.layout.NumpyArray` nested within several RegularArrays, one for each
dimension. However, RegularArrays can be used to make lists of any other type.

Like [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) and [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray), a RegularArray can
represent strings if its `__array__` parameter is `"string"` (UTF-8 assumed) or
`"bytestring"` (no encoding assumed) and it contains an [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)
of `dtype=np.uint8` whose `__array__` parameter is `"char"` (UTF-8 assumed) or
`"byte"` (no encoding assumed).

RegularArray corresponds to an Apache Arrow
[FixedSizeList](https://arrow.apache.org/docs/format/Columnar.html#fixed-size-list-layout).

To illustrate how the constructor arguments are interpreted, the following is a
simplified implementation of `__init__`, `__len__`, and `__getitem__`:

```default
class RegularArray(Content):
    def __init__(self, content, size, zeros_length=0):
        assert isinstance(content, Content)
        assert isinstance(size, int)
        assert isinstance(zeros_length, int)
        assert size >= 0
        if size != 0:
            length = len(content) // size  # floor division
        else:
            assert zeros_length >= 0
            length = zeros_length
        self.content = content
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, where):
        if isinstance(where, int):
            if where < 0:
                where += len(self)
            assert 0 <= where < len(self)
            return self.content[(where) * self.size : (where + 1) * self.size]

        elif isinstance(where, slice) and where.step is None:
            start = where.start * self.size
            stop = where.stop * self.size
            zeros_length = where.stop - where.start
            return RegularArray(
                self.content[start:stop], self.size, zeros_length
            )

        elif isinstance(where, str):
            return RegularArray(self.content[where], self.size, self.length)

        else:
            raise AssertionError(where)
```

#### \_content

#### \_size

#### \_length

#### \_zeros_length_generator *= None*

#### *property* size

#### form_cls *: awkward._typing.Final*

#### copy(content=UNSET, size=UNSET, zeros_length=UNSET, zeros_length_generator=UNSET, \*, parameters=UNSET)

#### \_\_copy_\_()

#### \_\_deepcopy_\_(memo)

#### \_\_getstate_\_()

#### *classmethod* simplified(content, size, zeros_length=0, zeros_length_generator=None, \*, parameters=None)

#### *property* offsets *: awkward.index.Index*

#### *property* starts *: awkward.index.Index*

#### *property* stops

#### \_form_with_key(getkey: awkward._typing.Callable[[awkward.contents.content.Content], [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)]) → awkward.forms.regularform.RegularForm

#### \_form_with_key_path(path: awkward.forms.form.FormKeyPathT) → awkward.forms.regularform.RegularForm

#### \_to_buffers(form: awkward.forms.form.Form, getkey: awkward._typing.Callable[[awkward.contents.content.Content, awkward.forms.form.Form, [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], container: [collections.abc.MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._nplikes.array_like.ArrayLike], backend: awkward._backends.backend.Backend, byteorder: [str](https://docs.python.org/3/library/stdtypes.html#str))

#### \_to_typetracer(forget_length: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self

#### \_touch_data(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### \_touch_shape(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### *property* length *: awkward._nplikes.shape.ShapeItem*

#### \_\_repr_\_()

#### \_repr(indent, pre, post)

#### to_ListOffsetArray64(start_at_zero: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → awkward.contents.listoffsetarray.ListOffsetArray

#### to_RegularArray()

#### maybe_to_NumpyArray() → awkward.contents.NumpyArray | [None](https://docs.python.org/3/library/constants.html#None)

#### \_getitem_nothing()

#### \_is_getitem_at_placeholder() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_is_getitem_at_virtual() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_getitem_at(where: awkward._nplikes.numpy_like.IndexType)

#### \_getitem_range(start: awkward._nplikes.numpy_like.IndexType, stop: awkward._nplikes.numpy_like.IndexType) → awkward.contents.content.Content

#### \_getitem_field(where: [str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._typing.SupportsIndex, only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → awkward.contents.content.Content

#### \_getitem_fields(where: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._typing.SupportsIndex], only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → awkward.contents.content.Content

#### \_carry(carry: awkward.index.Index, allow_lazy: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward.contents.content.Content

#### \_compact_offsets64(start_at_zero)

#### \_broadcast_tooffsets64(offsets: awkward.index.Index) → awkward.contents.listoffsetarray.ListOffsetArray

#### \_getitem_next_jagged(slicestarts: awkward.index.Index, slicestops: awkward.index.Index, slicecontent: awkward.contents.content.Content, tail) → awkward.contents.content.Content

#### \_getitem_next(head: awkward._slicing.SliceItem | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple), tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None)) → awkward.contents.content.Content

#### \_offsets_and_flattened(axis: [int](https://docs.python.org/3/library/functions.html#int), depth: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward.index.Index, awkward.contents.content.Content]

#### \_mergeable_next(other: awkward.contents.content.Content, mergebool: [bool](https://docs.python.org/3/library/functions.html#bool), mergecastable: awkward._typing.Literal[same_kind, equiv, family]) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_mergemany(others: [collections.abc.Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[awkward.contents.content.Content]) → awkward.contents.content.Content

#### \_fill_none(value: awkward.contents.content.Content) → awkward.contents.content.Content

#### \_local_index(axis, depth)

#### \_numbers_to_type(name, including_unknown)

#### \_is_unique(negaxis, starts, offsets, outlength)

#### \_unique(negaxis, starts, offsets, outlength)

#### \_argsort_next(negaxis, starts, shifts, offsets, outlength, ascending, stable)

#### \_sort_next(negaxis, starts, offsets, outlength, ascending, stable)

#### \_combinations(n, replacement, recordlookup, parameters, axis, depth)

#### \_reduce_next(reducer, negaxis, starts, shifts, offsets, outlength, mask, keepdims, behavior)

#### \_validity_error(path)

#### \_nbytes_part()

#### \_pad_none(target, axis, depth, clip)

#### \_to_backend_array(allow_missing, backend)

#### \_to_arrow(pyarrow: awkward._typing.Any, mask_node: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), validbytes: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int), options: awkward.contents.content.ToArrowOptions)

#### \_remove_structure(backend: awkward._backends.backend.Backend, options: awkward.contents.content.RemoveStructureOptions) → [list](https://docs.python.org/3/library/stdtypes.html#list)[awkward.contents.content.Content]

#### \_drop_none() → awkward.contents.content.Content

#### \_recursively_apply(action: awkward.contents.content.ImplementsApplyAction, depth: [int](https://docs.python.org/3/library/functions.html#int), depth_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), lateral_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), options: awkward.contents.content.ApplyActionOptions) → awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None)

#### \_to_packed(recursive: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → awkward._typing.Self

#### \_to_list(behavior, json_conversions)

#### \_to_backend(backend: awkward._backends.backend.Backend) → awkward._typing.Self

#### \_materialize(type_) → awkward._typing.Self

#### *property* \_is_all_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* \_is_any_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### \_is_equal_to(other: awkward._typing.Self, index_dtype: [bool](https://docs.python.org/3/library/functions.html#bool), numpyarray: [bool](https://docs.python.org/3/library/functions.html#bool), all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
