# ak.contents.ListArray

Defined in [awkward.contents.listarray](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/listarray.py) on [line 55](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/listarray.py#L55).

#### *class* ak.contents.ListArray(starts, stops, content, \*, parameters=None)

ListArray generalizes [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) by not
requiring its `content` to be in increasing order and by allowing it to
have unreachable elements between lists. Instead of a single `offsets` buffer,
ListArray has

* `starts`: The starting index of each list.
* `stops`: The stopping index of each list.

[`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) `offsets` may be related to `starts` and
`stops` by:

```default
starts = offsets[:-1]
stops = offsets[1:]
```

ListArrays are a common by-product of structure manipulation: as a result of
some operation, we might want to view slices or permutations of the `content`
without copying it to make a contiguous version of it. For that reason,
ListArrays are more useful in a data-manipulation library like Awkward Array
than in a data-representation library like Apache Arrow.

Like [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) and [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray), a ListArray can
represent strings if its `__array__` parameter is `"string"` (UTF-8 assumed) or
`"bytestring"` (no encoding assumed) and it contains an [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)
of `dtype=np.uint8` whose `__array__` parameter is `"char"` (UTF-8 assumed) or
`"byte"` (no encoding assumed).

There is no equivalent of ListArray in Apache Arrow.

To illustrate how the constructor arguments are interpreted, the following is a
simplified implementation of `__init__`, `__len__`, and `__getitem__`:

```default
class ListArray(Content):
    def __init__(self, starts, stops, content):
        assert isinstance(starts, (Index32, IndexU32, Index64))
        assert isinstance(stops, type(starts))
        assert isinstance(content, Content)
        assert len(stops) >= len(starts)  # usually equal
        for i in range(len(starts)):
            start = starts[i]
            stop = stops[i]
            if start != stop:
                assert start < stop  # i.e. start <= stop
                assert start >= 0
                assert stop <= len(content)
        self.starts = starts
        self.stops = stops
        self.content = content

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, where):
        if isinstance(where, int):
            if where < 0:
                where += len(self)
            assert 0 <= where < len(self)
            return self.content[self.starts[where] : self.stops[where]]

        elif isinstance(where, slice) and where.step is None:
            starts = self.starts[where.start : where.stop]
            stops = self.stops[where.start : where.stop]
            return ListArray(starts, stops, self.content)

        elif isinstance(where, str):
            return ListArray(self.starts, self.stops, self.content[where])

        else:
            raise AssertionError(where)
```

#### \_starts

#### \_stops

#### \_content

#### *property* starts *: awkward.index.Index*

#### *property* stops *: awkward.index.Index*

#### form_cls *: awkward._typing.Final*

#### copy(starts=UNSET, stops=UNSET, content=UNSET, \*, parameters=UNSET)

#### \_\_copy_\_()

#### \_\_deepcopy_\_(memo)

#### *classmethod* simplified(starts, stops, content, \*, parameters=None)

#### \_form_with_key(getkey: awkward._typing.Callable[[awkward.contents.content.Content], [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)]) → awkward.forms.listform.ListForm

#### \_form_with_key_path(path: awkward.forms.form.FormKeyPathT) → awkward.forms.listform.ListForm

#### \_to_buffers(form: awkward.forms.form.Form, getkey: awkward._typing.Callable[[awkward.contents.content.Content, awkward.forms.form.Form, [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], container: [collections.abc.MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._nplikes.array_like.ArrayLike], backend: awkward._backends.backend.Backend, byteorder: [str](https://docs.python.org/3/library/stdtypes.html#str))

#### \_to_typetracer(forget_length: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self

#### \_touch_data(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### \_touch_shape(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### *property* length *: awkward._nplikes.shape.ShapeItem*

#### \_\_repr_\_()

#### \_repr(indent, pre, post)

#### to_ListOffsetArray64(start_at_zero: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → awkward.contents.listoffsetarray.ListOffsetArray

#### to_RegularArray()

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

#### \_to_arrow(pyarrow: awkward._typing.Any, mask_node: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), validbytes: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int), options: awkward.contents.content.ToArrowOptions)

#### \_to_cudf(cudf: awkward._typing.Any, mask: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int))

#### \_to_backend_array(allow_missing, backend)

#### \_remove_structure(backend: awkward._backends.backend.Backend, options: awkward.contents.content.RemoveStructureOptions) → [list](https://docs.python.org/3/library/stdtypes.html#list)[awkward.contents.content.Content]

#### \_drop_none() → awkward.contents.content.Content

#### \_rebuild_without_nones(none_indexes, new_content)

#### \_recursively_apply(action: awkward.contents.content.ImplementsApplyAction, depth: [int](https://docs.python.org/3/library/functions.html#int), depth_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), lateral_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), options: awkward.contents.content.ApplyActionOptions) → awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None)

#### \_to_packed(recursive: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → awkward._typing.Self

#### \_to_list(behavior, json_conversions)

#### \_to_backend(backend: awkward._backends.backend.Backend) → awkward._typing.Self

#### \_materialize(type_) → awkward._typing.Self

#### *property* \_is_all_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* \_is_any_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### \_is_equal_to(other: awkward._typing.Self, index_dtype: [bool](https://docs.python.org/3/library/functions.html#bool), numpyarray: [bool](https://docs.python.org/3/library/functions.html#bool), all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
