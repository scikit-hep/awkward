# ak.contents.IndexedOptionArray

Defined in [awkward.contents.indexedoptionarray](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/indexedoptionarray.py) on [line 57](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/indexedoptionarray.py#L57).

#### *class* ak.contents.IndexedOptionArray(index, content, \*, parameters=None)

IndexedOptionArray is an [`ak.contents.IndexedArray`](sphinx-llm:0ec798a3207a4746a193b3e535c37f41#ak.contents.IndexedArray) for which
negative values in the index are interpreted as missing. It represents
[`ak.types.OptionType`](sphinx-llm:f8e0eb36e1334eef887fb5ab81458600#ak.types.OptionType) data like [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray),
[`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray), and [`ak.contents.UnmaskedArray`](sphinx-llm:c65ee37a79b9404bbe625ef99e63bade#ak.contents.UnmaskedArray), but
the flexibility of the arbitrary `index` makes it a common output of
many operations.

IndexedOptionArray doesn’t have a direct equivalent in Apache Arrow.

To illustrate how the constructor arguments are interpreted, the following is a
simplified implementation of `__init__`, `__len__`, and `__getitem__`:

```default
class IndexedOptionArray(Content):
    def __init__(self, index, content):
        assert isinstance(index, (Index32, Index64))
        assert isinstance(content, Content)
        for x in index:
            assert x < len(content)  # index[i] may be negative
        self.index = index
        self.content = content

    def __len__(self):
        return len(self.index)

    def __getitem__(self, where):
        if isinstance(where, int):
            if where < 0:
                where += len(self)
            assert 0 <= where < len(self)
            if self.index[where] < 0:
                return None
            else:
                return self.content[self.index[where]]

        elif isinstance(where, slice) and where.step is None:
            return IndexedOptionArray(
                self.index[where.start : where.stop], self.content
            )

        elif isinstance(where, str):
            return IndexedOptionArray(self.index, self.content[where])

        else:
            raise AssertionError(where)
```

#### \_index

#### \_content

#### *property* index

#### form_cls *: awkward._typing.Final*

#### copy(index=UNSET, content=UNSET, \*, parameters=UNSET)

#### \_\_copy_\_()

#### \_\_deepcopy_\_(memo)

#### *classmethod* simplified(index, content, \*, parameters=None)

#### \_form_with_key(getkey: awkward._typing.Callable[[awkward.contents.content.Content], [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)]) → awkward.forms.indexedoptionform.IndexedOptionForm

#### \_form_with_key_path(path: awkward.forms.form.FormKeyPathT) → awkward.forms.indexedoptionform.IndexedOptionForm

#### \_to_buffers(form: awkward.forms.form.Form, getkey: awkward._typing.Callable[[awkward.contents.content.Content, awkward.forms.form.Form, [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], container: [collections.abc.MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._nplikes.array_like.ArrayLike], backend: awkward._backends.backend.Backend, byteorder: [str](https://docs.python.org/3/library/stdtypes.html#str))

#### \_to_typetracer(forget_length: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self

#### \_touch_data(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### \_touch_shape(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### *property* length *: awkward._nplikes.shape.ShapeItem*

#### \_\_repr_\_()

#### \_repr(indent, pre, post)

#### to_IndexedOptionArray64() → [IndexedOptionArray](sphinx-llm:ab616f5550ca4a6f8d393549fa61bd8f)

#### to_ByteMaskedArray(valid_when)

#### to_BitMaskedArray(valid_when, lsb_order)

#### mask_as_bool(valid_when: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → awkward._nplikes.array_like.ArrayLike

#### \_getitem_nothing()

#### \_is_getitem_at_placeholder() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_is_getitem_at_virtual() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_getitem_at(where: awkward._nplikes.numpy_like.IndexType)

#### \_getitem_range(start: awkward._nplikes.numpy_like.IndexType, stop: awkward._nplikes.numpy_like.IndexType) → awkward.contents.content.Content

#### \_getitem_field(where: [str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._typing.SupportsIndex, only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → awkward.contents.content.Content

#### \_getitem_fields(where: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._typing.SupportsIndex], only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → awkward.contents.content.Content

#### \_carry(carry: awkward.index.Index, allow_lazy: [bool](https://docs.python.org/3/library/functions.html#bool)) → [IndexedOptionArray](sphinx-llm:ab616f5550ca4a6f8d393549fa61bd8f)

#### \_nextcarry_outindex()

#### \_getitem_next_jagged_generic(slicestarts, slicestops, slicecontent, tail)

#### \_getitem_next_jagged(slicestarts: awkward.index.Index, slicestops: awkward.index.Index, slicecontent: awkward.contents.content.Content, tail) → awkward.contents.content.Content

#### \_getitem_next(head: awkward._slicing.SliceItem | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple), tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None)) → awkward.contents.content.Content

#### project(mask=None)

#### \_offsets_and_flattened(axis: [int](https://docs.python.org/3/library/functions.html#int), depth: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward.index.Index, awkward.contents.content.Content]

#### \_mergeable_next(other: awkward.contents.content.Content, mergebool: [bool](https://docs.python.org/3/library/functions.html#bool), mergecastable: awkward._typing.Literal[same_kind, equiv, family]) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_merging_strategy(others)

#### \_reverse_merge(other)

#### \_mergemany(others: [collections.abc.Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[awkward.contents.content.Content]) → awkward.contents.content.Content

#### \_fill_none(value: awkward.contents.content.Content) → awkward.contents.content.Content

#### \_local_index(axis, depth)

#### \_is_subrange_equal(starts, stops, length, sorted=True)

#### \_numbers_to_type(name, including_unknown)

#### \_is_unique(negaxis, starts, offsets, outlength)

#### \_unique(negaxis, starts, offsets, outlength)

#### \_rearrange_nextshifts(next_length, shifts)

#### \_rearrange_prepare_next(offsets, outlength)

#### \_argsort_next(negaxis, starts, shifts, offsets, outlength, ascending, stable)

#### \_sort_next(negaxis, starts, offsets, outlength, ascending, stable)

#### \_reduce_next(reducer, negaxis, starts, shifts, offsets, outlength, mask, keepdims, behavior)

#### \_combinations(n, replacement, recordlookup, parameters, axis, depth)

#### \_validity_error(path)

#### \_nbytes_part()

#### \_pad_none(target, axis, depth, clip)

#### \_to_arrow(pyarrow: awkward._typing.Any, mask_node: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), validbytes: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int), options: awkward.contents.content.ToArrowOptions)

#### \_to_cudf(cudf: awkward._typing.Any, mask: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int))

#### \_to_backend_array(allow_missing, backend)

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

#### \_trim() → awkward._typing.Self
