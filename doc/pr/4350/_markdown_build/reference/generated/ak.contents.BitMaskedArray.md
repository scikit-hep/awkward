# ak.contents.BitMaskedArray

Defined in [awkward.contents.bitmaskedarray](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/bitmaskedarray.py) on [line 55](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/bitmaskedarray.py#L55).

#### *class* ak.contents.BitMaskedArray(mask, content, valid_when, length, lsb_order, length_generator=None, \*, parameters=None)

Like [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray), BitMaskedArray implements an
[`ak.types.OptionType`](sphinx-llm:f8e0eb36e1334eef887fb5ab81458600#ak.types.OptionType) with two buffers, `mask` and `content`.
However, the boolean `mask` values are packed into a bitmap.

BitMaskedArray has an additional parameter, `lsb_order`; if True,
the position of each bit is in
[Least-Significant Bit order](https://en.wikipedia.org/wiki/Bit_numbering)
(LSB):

```default
is_valid[j] = bool(mask[j // 8] & (1 << (j % 8))) == valid_when
```

If False, the position of each bit is in Most-Significant Bit order
(MSB):

```default
is_valid[j] = bool(mask[j // 8] & (128 >> (j % 8))) == valid_when
```

If the logical size of the buffer is not a multiple of 8, the `mask`
has to be padded. Thus, an explicit `length` is also part of the
class’s definition.

This is equivalent to *all* of Apache Arrow’s array types because they all
[use bitmaps](https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps)
to mask their data, with `valid_when=True` and `lsb_order=True`.

To illustrate how the constructor arguments are interpreted, the following is a
simplified implementation of `__init__`, `__len__`, and `__getitem__`:

```default
class BitMaskedArray(Content):
    def __init__(self, mask, content, valid_when, length, lsb_order):
        assert isinstance(mask, IndexU8)
        assert isinstance(content, Content)
        assert isinstance(valid_when, bool)
        assert isinstance(length, int) and length >= 0
        assert isinstance(lsb_order, bool)
        assert len(mask) <= len(content)
        self.mask = mask
        self.content = content
        self.valid_when = valid_when
        self.length = length
        self.lsb_order = lsb_order

    def __len__(self):
        return self.length

    def __getitem__(self, where):
        if isinstance(where, int):
            if where < 0:
                where += len(self)
            assert 0 <= where < len(self)
            if self.lsb_order:
                bit = bool(self.mask[where // 8] & (1 << (where % 8)))
            else:
                bit = bool(self.mask[where // 8] & (128 >> (where % 8)))
            if bit == self.valid_when:
                return self.content[where]
            else:
                return None

        elif isinstance(where, slice) and where.step is None:
            # In general, slices must convert BitMaskedArray to ByteMaskedArray.
            bytemask = np.unpackbits(
                self.mask, bitorder=("little" if self.lsb_order else "big")
            ).view(bool)
            return ByteMaskedArray(
                bytemask[where.start : where.stop],
                self.content[where.start : where.stop],
                valid_when=self.valid_when,
            )

        elif isinstance(where, str):
            return BitMaskedArray(
                self.mask,
                self.content[where],
                valid_when=self.valid_when,
                length=self.length,
                lsb_order=self.lsb_order,
            )

        else:
            raise AssertionError(where)
```

#### \_mask

#### \_content

#### \_valid_when

#### \_length

#### \_length_generator *= None*

#### \_lsb_order

#### *property* mask

#### *property* valid_when

#### *property* lsb_order

#### form_cls *: awkward._typing.Final*

#### copy(mask=UNSET, content=UNSET, valid_when=UNSET, length=UNSET, lsb_order=UNSET, length_generator=UNSET, \*, parameters=UNSET)

#### \_\_copy_\_()

#### \_\_deepcopy_\_(memo)

#### \_\_getstate_\_()

#### *classmethod* simplified(mask, content, valid_when, length, lsb_order, length_generator=None, \*, parameters=None)

#### \_form_with_key(getkey: awkward._typing.Callable[[awkward.contents.content.Content], [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)]) → awkward.forms.bitmaskedform.BitMaskedForm

#### \_form_with_key_path(path: awkward.forms.form.FormKeyPathT) → awkward.forms.bitmaskedform.BitMaskedForm

#### \_to_buffers(form: awkward.forms.form.Form, getkey: awkward._typing.Callable[[awkward.contents.content.Content, awkward.forms.form.Form, [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], container: [collections.abc.MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._nplikes.array_like.ArrayLike], backend: awkward._backends.backend.Backend, byteorder: [str](https://docs.python.org/3/library/stdtypes.html#str))

#### \_to_typetracer(forget_length: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self

#### \_touch_data(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### \_touch_shape(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### *property* length *: awkward._nplikes.shape.ShapeItem*

#### \_\_repr_\_()

#### \_repr(indent, pre, post)

#### to_IndexedOptionArray64() → awkward.contents.IndexedOptionArray

#### to_ByteMaskedArray()

#### to_BitMaskedArray(valid_when, lsb_order)

#### mask_as_bool(valid_when=None)

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

#### project(mask=None)

#### \_offsets_and_flattened(axis: [int](https://docs.python.org/3/library/functions.html#int), depth: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward.index.Index, awkward.contents.content.Content]

#### \_mergeable_next(other: awkward.contents.content.Content, mergebool: [bool](https://docs.python.org/3/library/functions.html#bool), mergecastable: awkward._typing.Literal[same_kind, equiv, family]) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_reverse_merge(other)

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

#### \_recursively_apply(action: awkward.contents.content.ImplementsApplyAction, depth: [int](https://docs.python.org/3/library/functions.html#int), depth_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), lateral_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), options: awkward.contents.content.ApplyActionOptions) → awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None)

#### \_to_packed(recursive: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → awkward._typing.Self

#### \_to_list(behavior, json_conversions)

#### \_to_backend(backend: awkward._backends.backend.Backend) → awkward._typing.Self

#### \_materialize(type_) → awkward._typing.Self

#### *property* \_is_all_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* \_is_any_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### \_is_equal_to(other: awkward._typing.Self, index_dtype: [bool](https://docs.python.org/3/library/functions.html#bool), numpyarray: [bool](https://docs.python.org/3/library/functions.html#bool), all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
