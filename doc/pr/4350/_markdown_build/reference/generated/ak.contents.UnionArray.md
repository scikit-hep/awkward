# ak.contents.UnionArray

Defined in [awkward.contents.unionarray](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/unionarray.py) on [line 58](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/unionarray.py#L58).

#### *class* ak.contents.UnionArray(tags, index, contents, \*, parameters=None)

UnionArray represents data drawn from an ordered list of `contents`,
which can have different types, using

* `tags`: buffer of integers indicating which content each array element draws from.
* `index`: buffer of integers indicating which element from the content to draw from.

UnionArrays correspond to Apache Arrow’s
[dense union type](https://arrow.apache.org/docs/format/Columnar.html#dense-union).
Awkward Array has no direct equivalent for Apache Arrow’s
[sparse union type](https://arrow.apache.org/docs/format/Columnar.html#sparse-union).

To illustrate how the constructor arguments are interpreted, the following is a
simplified implementation of `__init__`, `__len__`, and `__getitem__`:

```default
class UnionArray(Content):
    def __init__(self, tags, index, contents):
        assert isinstance(tags, Index8)
        assert isinstance(index, (Index32, IndexU32, Index64))
        assert isinstance(contents, list)
        assert len(index) >= len(tags)  # usually equal
        for x in tags:
            assert 0 <= x < len(contents)
        for i, x in enumerate(tags):
            assert 0 <= index[i] < len(contents[x])
        self.tags = tags
        self.index = index
        self.contents = contents

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, where):
        if isinstance(where, int):
            if where < 0:
                where += len(self)
            assert 0 <= where < len(self)
            return self.contents[self.tags[where]][self.index[where]]

        elif isinstance(where, slice) and where.step is None:
            return UnionArray(
                self.tags[where], self.index[where], self.contents
            )

        elif isinstance(where, str):
            return UnionArray(
                self.tags, self.index, [x[where] for x in self.contents]
            )

        else:
            raise AssertionError(where)
```

#### \_tags

#### \_index

#### \_contents

#### *property* tags

#### *property* index

#### form_cls *: awkward._typing.Final*

#### copy(tags=UNSET, index=UNSET, contents=UNSET, \*, parameters=UNSET)

#### \_\_copy_\_()

#### \_\_deepcopy_\_(memo)

#### *classmethod* simplified(tags, index, contents, \*, parameters=None, mergebool=False, mergecastable='same_kind', dropunused=False)

#### \_form_with_key(getkey: awkward._typing.Callable[[awkward.contents.content.Content], [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)]) → awkward.forms.unionform.UnionForm

#### \_form_with_key_path(path: awkward.forms.form.FormKeyPathT) → awkward.forms.unionform.UnionForm

#### \_to_buffers(form: awkward.forms.form.Form, getkey: awkward._typing.Callable[[awkward.contents.content.Content, awkward.forms.form.Form, [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], container: [collections.abc.MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._nplikes.array_like.ArrayLike], backend: awkward._backends.backend.Backend, byteorder: [str](https://docs.python.org/3/library/stdtypes.html#str))

#### \_to_typetracer(forget_length: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self

#### \_touch_data(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### \_touch_shape(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### *property* length *: awkward._nplikes.shape.ShapeItem*

#### \_\_repr_\_()

#### \_repr(indent, pre, post)

#### \_getitem_nothing()

#### \_is_getitem_at_placeholder() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_is_getitem_at_virtual() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_getitem_at(where: awkward._nplikes.numpy_like.IndexType)

#### \_getitem_range(start: awkward._nplikes.numpy_like.IndexType, stop: awkward._nplikes.numpy_like.IndexType) → awkward.contents.content.Content

#### \_getitem_field(where: [str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._typing.SupportsIndex, only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → awkward.contents.content.Content

#### \_getitem_fields(where: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._typing.SupportsIndex], only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → awkward.contents.content.Content

#### \_carry(carry: awkward.index.Index, allow_lazy: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward.contents.content.Content

#### \_union_of_optionarrays(index, parameters)

#### project(index)

#### *static* regular_index(tags: awkward.index.Index, \*, backend: awkward._backends.backend.Backend, index_cls: [type](https://docs.python.org/3/library/functions.html#type)[awkward.index.Index] = Index64)

#### \_regular_index(tags: awkward.index.Index) → awkward.index.Index

#### *static* nested_tags_index(offsets: awkward.index.Index, counts: [collections.abc.Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[awkward.index.Index], \*, backend: awkward._backends.backend.Backend, tags_cls: [type](https://docs.python.org/3/library/functions.html#type)[awkward.index.Index] = Index8, index_cls: [type](https://docs.python.org/3/library/functions.html#type)[awkward.index.Index] = Index64) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward.index.Index, awkward.index.Index]

#### \_getitem_next_jagged_generic(slicestarts, slicestops, slicecontent, tail)

#### \_getitem_next_jagged(slicestarts: awkward.index.Index, slicestops: awkward.index.Index, slicecontent: awkward.contents.content.Content, tail) → awkward.contents.content.Content

#### \_getitem_next(head: awkward._slicing.SliceItem | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple), tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None)) → awkward.contents.content.Content

#### \_offsets_and_flattened(axis: [int](https://docs.python.org/3/library/functions.html#int), depth: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward.index.Index, awkward.contents.content.Content]

#### \_mergeable_next(other: awkward.contents.content.Content, mergebool: [bool](https://docs.python.org/3/library/functions.html#bool), mergecastable: awkward._typing.Literal[same_kind, equiv, family]) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_merging_strategy(others)

#### \_reverse_merge(other)

#### \_mergemany(others: [collections.abc.Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[awkward.contents.content.Content]) → awkward.contents.content.Content

#### \_fill_none(value: awkward.contents.content.Content) → awkward.contents.content.Content

#### \_local_index(axis, depth)

#### \_combinations(n, replacement, recordlookup, parameters, axis, depth)

#### \_numbers_to_type(name, including_unknown)

#### \_is_unique(negaxis, starts, offsets, outlength)

#### \_unique(negaxis, starts, offsets, outlength)

#### \_argsort_next(negaxis, starts, shifts, offsets, outlength, ascending, stable)

#### \_sort_next(negaxis, starts, offsets, outlength, ascending, stable)

#### \_reduce_next(reducer, negaxis, starts, shifts, offsets, outlength, mask, keepdims, behavior)

#### \_validity_error(path)

#### \_nbytes_part()

#### \_pad_none(target, axis, depth, clip)

#### \_to_arrow(pyarrow: awkward._typing.Any, mask_node: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), validbytes: awkward.contents.content.Content | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int), options: awkward.contents.content.ToArrowOptions)

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
