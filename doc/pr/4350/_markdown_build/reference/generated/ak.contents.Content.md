# ak.contents.Content

Defined in [awkward.contents.content](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/content.py) on [line 136](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/contents/content.py#L136).

#### *class* ak.contents.Content

#### \_init(parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), backend: awkward._backends.backend.Backend)

#### *property* backend *: awkward._backends.backend.Backend*

#### *property* form *: awkward.forms.form.Form*

#### form_with_key(form_key: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) | [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) = 'node{id}', id_start: [int](https://docs.python.org/3/library/functions.html#int) = 0) → awkward.forms.form.Form

#### *abstract* \_form_with_key(getkey: [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)], [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)]) → awkward.forms.form.Form

#### form_with_key_path(root: awkward.forms.form.FormKeyPathT = ()) → awkward.forms.form.Form

#### *abstract* \_form_with_key_path(path: awkward.forms.form.FormKeyPathT) → awkward.forms.form.Form

#### *property* form_cls *: [type](https://docs.python.org/3/library/functions.html#type)[awkward.forms.form.Form]*

* **Abstractmethod:**

#### to_typetracer(forget_length: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → awkward._typing.Self

#### *abstract* \_to_typetracer(forget_length: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self

#### *abstract* \_touch_data(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### *abstract* \_touch_shape(recursive: [bool](https://docs.python.org/3/library/functions.html#bool))

#### *property* length *: awkward._nplikes.shape.ShapeItem*

* **Abstractmethod:**

#### *abstract* \_to_buffers(form: awkward.forms.form.Form, getkey: collections.abc.Callable[[Content, awkward.forms.form.Form, str], str], container: collections.abc.MutableMapping[str, awkward._typing.Any] | None, backend: awkward._backends.backend.Backend, byteorder: awkward._typing.Literal[<, >]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward.forms.form.Form, [int](https://docs.python.org/3/library/functions.html#int), [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any]]

#### \_\_len_\_() → [int](https://docs.python.org/3/library/functions.html#int)

#### \_repr_extra(indent: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### maybe_to_NumpyArray()

#### \_maybe_index_error(error: awkward._kernels.KernelError | [None](https://docs.python.org/3/library/constants.html#None), slicer)

#### \_\_array_ufunc_\_(ufunc, method, \*inputs, \*\*kwargs)

#### \_\_array_function_\_(func, types, args, kwargs)

#### \_\_array_\_(dtype=None, copy=None)

#### \_\_iter_\_()

#### \_getitem_next_field(head: awkward._slicing.SliceItem | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple), tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None))

#### \_getitem_next_fields(head: awkward._slicing.SliceItem, tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None)) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### \_getitem_next_newaxis(tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None)) → awkward.contents.regulararray.RegularArray

#### \_getitem_next_ellipsis(tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None)) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### \_getitem_next_regular_missing(head: awkward.contents.indexedoptionarray.IndexedOptionArray, tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None), raw: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e), length: [int](https://docs.python.org/3/library/functions.html#int)) → awkward.contents.regulararray.RegularArray

#### \_getitem_next_missing_jagged(head: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e), tail, advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None), that: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)) → awkward.contents.regulararray.RegularArray

#### \_getitem_next_missing(head: awkward.contents.indexedoptionarray.IndexedOptionArray, tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None)) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### \_\_getitem_\_(where)

#### \_getitem(where, named_axis: awkward._typing.Type[awkward._namedaxis.NamedAxis] = NamedAxis)

#### *abstract* \_is_getitem_at_placeholder() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### *abstract* \_is_getitem_at_virtual() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### *abstract* \_getitem_at(where: awkward._nplikes.numpy_like.IndexType)

#### *abstract* \_getitem_range(start: awkward._nplikes.numpy_like.IndexType, stop: awkward._nplikes.numpy_like.IndexType) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* \_getitem_field(where: [str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._typing.SupportsIndex, only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* \_getitem_fields(where: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* \_getitem_next(head: awkward._slicing.SliceItem | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple), tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...], advanced: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None)) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* \_getitem_next_jagged(slicestarts: awkward.index.Index, slicestops: awkward.index.Index, slicecontent: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e), tail: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward._slicing.SliceItem, ...]) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* \_carry(carry: awkward.index.Index, allow_lazy: [bool](https://docs.python.org/3/library/functions.html#bool)) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### \_local_index_axis0() → awkward.contents.numpyarray.NumpyArray

#### *abstract* \_mergeable_next(other: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e), mergebool: [bool](https://docs.python.org/3/library/functions.html#bool), mergecastable: awkward._typing.Literal[same_kind, equiv, family]) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### *abstract* \_mergemany(others: [collections.abc.Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)]) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### \_merging_strategy(others: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)], [list](https://docs.python.org/3/library/stdtypes.html#list)[[Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)]]

#### *abstract* \_local_index(axis: [int](https://docs.python.org/3/library/functions.html#int), depth: [int](https://docs.python.org/3/library/functions.html#int))

#### *abstract* \_reduce_next(reducer: awkward._reducers.Reducer, negaxis: [int](https://docs.python.org/3/library/functions.html#int), starts: awkward.index.Index, shifts: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None), offsets: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None), outlength: [int](https://docs.python.org/3/library/functions.html#int), mask: [bool](https://docs.python.org/3/library/functions.html#bool), keepdims: [bool](https://docs.python.org/3/library/functions.html#bool), behavior: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None))

#### *abstract* \_argsort_next(negaxis: [int](https://docs.python.org/3/library/functions.html#int), starts: awkward.index.Index, shifts: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None), offsets: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None), outlength: [int](https://docs.python.org/3/library/functions.html#int), ascending: [bool](https://docs.python.org/3/library/functions.html#bool), stable: [bool](https://docs.python.org/3/library/functions.html#bool))

#### *abstract* \_sort_next(negaxis: [int](https://docs.python.org/3/library/functions.html#int), starts: awkward.index.Index, offsets: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None), outlength: [int](https://docs.python.org/3/library/functions.html#int), ascending: [bool](https://docs.python.org/3/library/functions.html#bool), stable: [bool](https://docs.python.org/3/library/functions.html#bool))

#### \_combinations_axis0(n: [int](https://docs.python.org/3/library/functions.html#int), replacement: [bool](https://docs.python.org/3/library/functions.html#bool), recordlookup: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None), parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None))

#### *abstract* \_combinations(n: [int](https://docs.python.org/3/library/functions.html#int), replacement: [bool](https://docs.python.org/3/library/functions.html#bool), recordlookup: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None), parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), axis: [int](https://docs.python.org/3/library/functions.html#int), depth: [int](https://docs.python.org/3/library/functions.html#int))

#### *abstract* \_validity_error(path: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### *property* nbytes *: [int](https://docs.python.org/3/library/functions.html#int)*

#### purelist_parameter(key: [str](https://docs.python.org/3/library/stdtypes.html#str))

Return the value of the outermost parameter matching `key` in a sequence
of nested lists, stopping at the first record or tuple layer.

If a layer has [`ak.types.UnionType`](sphinx-llm:5e1485b00d7b47f39f0082bb755fa92f#ak.types.UnionType), the value is only returned if all
possibilities have the same value.

#### purelist_parameters(\*keys: [str](https://docs.python.org/3/library/stdtypes.html#str))

Return the value of the outermost parameter matching one of `keys` in a sequence
of nested lists, stopping at the first record or tuple layer.

If a layer has [`ak.types.UnionType`](sphinx-llm:5e1485b00d7b47f39f0082bb755fa92f#ak.types.UnionType), the value is only returned if all
possibilities have the same value.

#### *abstract* \_is_unique(negaxis: awkward._typing.AxisMaybeNone, starts: awkward.index.Index, offsets: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None), outlength: [int](https://docs.python.org/3/library/functions.html#int)) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### *abstract* \_unique(negaxis: awkward._typing.AxisMaybeNone, starts: awkward.index.Index, offsets: awkward.index.Index | [None](https://docs.python.org/3/library/constants.html#None), outlength: [int](https://docs.python.org/3/library/functions.html#int))

#### \_pad_none_axis0(target: [int](https://docs.python.org/3/library/functions.html#int), clip: [bool](https://docs.python.org/3/library/functions.html#bool)) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* \_pad_none(target: [int](https://docs.python.org/3/library/functions.html#int), axis: [int](https://docs.python.org/3/library/functions.html#int), depth: [int](https://docs.python.org/3/library/functions.html#int), clip: [bool](https://docs.python.org/3/library/functions.html#bool)) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### to_arrow(list_to32: [bool](https://docs.python.org/3/library/functions.html#bool) = False, string_to32: [bool](https://docs.python.org/3/library/functions.html#bool) = False, bytestring_to32: [bool](https://docs.python.org/3/library/functions.html#bool) = False, emptyarray_to=None, categorical_as_dictionary: [bool](https://docs.python.org/3/library/functions.html#bool) = False, extensionarray: [bool](https://docs.python.org/3/library/functions.html#bool) = True, count_nulls: [bool](https://docs.python.org/3/library/functions.html#bool) = True, record_is_scalar: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

#### *abstract* \_to_arrow(pyarrow: awkward._typing.Any, mask_node: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e) | [None](https://docs.python.org/3/library/constants.html#None), validbytes: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e) | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int), options: ToArrowOptions)

#### *abstract* \_to_cudf(cudf: awkward._typing.Any, mask: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e) | [None](https://docs.python.org/3/library/constants.html#None), length: [int](https://docs.python.org/3/library/functions.html#int))

#### to_backend_array(allow_missing: [bool](https://docs.python.org/3/library/functions.html#bool) = True, \*, backend: awkward._backends.backend.Backend | [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

#### *abstract* \_to_backend_array(allow_missing: [bool](https://docs.python.org/3/library/functions.html#bool), backend: awkward._backends.backend.Backend)

#### drop_none()

#### *abstract* \_drop_none() → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* \_remove_structure(backend: awkward._backends.backend.Backend, options: RemoveStructureOptions) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)]

#### *abstract* \_recursively_apply(action: ImplementsApplyAction, depth: [int](https://docs.python.org/3/library/functions.html#int), depth_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), lateral_context: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None), options: ApplyActionOptions) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e) | [None](https://docs.python.org/3/library/constants.html#None)

#### to_json(nan_string: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, posinf_string: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, neginf_string: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, complex_record_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, convert_bytes: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None, behavior: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)

#### to_packed(recursive: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* \_to_packed(recursive: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### to_list(behavior: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)

#### *abstract* \_to_list(behavior: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None), json_conversions: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)

#### \_to_list_custom(behavior: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None), json_conversions: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None))

#### *abstract* \_offsets_and_flattened(axis: [int](https://docs.python.org/3/library/functions.html#int), depth: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward.index.Index, [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)]

#### to_backend(backend: awkward._backends.backend.Backend | [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → awkward._typing.Self

#### *abstract* \_to_backend(backend: awkward._backends.backend.Backend) → awkward._typing.Self

#### materialize(type_: [type](https://docs.python.org/3/library/functions.html#type) = MaterializableArray) → awkward._typing.Self

#### *abstract* \_materialize(type_) → awkward._typing.Self

#### *property* is_all_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* \_is_all_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

* **Abstractmethod:**

#### *property* is_any_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* \_is_any_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

* **Abstractmethod:**

#### with_parameter(key: [str](https://docs.python.org/3/library/stdtypes.html#str), value: awkward._typing.Any) → awkward._typing.Self

#### *abstract* \_\_copy_\_()

#### *abstract* \_\_deepcopy_\_(memo)

#### is_equal_to(other: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e), index_dtype: [bool](https://docs.python.org/3/library/functions.html#bool) = True, numpyarray: [bool](https://docs.python.org/3/library/functions.html#bool) = True, \*, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_is_equal_to_generic(other: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e), all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### *abstract* \_is_equal_to(other: awkward._typing.Self, index_dtype: [bool](https://docs.python.org/3/library/functions.html#bool), numpyarray: [bool](https://docs.python.org/3/library/functions.html#bool), all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### *abstract* \_repr(indent: [str](https://docs.python.org/3/library/stdtypes.html#str), pre: [str](https://docs.python.org/3/library/stdtypes.html#str), post: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### *abstract* \_numbers_to_type(name: [str](https://docs.python.org/3/library/stdtypes.html#str), including_unknown: [bool](https://docs.python.org/3/library/functions.html#bool)) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* \_fill_none(value: [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)) → [Content](sphinx-llm:0d9b5db1a6a84f80aa1c58cc98ed1f0e)

#### *abstract* copy(\*, parameters: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None) = UNSET) → awkward._typing.Self

#### *classmethod* \_arrow_needs_option_type()
