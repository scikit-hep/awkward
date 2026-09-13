# ak.forms.Form

Defined in [awkward.forms.form](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/forms/form.py) on [line 384](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/forms/form.py#L384).

#### *class* ak.forms.Form

#### \_init(\*, parameters: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None), form_key: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None))

#### *property* form_key

#### \_\_str_\_()

#### to_dict(verbose=True)

#### \_to_dict_extra(out, verbose)

#### to_json()

#### \_repr_args()

#### *abstract property* type

#### columns(list_indicator=None, column_prefix=())

#### select_columns(specifier, expand_braces=True, \*, prune_unions_and_records: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

select_columns returns a new Form with only columns and sub-columns selected.
Returns an empty Form if no columns matched the specifier(s).

`specifier` can be a `str | Iterable[str | Iterable[str]]`.
Strings may include shell-globbing-style wildcards “\*” and “?”.
If `expand_braces` is `True` (the default), strings may include alternatives in braces.
For example, `["a.{b,c}.d"]` is equivalent to `["a.b.d", "a.c.d"]`.
Glob-style matching would also suit this single-character instance: `"a.[bc].d"`.
If specifier is a list which contains a list/tuple, that inner list will be interpreted as
column and subcolumn specifiers. They *may* contain wildcards, but “.” will not be
interpreted as a `<field>.<subfield>` pattern.

#### column_types()

#### *abstract* \_columns(path, output, list_indicator)

#### *abstract* \_prune_columns(is_inside_record_or_union: [bool](https://docs.python.org/3/library/functions.html#bool)) → [Form](sphinx-llm:48e291baf75c488f81567a069e30ff1e) | [None](https://docs.python.org/3/library/constants.html#None)

#### *abstract* \_select_columns(match_specifier: \_SpecifierMatcher) → [Form](sphinx-llm:48e291baf75c488f81567a069e30ff1e) | [None](https://docs.python.org/3/library/constants.html#None)

#### *abstract* \_column_types()

#### *abstract* \_to_dict_part(verbose, toplevel)

#### length_zero_array(\*, backend=numpy_backend, highlevel=False, behavior=None)

#### length_one_array(\*, backend=numpy_backend, highlevel=False, behavior=None)

#### *abstract* \_expected_from_buffers(getkey: [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[Form](sphinx-llm:48e291baf75c488f81567a069e30ff1e), [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], recursive: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Iterator[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.DType]]

#### expected_from_buffers(buffer_key='{form_key}-{attribute}', recursive=True)

* **Parameters:**
  * **buffer_key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *callable*) – Python format string containing
    `"{form_key}"` and/or `"{attribute}"` or a function that takes these
    as keyword arguments and returns a string to use as a key for a buffer
    in the `container`.
  * **recursive** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, recurse into subforms; otherwise, yield
    only the (buffer_key, dtype) pairs for this form object.

Yield (buffer_key, dtype) pairs describing the expected buffer keys,
and their corresponding dtypes, that a call to [`ak.from_buffers`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers) would
be expected to find from the `container` object.

#### is_equal_to(other: awkward._typing.Any, \*, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool) = False, form_key: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_\_eq_\_

#### *abstract* \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool), form_key: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_is_equal_to_generic(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool), form_key: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
