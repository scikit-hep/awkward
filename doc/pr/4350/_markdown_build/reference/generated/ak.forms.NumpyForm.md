# ak.forms.NumpyForm

Defined in [awkward.forms.numpyform](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/forms/numpyform.py) on [line 48](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/forms/numpyform.py#L48).

#### *class* ak.forms.NumpyForm(primitive, inner_shape=(), \*, parameters=None, form_key=None)

#### \_primitive

#### \_inner_shape *= ()*

#### *property* primitive

#### *property* inner_shape

#### copy(primitive=UNSET, inner_shape=UNSET, \*, parameters=UNSET, form_key=UNSET)

#### *classmethod* simplified(primitive, inner_shape=(), \*, parameters=None, form_key=None)

#### *property* itemsize

#### \_\_repr_\_()

#### \_to_dict_part(verbose, toplevel)

#### *property* type

#### to_RegularForm() → awkward.forms.regularform.RegularForm | [NumpyForm](sphinx-llm:115922a3e22b4681a54b5bd007e8b4c5)

#### \_columns(path, output, list_indicator)

#### \_select_columns(match_specifier: awkward.forms.form._SpecifierMatcher) → awkward._typing.Self

#### \_prune_columns(is_inside_record_or_union: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self

#### \_column_types()

#### \_\_setstate_\_(state)

#### \_expected_from_buffers(getkey: [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[awkward.forms.form.Form, [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], recursive: [bool](https://docs.python.org/3/library/functions.html#bool)) → [collections.abc.Iterator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.DType]]

#### \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool), form_key: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
