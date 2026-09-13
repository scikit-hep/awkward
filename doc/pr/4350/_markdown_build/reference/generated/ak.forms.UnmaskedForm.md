# ak.forms.UnmaskedForm

Defined in [awkward.forms.unmaskedform](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/forms/unmaskedform.py) on [line 22](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/forms/unmaskedform.py#L22).

#### *class* ak.forms.UnmaskedForm(content, \*, parameters=None, form_key=None)

Abstract base class for generic types.

A generic type is typically declared by inheriting from
this class parameterized with one or more type variables.
For example, a generic mapping type might be defined as:

```default
class Mapping(Generic[KT, VT]):
    def __getitem__(self, key: KT) -> VT:
        ...
    # Etc.
```

This class can then be used as follows:

```default
def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
    try:
        return mapping[key]
    except KeyError:
        return default
```

#### \_content *: awkward.forms.form.Form*

#### *property* content

#### copy(content=UNSET, \*, parameters=UNSET, form_key=UNSET)

#### *classmethod* simplified(content, \*, parameters=None, form_key=None)

#### \_\_repr_\_()

#### \_to_dict_part(verbose, toplevel)

#### *property* type

#### \_columns(path, output, list_indicator)

#### \_prune_columns(is_inside_record_or_union: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self | [None](https://docs.python.org/3/library/constants.html#None)

#### \_select_columns(match_specifier: awkward.forms.form._SpecifierMatcher) → awkward._typing.Self

#### \_column_types()

#### \_\_setstate_\_(state)

#### \_expected_from_buffers(getkey: [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[awkward.forms.form.Form, [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], recursive: [bool](https://docs.python.org/3/library/functions.html#bool)) → [collections.abc.Iterator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.DType]]

#### \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool), form_key: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
