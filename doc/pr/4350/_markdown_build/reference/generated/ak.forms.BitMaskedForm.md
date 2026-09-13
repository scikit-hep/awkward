# ak.forms.BitMaskedForm

Defined in [awkward.forms.bitmaskedform](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/forms/bitmaskedform.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/forms/bitmaskedform.py#L19).

#### *class* ak.forms.BitMaskedForm(mask, content, valid_when, lsb_order, \*, parameters=None, form_key=None)

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

#### \_mask

#### \_valid_when

#### \_lsb_order

#### *property* mask

#### *property* content

#### *property* valid_when

#### *property* lsb_order

#### copy(mask=UNSET, content=UNSET, valid_when=UNSET, lsb_order=UNSET, \*, parameters=UNSET, form_key=UNSET)

#### *classmethod* simplified(mask, content, valid_when, lsb_order, \*, parameters=None, form_key=None)

#### \_\_repr_\_()

#### \_to_dict_part(verbose, toplevel)

#### *property* type

#### \_columns(path, output, list_indicator)

#### \_prune_columns(is_inside_record_or_union: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Self | [None](https://docs.python.org/3/library/constants.html#None)

#### \_select_columns(match_specifier: awkward.forms.form._SpecifierMatcher) → awkward._typing.Self

#### \_column_types()

#### \_\_setstate_\_(state)

#### \_expected_from_buffers(getkey: [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[awkward.forms.form.Form, [str](https://docs.python.org/3/library/stdtypes.html#str)], [str](https://docs.python.org/3/library/stdtypes.html#str)], recursive: [bool](https://docs.python.org/3/library/functions.html#bool)) → awkward._typing.Iterator[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.DType]]

#### \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool), form_key: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
