# ak.types.RecordType

Defined in [awkward.types.recordtype](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/recordtype.py) on [line 20](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/recordtype.py#L20).

#### *class* ak.types.RecordType(contents: [collections.abc.Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[awkward.types.type.Type], fields: [collections.abc.Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)], \*, parameters: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None) = None)

#### copy(\*, contents: [collections.abc.Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[awkward.types.type.Type] | awkward._util.Sentinel = UNSET, fields: [collections.abc.Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | awkward._util.Sentinel | [None](https://docs.python.org/3/library/constants.html#None) = UNSET, parameters: awkward._typing.JSONMapping | awkward._util.Sentinel | [None](https://docs.python.org/3/library/constants.html#None) = UNSET) → awkward._typing.Self

#### \_contents *: [list](https://docs.python.org/3/library/stdtypes.html#list)[awkward.types.type.Type]*

#### \_fields *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### \_parameters *: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### *property* contents *: [list](https://docs.python.org/3/library/stdtypes.html#list)[awkward.types.type.Type]*

#### *property* fields *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### *property* is_tuple *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### \_str_parameters_exclude *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...]* *= ('_\_categorical_\_', '_\_record_\_')*

#### \_str(indent: [str](https://docs.python.org/3/library/stdtypes.html#str), compact: [bool](https://docs.python.org/3/library/functions.html#bool), behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### \_\_repr_\_()

#### \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### index_to_field(index: [int](https://docs.python.org/3/library/functions.html#int)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### field_to_index(field: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [int](https://docs.python.org/3/library/functions.html#int)

#### has_field(field: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### content(index_or_field: [int](https://docs.python.org/3/library/functions.html#int) | [str](https://docs.python.org/3/library/stdtypes.html#str)) → awkward.types.type.Type
