# ak.types.UnionType

Defined in [awkward.types.uniontype](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/uniontype.py) on [line 16](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/uniontype.py#L16).

#### *class* ak.types.UnionType(contents: [collections.abc.Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[awkward.types.type.Type], \*, parameters: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None) = None)

#### \_contents *: [list](https://docs.python.org/3/library/stdtypes.html#list)[awkward.types.type.Type]*

#### copy(\*, contents: [list](https://docs.python.org/3/library/stdtypes.html#list)[awkward.types.type.Type] | awkward._util.Sentinel = UNSET, parameters: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None) | awkward._util.Sentinel = UNSET) → awkward._typing.Self

#### \_parameters *= None*

#### *property* contents *: [list](https://docs.python.org/3/library/stdtypes.html#list)[awkward.types.type.Type]*

#### \_str(indent: [str](https://docs.python.org/3/library/stdtypes.html#str), compact: [bool](https://docs.python.org/3/library/functions.html#bool), behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### \_\_repr_\_() → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
