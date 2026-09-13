# ak.types.Type

Defined in [awkward.types.type](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/type.py) on [line 17](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/type.py#L17).

#### *class* ak.types.Type

#### \_parameters *: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None)*

#### *abstract* copy(\*, parameters: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None) | awkward._util.Sentinel = UNSET) → awkward._typing.Self

#### *property* parameters *: awkward._typing.JSONMapping*

#### parameter(key: [str](https://docs.python.org/3/library/stdtypes.html#str)) → awkward._typing.JSONSerializable | [None](https://docs.python.org/3/library/constants.html#None)

#### \_\_str_\_() → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### *abstract* \_str(indent: [str](https://docs.python.org/3/library/stdtypes.html#str), compact: [bool](https://docs.python.org/3/library/functions.html#bool), behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### show(stream=STDOUT)

#### \_str_parameters_exclude *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...]* *= ('_\_categorical_\_',)*

#### \_str_categorical_begin() → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### \_str_categorical_end() → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### \_str_parameters() → [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)

#### \_repr_args() → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### is_equal_to(other: awkward._typing.Any, \*, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [bool](https://docs.python.org/3/library/functions.html#bool)

#### \_\_eq_\_

#### *abstract* \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
