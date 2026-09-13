# ak.types.NumpyType

Defined in [awkward.types.numpytype](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/numpytype.py) on [line 100](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/numpytype.py#L100).

#### *class* ak.types.NumpyType(primitive: [str](https://docs.python.org/3/library/stdtypes.html#str), \*, parameters: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None) = None)

#### copy(\*, primitive: [str](https://docs.python.org/3/library/stdtypes.html#str) | awkward._util.Sentinel = UNSET, parameters: awkward._typing.JSONMapping | awkward._util.Sentinel | [None](https://docs.python.org/3/library/constants.html#None) = UNSET) → [NumpyType](sphinx-llm:b256c7aad302449bbadc5fc48f57b919)

#### \_primitive *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### \_parameters *: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### *property* primitive *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### \_str_parameters_exclude *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...]* *= ('_\_categorical_\_', '_\_unit_\_')*

#### \_get_typestr(behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None)) → [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)

#### \_str(indent: [str](https://docs.python.org/3/library/stdtypes.html#str), compact: [bool](https://docs.python.org/3/library/functions.html#bool), behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### \_\_repr_\_() → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
