# ak.types.OptionType

Defined in [awkward.types.optiontype](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/optiontype.py) on [line 22](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/optiontype.py#L22).

#### *class* ak.types.OptionType(content: awkward.types.type.Type, \*, parameters: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None) = None)

#### copy(\*, content: awkward.types.type.Type | awkward._util.Sentinel = UNSET, parameters: awkward._typing.JSONMapping | awkward._util.Sentinel | [None](https://docs.python.org/3/library/constants.html#None) = UNSET) → [OptionType](sphinx-llm:679c16d93ad64a7f81e0b9102d128851)

#### \_content *: awkward.types.type.Type*

#### \_parameters *: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### *property* content *: awkward.types.type.Type*

#### \_str(indent: [str](https://docs.python.org/3/library/stdtypes.html#str), compact: [bool](https://docs.python.org/3/library/functions.html#bool), behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### \_\_repr_\_() → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### simplify_option_union() → awkward.types.type.Type

#### \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
