# ak.types.RegularType

Defined in [awkward.types.regulartype](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/regulartype.py) on [line 17](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/regulartype.py#L17).

#### *class* ak.types.RegularType(content: awkward.types.type.Type, size: awkward._nplikes.shape.ShapeItem, \*, parameters: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None) = None)

#### copy(\*, content: awkward.types.type.Type | awkward._util.Sentinel = UNSET, size: awkward._nplikes.shape.ShapeItem | awkward._util.Sentinel = UNSET, parameters: awkward._typing.JSONMapping | awkward._util.Sentinel | [None](https://docs.python.org/3/library/constants.html#None) = UNSET) → awkward._typing.Self

#### \_content *: awkward.types.type.Type*

#### \_size *: awkward._nplikes.shape.ShapeItem*

#### \_parameters *: awkward._typing.JSONMapping | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### *property* content *: awkward.types.type.Type*

#### *property* size *: awkward._nplikes.shape.ShapeItem*

#### \_get_typestr(behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None)) → [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)

#### \_str(indent: [str](https://docs.python.org/3/library/stdtypes.html#str), compact: [bool](https://docs.python.org/3/library/functions.html#bool), behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### \_\_repr_\_() → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### \_is_equal_to(other: awkward._typing.Any, all_parameters: [bool](https://docs.python.org/3/library/functions.html#bool)) → [bool](https://docs.python.org/3/library/functions.html#bool)
