# ak.record.Record

Defined in [awkward.record](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/record.py) on [line 25](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/record.py#L25).

#### *class* ak.record.Record(array, at)

Represents a single value from a [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray).

As this is a columnar representation, the Record contains a
`ak.layout.RecordArray`, rather than the other way around.
Its two fields are

* `array`: the `ak.layout.RecordArray` and
* `at`: the index posiion where this Record is found.

The Record shares a reference with its `ak.layout.RecordArray`;
it is not a copy.

#### *property* array

#### *property* backend

#### *property* at

#### *property* fields

#### *property* is_tuple

#### to_tuple() → awkward._typing.Self

#### *property* contents

#### content(index_or_field)

#### \_\_repr_\_()

#### \_repr(indent, pre, post)

#### validity_error(path='layout.array')

#### *property* parameters

#### parameter(key)

#### purelist_parameter(key) → awkward._typing.JSONSerializable

#### purelist_parameters(\*keys)

#### *property* purelist_isregular

#### *property* purelist_depth

#### *property* minmax_depth

#### *property* branch_depth

#### \_touch_data(recursive)

#### \_touch_shape(recursive)

#### \_\_getitem_\_(where)

#### \_getitem(where)

#### \_getitem_field(where, only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ()) → awkward.contents.content.Content

#### \_getitem_fields(where, only_fields: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ())

#### to_packed(recursive: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → awkward._typing.Self

#### to_list(behavior=None)

#### \_to_list(behavior, json_conversions)

#### to_backend(backend: awkward._backends.backend.Backend | [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → awkward._typing.Self

#### materialize() → awkward._typing.Self

#### *property* is_all_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* is_any_materialized *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### \_\_copy_\_() → awkward._typing.Self

#### \_\_deepcopy_\_(memo)

#### copy(array=UNSET, at=UNSET) → awkward._typing.Self
