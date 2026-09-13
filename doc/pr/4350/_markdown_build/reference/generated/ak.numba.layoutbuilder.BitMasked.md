# ak.numba.layoutbuilder.BitMasked

Defined in [awkward.numba.layoutbuilder](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/numba/layoutbuilder.py) on [line 428](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/numba/layoutbuilder.py#L428).

#### *class* ak.numba.layoutbuilder.BitMasked(dtype, content, valid_when, lsb_order, \*, parameters=None, initial=1024, resize=8.0)

#### \_mask

#### \_content

#### \_valid_when

#### \_lsb_order

#### \_current_byte_index

#### \_\_repr_\_()

#### numbatype()

#### *property* content

#### *property* valid_when

#### *property* lsb_order

#### *property* form

#### \_append_begin()

Private helper function.

#### \_append_end()

Private helper function.

#### append_valid()

#### extend_valid(size)

#### append_invalid()

#### extend_invalid(size)

#### clear()

#### \_\_len_\_()

#### is_valid(error: [str](https://docs.python.org/3/library/stdtypes.html#str))

#### snapshot() → awkward.contents.Content
