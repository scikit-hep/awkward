# ak.without_field

Defined in [awkward.operations.ak_without_field](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_without_field.py) on [line 17](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_without_field.py#L17).

#### ak.without_field(array, where, \*, highlevel=True, behavior=None, attrs=None)

Returns an array or record with the named field removed.

This function does not change the array in-place.

See [`ak.Array.__delitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__delitem__) and [`ak.Record.__delitem__`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record.__delitem__) for a variant that
changes the high-level object in-place. (These methods internally use
[`ak.without_field`](sphinx-llm:e9c5c99af680435298573783508de98e), so performance is not a factor in choosing one over the
other.)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **where** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *non-empy sequence* *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If str, the name of the field
    to be removed. If a sequence, it is interpreted as a path where to
    remove the field in a nested record.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) or [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) (or low-level equivalent, if
  `highlevel=False`) with an existing field removed.
