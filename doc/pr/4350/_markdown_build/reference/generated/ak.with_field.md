# ak.with_field

Defined in [awkward.operations.ak_with_field](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_with_field.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_with_field.py#L19).

#### ak.with_field(array, what, where=None, \*, highlevel=True, behavior=None, attrs=None)

Returns an array or record with a new field added, or an existing field replaced.

This function returns a new array or record; see [`ak.Array.__setitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__setitem__)
and [`ak.Record.__setitem__`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record.__setitem__) for a variant that changes the high-level
object in-place. (These methods internally use [`ak.with_field`](sphinx-llm:542360a596bb42ca8c94628d544b745f), so
performance is not a factor in choosing one over the other.)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **what** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes) to add as a new field.
  * **where** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *non-empy sequence* *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If None, the new field
    has no name (can be accessed as an integer slot number in a
    string); If str, the name of the new field. If a sequence, it is
    interpreted as a path where to add the field in a nested record.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) or [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) (or low-level equivalent, if
  `highlevel=False`) with a new field attached.
