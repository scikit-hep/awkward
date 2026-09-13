# ak.to_numpy

Defined in [awkward.operations.ak_to_numpy](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_numpy.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_numpy.py#L12).

#### ak.to_numpy(array, \*, allow_missing=True)

Converts an Awkward Array into a NumPy array, if possible.

If the data are numerical and regular (nested lists have equal lengths in
each dimension, as described by the [`ak.Array.type`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.type)), they can be losslessly
converted to a NumPy array and this function returns without an error.

Otherwise, the function raises an error. It does not create a NumPy array
with dtype `"O"` for `np.object_` (see the
[note on object_ type](https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html#arrays-scalars-built-in))
since silent conversions to dtype `"O"` arrays would not only be a
significant performance hit, but would also break functionality, since
nested lists in a NumPy `"O"` array are severed from the array and cannot
be sliced as dimensions.

If `array` is not an Awkward Array, then this function is equivalent to
calling `np.asarray` on it.

If `allow_missing` is True; NumPy
[masked arrays](https://docs.scipy.org/doc/numpy/reference/maskedarray.html)
are a possible result; otherwise, missing values (None) cause this
function to raise an error.

See also [`ak.from_numpy`](sphinx-llm:efd7416013fa442595132125c93aa4ea#ak.from_numpy) and [`ak.to_cupy`](sphinx-llm:0d44c808b5dd4b28b9cc5805ab718f3f#ak.to_cupy).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **allow_missing** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – allow missing (None) values.
* **Returns:**
  A NumPy array with the same data as `array`, if the conversion is possible.
