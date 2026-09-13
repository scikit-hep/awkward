# ak.from_numpy

Defined in [awkward.operations.ak_from_numpy](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_numpy.py) on [line 11](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_numpy.py#L11).

#### ak.from_numpy(array, \*, regulararray=False, recordarray=True, highlevel=True, behavior=None, primitive_policy='error', attrs=None)

Converts a NumPy array into an Awkward Array.

The data is not copied: the Awkward Array shares the NumPy array’s
buffers. There are two exceptions: the mask of a `np.ma.MaskedArray` is
converted, although its data buffer is still shared, and an array of
Unicode strings is re-encoded as UTF-8.

The resulting layout can only involve the following [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) types:

* [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)
* [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray) or [`ak.contents.UnmaskedArray`](sphinx-llm:c65ee37a79b9404bbe625ef99e63bade#ak.contents.UnmaskedArray) if the
  `array` is an np.ma.MaskedArray.
* [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray) if `regulararray=True`.
* [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) if `recordarray=True`.

See also [`ak.to_numpy`](sphinx-llm:0dc12c9b410c4566b8e6578a4b063dbf#ak.to_numpy) and [`ak.from_cupy`](sphinx-llm:6894cf5bed1a44eb9318ed3becb4e639#ak.from_cupy).

* **Parameters:**
  * **array** (*np.ndarray*) – The NumPy array to convert into an Awkward Array.
    This array can be a np.ma.MaskedArray.
  * **regulararray** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True and the array is multidimensional,
    the dimensions are represented by nested [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray)
    nodes; if False and the array is multidimensional, the dimensions
    are represented by a multivalued [`ak.contents.NumpyArray.shape`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray.shape).
    If the array is one-dimensional, this has no effect.
  * **recordarray** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True and the array is a NumPy structured array
    (dtype.names is not None), the fields are represented by an
    [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray); if False and the array is a structured
    array, the structure is left in the [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) `format`,
    which some functions do not recognize.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) built from the given NumPy array.
