# ak.from_dlpack

Defined in [awkward.operations.ak_from_dlpack](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_dlpack.py) on [line 15](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_dlpack.py#L15).

#### ak.from_dlpack(array, \*, prefer_cpu=True, regulararray=False, highlevel=True, behavior=None, primitive_policy='error', attrs=None)

Converts a DLPack-aware array into an Awkward Array.

The data is not copied: the buffer is shared through the DLPack protocol.

The resulting layout may involve the following [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) types
(only):

* [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)
* [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray) if `regulararray=True`.

* **Parameters:**
  * **array** (*cp.ndarray*) – The DLPack-supporting array to convert into an
    Awkward Array.
  * **prefer_cpu** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, and the array device supports both CPU and
    GPU backends, prefer the CPU; otherwise, prefer the GPU.
  * **regulararray** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True and the array is multidimensional,
    the dimensions are represented by nested [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray)
    nodes; if False and the array is multidimensional, the dimensions
    are represented by a multivalued [`ak.contents.NumpyArray.shape`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray.shape).
    If the array is one-dimensional, this has no effect.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) built from the given DLPack-aware array.
