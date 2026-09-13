# ak.from_cupy

Defined in [awkward.operations.ak_from_cupy](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_cupy.py) on [line 11](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_cupy.py#L11).

#### ak.from_cupy(array, \*, regulararray=False, highlevel=True, behavior=None, primitive_policy='error', attrs=None)

Converts a CuPy array into an Awkward Array.

The data is not copied: the Awkward Array shares the CuPy array’s GPU
buffer.

The resulting layout may involve the following [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) types
(only):

* [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)
* [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray) if `regulararray=True`.

See also [`ak.to_cupy`](sphinx-llm:0d44c808b5dd4b28b9cc5805ab718f3f#ak.to_cupy), [`ak.from_numpy`](sphinx-llm:efd7416013fa442595132125c93aa4ea#ak.from_numpy) and [`ak.from_jax`](sphinx-llm:b8454383661d47feaa9d4067f140e358#ak.from_jax).

* **Parameters:**
  * **array** (*cp.ndarray*) – The CuPy array to convert into an Awkward Array.
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
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) built from the given CuPy array.
