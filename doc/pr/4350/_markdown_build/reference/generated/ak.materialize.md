# ak.materialize

Defined in [awkward.operations.ak_materialize](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_materialize.py) on [line 11](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_materialize.py#L11).

#### ak.materialize(array, highlevel=True, behavior=None, attrs=None)

Materializes any virtual buffers in the array.

If the input array is not an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) or an [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content), an
error will be raised.

The buffers of the returned array are no longer `VirtualNDArray` objects
even if there were any. They will become one of `numpy.ndarray`,
`cupy.ndarray`, or `jax.numpy.ndarray` objects, depending on the array’s
backend.

* **Parameters:**
  * **array** – Array-like data (either an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) or an [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content)).
    An array that may contain virtual buffers to be materialized.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default is True*) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array);
    otherwise, return a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with the same data as the input and all virtual buffers
  materialized.
