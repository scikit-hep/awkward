# ak.to_cupy

Defined in [awkward.operations.ak_to_cupy](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_cupy.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_cupy.py#L12).

#### ak.to_cupy(array)

Converts an Awkward Array into a CuPy array, if possible.

If the data are numerical and regular (nested lists have equal lengths in
each dimension, as described by the [`ak.Array.type`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.type)), they can be losslessly
converted to a CuPy array and this function returns without an error.

Otherwise, the function raises an error.

If `array` is a scalar, it is converted into a CuPy scalar.

See also [`ak.from_cupy`](sphinx-llm:6894cf5bed1a44eb9318ed3becb4e639#ak.from_cupy) and [`ak.to_numpy`](sphinx-llm:0dc12c9b410c4566b8e6578a4b063dbf#ak.to_numpy).

* **Parameters:**
  **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
* **Returns:**
  A CuPy array with the same data as `array`, if the conversion is possible.
