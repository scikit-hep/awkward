# ak.to_cudf

Defined in [awkward.operations.ak_to_cudf](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_cudf.py) on [line 10](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_cudf.py#L10).

#### ak.to_cudf(array)

Converts an Awkward Array into a cuDF Series.

Buffers that are not already in GPU memory will be transferred, and some
structural reformatting may happen to account for differences in
architecture.

This function requires the `cudf` library (< 25.12.00) and a compatible
GPU. cuDF versions 25.12.00 and later are not currently supported due to
incompatible changes in cuDF internals.

See also [`ak.to_cupy`](sphinx-llm:0d44c808b5dd4b28b9cc5805ab718f3f#ak.to_cupy), [`ak.from_cupy`](sphinx-llm:6894cf5bed1a44eb9318ed3becb4e639#ak.from_cupy), [`ak.to_dataframe`](sphinx-llm:470705b5237144ff83c26cdb7cdcee4d#ak.to_dataframe).

* **Parameters:**
  **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
* **Returns:**
  A cuDF Series with the same data as `array`.
