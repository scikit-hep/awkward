# ak.str.join_element_wise

Defined in [awkward.operations.str.akstr_join_element_wise](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_join_element_wise.py) on [line 20](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_join_element_wise.py#L20).

#### ak.str.join_element_wise(\*arrays, highlevel=True, behavior=None, attrs=None)

* **Parameters:**
  * **arrays** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Broadcasts and concatenates all but the last array of strings in `arrays`;
the last is used as a separator.

Note: this function does not raise an error if the `array` does not
contain any string or bytestring data.

Requires the pyarrow library and calls
[pyarrow.compute.binary_join_element_wise](https://arrow.apache.org/docs/python/generated/pyarrow.compute.binary_join_element_wise.html).

Unlike Arrow’s `binary_join_element_wise`, this function has no `null_handling`
and `null_replacement` arguments. This function’s behavior is like
`null_handling="emit_null"` (Arrow’s default). The other cases can be implemented
with Awkward slices, [`ak.drop_none`](sphinx-llm:2495120d552d4c9f9d081cd38e6b71b4#ak.drop_none), and [`ak.fill_none`](sphinx-llm:e26b7c2675674008bc03fc5f18da53c5#ak.fill_none).

See also: [`ak.str.join`](sphinx-llm:b8306e5723e040b0a6ca255a4241c142#ak.str.join).
