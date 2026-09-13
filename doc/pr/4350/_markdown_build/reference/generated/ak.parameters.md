# ak.parameters

Defined in [awkward.operations.ak_parameters](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_parameters.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_parameters.py#L19).

#### ak.parameters(array)

Returns the parameters dict of the outermost layout node.

Parameters are a dict from str to JSON-like objects, usually strings.
Every [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) node has a different set of parameters. Some
key names are special, such as `"__record__"` and `"__array__"` that name
particular records and arrays as capable of supporting special behaviors.

See [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) and [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for a more complete description of
behaviors.

* **Parameters:**
  **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
* **Returns:**
  The parameters dict of the outermost node of `array` (many types
  supported, including all Awkward Arrays and Records).
