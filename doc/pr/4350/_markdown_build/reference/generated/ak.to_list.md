# ak.to_list

Defined in [awkward.operations.ak_to_list](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_list.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_list.py#L19).

#### ak.to_list(array)

Converts an Awkward Array into Python objects.

If `array` is not recognized as an array, it is passed through as-is.

Awkward Array types have the following Pythonic translations.

* [`ak.types.NumpyType`](sphinx-llm:7284cac2045946428822c632c80c9067#ak.types.NumpyType): converted into bool, int, float, datetimes, etc.
  (Same as NumPy’s `ndarray.tolist`.)
* [`ak.types.OptionType`](sphinx-llm:f8e0eb36e1334eef887fb5ab81458600#ak.types.OptionType): missing values are converted into None.
* [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType): converted into list.
* [`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType): also converted into list. Python (and JSON)
  forms lose information about the regularity of list lengths.
* [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType) with parameter `"__array__"` equal to
  `"__bytestring__"`: converted into bytes.
* [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType) with parameter `"__array__"` equal to
  `"__string__"`: converted into str.
* `ak.types.RecordArray` without field names: converted into tuple.
* `ak.types.RecordArray` with field names: converted into dict.
* `ak.types.UnionArray`: Python data are naturally heterogeneous.

See also [`ak.from_iter`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) and [`ak.Array.tolist`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.tolist).

* **Parameters:**
  **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
* **Returns:**
  The contents of `array` as Python objects (lists, dicts, numbers, etc.).
