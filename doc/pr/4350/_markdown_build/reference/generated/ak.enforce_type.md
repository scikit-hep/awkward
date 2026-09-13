# ak.enforce_type

Defined in [awkward.operations.ak_enforce_type](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_enforce_type.py) on [line 22](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_enforce_type.py#L22).

#### ak.enforce_type(array, type, \*, highlevel=True, behavior=None, attrs=None)

Returns an array whose structure is modified to match the given type.

In addition to preserving the existing type and/or changing parameters,

- [`ak.types.OptionType`](sphinx-llm:f8e0eb36e1334eef887fb5ab81458600#ak.types.OptionType) can be added
  ```pycon
  >>> a = ak.Array([1, 2, 3])
  >>> a.type.show()
  3 * int64
  >>> b = ak.enforce_type(a, "?int64")
  >>> b.type.show()
  3 * ?int64
  ```

  or removed (if there are no missing values)
  ```pycon
  >>> a = ak.Array([1, 2, 3, None])
  >>> b = a[:-1]
  >>> b.type.show()
  3 * ?int64
  >>> c = ak.enforce_type(b, "int64")
  >>> c.type.show()
  3 * int64
  ```
- [`ak.types.UnionType`](sphinx-llm:5e1485b00d7b47f39f0082bb755fa92f#ak.types.UnionType) can
  * grow to include new variant types,

  ```pycon
  >>> a = ak.Array([{'x': 1}, 2.0])
  >>> a.type.show()
  2 * union[
      {
          x: int64
      },
      float64
  ]
  >>> b = ak.enforce_type(a, "union[{x: int64}, float64, string]")
  >>> b.type.show()
  2 * union[
  {
      x: float32
  },
      float64,
      string
  ]
  ```

  * convert to a single type,

  ```pycon
  >>> a = ak.concatenate([
  ...   ak.Array([{'x': 1}, {'x': 2}]),
  ...   ak.Array([{'x': True, "y": None}, {'x': False, "y": None}])
  ... ])
  >>> a.type.show()
  4 * union[
      {
          x: int64
      },
      {
          x: bool,
          y: ?unknown
      }
  ]
  >>> b = ak.enforce_type(a, "{x: float64}")
  >>> b.type.show()
  4 * {
      x: float64
  }
  ```

  * project to a single type (if conversion to a single type is not possible, and the union contains no values for this type),

  ```pycon
  >>> a = ak.concatenate([
  ...   ak.Array([{'x': 1}, {'x': 2}]),
  ...   ak.Array([{'x': "yes", "y": None}, {'x': "no", "y": None}])
  ... ])
  >>> b = a[:2]
  >>> b.type.show()
  2 * union[
      {
          x: int64
      },
      {
          x: string,
          y: ?unknown
      }
  ]
  >>> c = ak.enforce_type(b, "{x: int64}")
  >>> c.type.show()
  2 * {
      x: int64
  }
  ```

  * change type in a single variant.

  ```pycon
  >>> a = ak.Array([{'x': 1}, 2.0])
  >>> a.type.show()
  2 * union[
      {
          x: int64
      },
      float64
  ]
  >>> b = ak.enforce_type(a, "union[{x: float32}, float64]")
  >>> b.type.show()
  2 * union[
  {
      x: float32
  },
      float64
  ]
  ```

  Due to these rules, changes to more than one variant of a union must be performed with multiple calls to [`ak.enforce_type`](sphinx-llm:cc82b226ee1b455b853bf3b9011fe98d)
- [`ak.types.RecordType`](sphinx-llm:516b58d6a2f7412aadbaf0859053019a#ak.types.RecordType) can
  * grow to include new optional fields / slots,

  ```pycon
  >>> a = ak.Array([{'x': 1}])
  >>> a.type.show()
  1 * {
      x: int64
  }
  >>> b = ak.enforce_type(a, "{x: int64, y: ?float32}")
  >>> b.type.show()
  1 * {
      x: int64,
      y: ?float32
  }
  ```

  * shrink to drop existing fields / slots.

  ```pycon
  >>> a = ak.Array([{'x': 1, 'y': 1j+3}])
  >>> a.type.show()
  1 * {
      x: int64,
      y: complex128
  }
  >>> b = ak.enforce_type(a, "{x: int64}")
  >>> b.type.show()
  1 * {
      x: int64
  }
  ```

  A [`ak.types.RecordType`](sphinx-llm:516b58d6a2f7412aadbaf0859053019a#ak.types.RecordType) may only be converted to another [`ak.types.RecordType`](sphinx-llm:516b58d6a2f7412aadbaf0859053019a#ak.types.RecordType) if it is of the same flavour, i.e.
  tuples can be converted to tuples, or records to records. Where a new field/slot is added to a [`ak.types.RecordType`](sphinx-llm:516b58d6a2f7412aadbaf0859053019a#ak.types.RecordType),
  it must be an [`ak.types.OptionType`](sphinx-llm:f8e0eb36e1334eef887fb5ab81458600#ak.types.OptionType). For tuples, slots may only be added to the end of the tuple
- [`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType) can convert to a [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType)
  ```pycon
  >>> a = ak.to_regular([[1, 2, 3], [4, 5, 6]])
  >>> a.type.show()
  2 * 3 * int64
  >>> b = ak.enforce_type(a, "var * int64")
  >>> b.type.show()
  2 * var * int64
  ```
- [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType) can convert to a [`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType)
  ```pycon
  >>> a = ak.Array([[1, 2, 3], [4, 5, 6]])
  >>> a.type.show()
  2 * var * int64
  >>> b = ak.enforce_type(a, "3 * int64")
  >>> b.type.show()
  2 * 3 * int64
  ```
- [`ak.types.NumpyType`](sphinx-llm:7284cac2045946428822c632c80c9067#ak.types.NumpyType) can change primitive
  ```pycon
  >>> a = ak.Array([1, 2, 3])
  >>> a.type.show()
  3 * int64
  >>> b = ak.enforce_type(a, "float32")
  >>> b.type.show()
  3 * float32
  ```
- [`ak.types.UnknownType`](sphinx-llm:4c8a8f1814d7428e80e2d1e3a84b2dde#ak.types.UnknownType) can be converted to any other type
  ```pycon
  >>> a = ak.Array([])
  >>> a.type.show()
  0 * unknown
  >>> b = ak.enforce_type(a, "float32")
  >>> b.type.show()
  0 * float32
  ```

  and can be converted to from any other type.
  ```pycon
  >>> a = ak.Array([1, 2, 3])
  >>> a.type.show()
  3 * int64
  >>> b = ak.enforce_type(a, "?unknown")
  >>> b.type.show()
  3 * ?unknown
  ```

The conversion rules outlined above are not data-dependent; the appropriate rule is chosen from the layout and the
given type value. If the conversion is not possible given the layout data, e.g. a conversion from an irregular list
to a regular type, it will fail.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **type** ([`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type), or str) – The type that `array` will be enforced to.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array whose structure is modified to match the given type.
