# ak.strings_astype

Defined in [awkward.operations.ak_strings_astype](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_strings_astype.py) on [line 17](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_strings_astype.py#L17).

#### ak.strings_astype(array, to, \*, highlevel=True, behavior=None, attrs=None)

Converts all strings in the array to a new type, leaving the structure untouched.

See also `ak.numbers_astype`.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **to** (*dtype* *or* *dtype specifier*) – Type to convert the strings into.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with every string converted to the given `to` type, leaving the
  structure untouched.

### Examples

For example,

```pycon
>>> array = ak.Array(["1", "2", "    3    ", "00004", "-5"])
>>> ak.strings_astype(array, np.int32)
<Array [1, 2, 3, 4, -5] type='5 * int32'>
```

and

```pycon
>>> array = ak.Array(["1.1", "2.2", "    3.3    ", "00004.4", "-5.5"])
>>> ak.strings_astype(array, np.float64)
<Array [1.1, 2.2, 3.3, 4.4, -5.5] type='5 * float64'>
```

and finally,

```pycon
>>> array = ak.Array([["1.1", "2.2", "    3.3    "], [], ["00004.4", "-5.5"]])
>>> ak.strings_astype(array, np.float64)
<Array [[1.1, 2.2, 3.3], [], [4.4, -5.5]] type='3 * var * float64'>
```
