# ak.unzip

Defined in [awkward.operations.ak_unzip](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_unzip.py) on [line 15](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_unzip.py#L15).

#### ak.unzip(array, \*, how=tuple, highlevel=True, behavior=None, attrs=None)

Splits records or tuples into a tuple or dict of arrays, one per field.

If the `array` does not contain tuples or records, the single `array` is
placed in a length 1 Python tuple (or dict).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **how** ([*type*](https://docs.python.org/3/library/functions.html#type)) – The type of the returned output. This can be `tuple` or `dict`.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  A Python tuple (or dict) of arrays, one for each field of the tuples or
  records in `array`.

### Examples

For example,

```pycon
>>> array = ak.Array([{"x": 1.1, "y": [1]},
...                   {"x": 2.2, "y": [2, 2]},
...                   {"x": 3.3, "y": [3, 3, 3]}])
>>> x, y = ak.unzip(array)
>>> x
<Array [1.1, 2.2, 3.3] type='3 * float64'>
>>> y
<Array [[1], [2, 2], [3, 3, 3]] type='3 * var * int64'>
```

The `how` argument determines the structure of the output. Using `how=dict`
returns a dictionary of arrays instead of a tuple, and let’s you round-trip
through `ak.zip`:

```pycon
>>> array = ak.Array([{"x": 1.1, "y": [1]},
...                   {"x": 2.2, "y": [2, 2]},
...                   {"x": 3.3, "y": [3, 3, 3]}])
>>> x = ak.unzip(array, how=dict)
>>> x
{'x': <Array [1.1, 2.2, 3.3] type='3 * float64'>,
 'y': <Array [[1], [2, 2], [3, 3, 3]] type='3 * var * int64'>}
>>> assert ak.zip(ak.unzip(array, how=dict), depth_limit=1).to_list() == array.to_list()  # True
```
