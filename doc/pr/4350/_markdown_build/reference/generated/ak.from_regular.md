# ak.from_regular

Defined in [awkward.operations.ak_from_regular](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_regular.py) on [line 17](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_regular.py#L17).

#### ak.from_regular(array, axis=1, \*, highlevel=True, behavior=None, attrs=None)

Converts one or all regular axes into irregular ones.

See also [`ak.to_regular`](sphinx-llm:2a2dc6754b2948d79f20c53eb2b3d2b9#ak.to_regular).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* *None*) – The dimension at which this operation is applied.
    The outermost dimension is `0`, followed by `1`, etc., and negative
    values count backward from the innermost: `-1` is the innermost
    dimension, `-2` is the next level up, etc. If None, convert all
    regular dimensions into variable ones.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with one or all regular axes converted to irregular (var) axes.

### Examples

```pycon
>>> regular = ak.Array(np.arange(2*3*5).reshape(2, 3, 5))
>>> regular.type.show()
2 * 3 * 5 * int64
>>> ak.from_regular(regular).type.show()
2 * var * 5 * int64
>>> ak.from_regular(regular, axis=2).type.show()
2 * 3 * var * int64
>>> ak.from_regular(regular, axis=-1).type.show()
2 * 3 * var * int64
```
