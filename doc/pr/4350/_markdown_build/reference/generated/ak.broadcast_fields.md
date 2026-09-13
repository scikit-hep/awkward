# ak.broadcast_fields

Defined in [awkward.operations.ak_broadcast_fields](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_broadcast_fields.py) on [line 18](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_broadcast_fields.py#L18).

#### ak.broadcast_fields(\*arrays, highlevel=True, behavior=None, attrs=None)

Returns a list of arrays whose record types contain the same fields.

* **Parameters:**
  * **arrays** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  Return a list of arrays whose types contain the same number of fields. Unlike
  [`ak.broadcast_arrays`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays), this function does not require record types to occur at the
  same depths. Where fields are missing from one record, they are inserted at the same
  position with an `option[unknown]` type. This type is easily erased by ufunc and
  concatenation operations.

### Examples

```pycon
>>> x, y = ak.broadcast_fields(
...     [{"x": {"y": 1, "z": 2, "w": [1]}}],
...     [{"x": [{"y": 1}]}],
... )
>>> x.type.show()
1 * {
    x: {
        y: int64,
        z: int64,
        w: var * int64
    }
}
>>> y.type.show()
1 * {
    x: var * {
        y: int64,
        z: ?unknown,
        w: ?unknown
    }
}
```
