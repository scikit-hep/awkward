# ak.pad_none

Defined in [awkward.operations.ak_pad_none](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_pad_none.py) on [line 20](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_pad_none.py#L20).

#### ak.pad_none(array, target, axis=1, \*, clip=False, highlevel=True, behavior=None, attrs=None)

Increases the lengths of lists to a target length by adding None values.

Note that the `clip` parameter not only determines whether the lengths are
at least `target` or exactly `target`, it also determines the type of the
output:

* `clip=True` returns regular lists ([`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType)), and
* `clip=False` returns in-principle variable lengths
  ([`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType)).

The in-principle variable-length lists might, in fact, all have the same
length, but the type difference is significant, for instance in
broadcasting rules (see [`ak.broadcast_arrays`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays)).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **target** ([*int*](https://docs.python.org/3/library/functions.html#int)) – The intended length of the lists. If `clip=True`,
    the output lists will have exactly this length; otherwise,
    they will have *at least* this length.
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The dimension at which this operation is applied. The
    outermost dimension is `0`, followed by `1`, etc., and negative
    values count backward from the innermost: `-1` is the innermost
    dimension, `-2` is the next level up, etc.
    If a str, it is interpreted as the name of the axis which maps
    to an int if named axes are present. Named axes are attached
    to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and removed with
    [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **clip** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, the output lists will have regular lengths
    ([`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType)) of exactly `target`; otherwise the
    output lists will have in-principle variable lengths
    ([`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType)) of at least `target`.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array whose lists are padded with None to at least the target `length`.

### Examples

Consider the following

```pycon
>>> array = ak.Array([[[1.1, 2.2, 3.3],
...                    [],
...                    [4.4, 5.5],
...                    [6.6]],
...                   [],
...                   [[7.7],
...                    [8.8, 9.9]
...                   ]])
```

At `axis=0`, this operation pads the whole array, adding None at the
outermost level:

```pycon
>>> ak.pad_none(array, 5, axis=0).show()
[[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]],
 [],
 [[7.7], [8.8, 9.9]],
 None,
 None]
```

At `axis=1`, this operation pads the first nested level:

```pycon
>>> ak.pad_none(array, 3, axis=1).show()
[[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]],
 [None, None, None],
 [[7.7], [8.8, 9.9], None]]
```

And so on for higher values of `axis`:

```pycon
>>> ak.pad_none(array, 2, axis=2).show()
[[[1.1, 2.2, 3.3], [None, None], [4.4, 5.5], [6.6, None]],
 [],
 [[7.7, None], [8.8, 9.9]]]
```

The difference between

```pycon
>>> ak.pad_none(array, 2, axis=2)
<Array [[[1.1, 2.2, 3.3], ..., [...]], ...] type='3 * var * var * ?float64'>
```

and

```pycon
>>> ak.pad_none(array, 2, axis=2, clip=True)
<Array [[[1.1, 2.2], ..., [6.6, None]], ...] type='3 * var * 2 * ?float64'>
```

is not just in the length of `[1.1, 2.2, 3.3]` vs `[1.1, 2.2]`, but also
in the distinction between the following types.

```pycon
>>> ak.pad_none(array, 2, axis=2).type.show()
3 * var * var * ?float64
>>> ak.pad_none(array, 2, axis=2, clip=True).type.show()
3 * var *   2 * ?float64
```
