# ak.where

Defined in [awkward.operations.ak_where](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_where.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_where.py#L19).

#### ak.where(condition, \*args, mergebool=True, highlevel=True, behavior=None, attrs=None)

Selects elements from `x` or `y` by a condition, or finds where it is True.

This function has a one-argument form, `condition` without `x` or `y`, and
a three-argument form, `condition`, `x`, and `y`. In the one-argument form,
it is completely equivalent to NumPy’s
[nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html)
function.

In the three-argument form, it acts as a vectorized ternary operator:
`condition`, `x`, and `y` must all have the same length and:

```default
output[i] = x[i] if condition[i] else y[i]
```

for all `i`. The structure of `x` and `y` do not need to be the same; if
they are incompatible types, the output will have `ak.type.UnionType`.

* **Parameters:**
  * **condition** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes) of booleans.
  * **x** – Optional array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes) with the same
    length as `condition`.
  * **y** – Optional array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes) with the same
    length as `condition`.
  * **mergebool** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default is True*) – If True, boolean and numeric data
    can be combined into the same buffer, losing information about
    False vs `0` and True vs `1`; otherwise, they are kept in separate
    buffers with distinct types (using an [`ak.contents.UnionArray`](sphinx-llm:dd41ab713e6a4626a50f392463f45ad1#ak.contents.UnionArray)).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default is True*) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array);
    otherwise, return a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with elements taken from `x` where `condition` is true and from `y`
  otherwise (three-argument form), or the integer indices where `condition` is
  true (one-argument form).
