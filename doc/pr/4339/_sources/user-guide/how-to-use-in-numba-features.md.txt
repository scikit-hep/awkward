---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

Awkward Array features that are supported in Numba-compiled functions
=====================================================================

See the [Numba documentation](https://numba.readthedocs.io/), which maintains lists of

* [supported Python language features](https://numba.pydata.org/numba-doc/dev/reference/pysupported.html) and
* [supported NumPy library features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)

in JIT-compiled functions. This page describes the supported Awkward Array library features.

```{code-cell} ipython3
import awkward as ak
import numpy as np
import numba as nb
```

## Passing Awkward Arrays as arguments to a function

The main use is to pass an Awkward Array into a function that has been JIT-compiled by Numba. As many arguments as you want can be Awkward Arrays, and they don't have to have the same length or shape.

```{code-cell} ipython3
array1 = ak.Array([[0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
array2 = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

```{code-cell} ipython3
@nb.jit
def first_array(array):
    for i, list_of_numbers in enumerate(array):
        for x in list_of_numbers:
            if x == 3.3:
                return i

@nb.jit
def second_array(array):
    for i, list_of_records in enumerate(array):
        for record in list_of_records:
            if record.x == 3.3:
                return i

@nb.jit
def where_is_3_point_3(a, b):
    return first_array(a), second_array(b)
```

```{code-cell} ipython3
where_is_3_point_3(array1, array2)
```

The only constraint is that union types can't be _accessed_ within the compiled function. (Heterogeneous _parts_ of an array can be ignored and passed through a compiled function.)

## Returning Awkward Arrays from a function

Parts of the input array can be returned from a compiled function.

```{code-cell} ipython3
@nb.jit
def first_array(array):
    for list_of_numbers in array:
        for x in list_of_numbers:
            if x == 3.3:
                return list_of_numbers

@nb.jit
def second_array(array):
    for list_of_records in array:
        for record in list_of_records:
            if record.x == 3.3:
                return record

@nb.jit
def find_3_point_3(a, b):
    return first_array(a), second_array(b)
```

```{code-cell} ipython3
found_a, found_b = find_3_point_3(array1, array2)
```

```{code-cell} ipython3
found_a
```

```{code-cell} ipython3
found_b
```

## Cannot use `ak.*` functions or ufuncs

Outside of a compiled function, Awkward's vectorized `ak.*` functions and NumPy's [universal functions (ufuncs)](https://numpy.org/doc/stable/reference/ufuncs.html) should be highly preferred over for-loop iteration because they are much faster.

Inside of a compiled function, however, they can't be used at all. Use for-loops and if-statements instead.

This is an either-or choice at the boundary of a `@nb.jit`-compiled function. (Even if `ak.*` had been implemented in Numba's compiled context, it would be slower than _compiled_ for-loops and if-statements because of the intermediate arrays they would necessarily create.)

## Cannot use fancy slicing

Similarly, any slicing other than

* a single integer, like `array[i]` where `i` is an integer, or
* a single record field as a _constant, literal_ string, like `array["x"]` or `array.x`,

is not allowed. Unpack the data structures one level at a time.

## Casting one-dimensional arrays as NumPy

One-dimensional Awkward Arrays of numbers, which are completely equivalent to NumPy arrays, can be _cast_ as NumPy arrays within the compiled function.

```{code-cell} ipython3
@nb.jit
def return_last_y_list_squared(array):
    y_list_squared = None
    for list_of_records in array:
        for record in list_of_records:
            y_list_squared = np.asarray(record.y)**2
    return y_list_squared
```

```{code-cell} ipython3
return_last_y_list_squared(array2)
```

This ability to cast Awkward Arrays as NumPy arrays, and then use NumPy's ufuncs or fancy slicing, softens the law against vectorized functions in the compiled context. (However, making intermediate NumPy arrays is just as bad as making intermediate Awkward Arrays.

## Creating new arrays with `ak.ArrayBuilder`

Numba can create NumPy arrays inside a compiled function and return them as NumPy arrays in Python, but Awkward Arrays are more complex and this is not possible. (Aside from implementation, what would be the interface? Data in Numba's compiled context must be fully typed, and Awkward Array types are complex.)

Instead, arrays can be built with {obj}`ak.ArrayBuilder`, which can be used in compiled contexts and discovers type dynamically. Each {obj}`ak.ArrayBuilder` must be instantiated outside of a compiled function and passed in, and then its {func}`ak.ArrayBuilder.snapshot` (which creates the {obj}`ak.Array`) must be called outside of the compiled function, like this:

```{code-cell} ipython3
@nb.jit
def create_ragged_array(builder, n):
    for i in range(n):
        builder.begin_list()
        for j in range(i):
            builder.integer(j)
        builder.end_list()
    return builder
```

```{code-cell} ipython3
builder = ak.ArrayBuilder()

create_ragged_array(builder, 10)

array = builder.snapshot()

array
```

or, more succintly,

```{code-cell} ipython3
create_ragged_array(ak.ArrayBuilder(), 10).snapshot()
```

Note that we didn't need to specify that the type of the data would be `var * int64`; this was determined by the way that {obj}`ak.ArrayBuilder` was called: {func}`ak.ArrayBuilder.integer` was only ever called between {func}`ak.ArrayBuilder.begin_list` and {func}`ak.ArrayBuilder.end_list`, and hence the type is `var * int64`.

Note that {obj}`ak.ArrayBuilder` can be used outside of compiled functions, too, so it can be tested interactively:

```{code-cell} ipython3
with builder.record():
    builder.field("x").real(3.14)
    with builder.field("y").list():
        builder.string("one")
        builder.string("two")
        builder.string("three")
```

```{code-cell} ipython3
builder.snapshot()
```

But the context managers, `with builder.record()` and `with builder.list()`, don't work in Numba-compiled functions because Numba does not yet support it as a language feature.

## Overriding behavior with `ak.behavior`

Just as behaviors can be customized for Awkward Arrays in general, they can be customized in the compiled context as well. See the last section of the {obj}`ak.behavior` reference for details.
