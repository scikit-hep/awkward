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

How to ensure that an array is valid
====================================

Awkward Arrays are complex data structures with their own rules for internal consistency. In principle, all data sources should serve valid array structures and all operations on valid structures should return valid structures. However, errors sometimes happen.

Awkward Array's compiled routines check for validity in the course of computation, so that errors are reported as Python exceptions, rather than undefined behavior or segmentation faults. However, those errors can be hard to understand because the invalid structure might have been constructed much earlier in a program than the point where it is discovered.

For that reason, you have tools to check an Awkward Array's internal validity: {func}`ak.is_valid`, {func}`ak.validity_error`, and the `check_valid` argument to constructors like {obj}`ak.Array`.

```{code-cell} ipython3
import awkward as ak
```

To demonstrate, here's a valid array:

```{code-cell} ipython3
array_is_valid = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
array_is_valid
```

and here is a copy of it that I will make invalid.

```{code-cell} ipython3
array_is_invalid = ak.copy(array_is_valid)
```

```{code-cell} ipython3
array_is_invalid.layout
```

```{code-cell} ipython3
array_is_invalid.layout.offsets.data
```

```{code-cell} ipython3
array_is_invalid.layout.offsets.data[3] = 100

array_is_invalid.layout
```

The {func}`ak.is_valid` function only tells us whether an array is valid or not:

```{code-cell} ipython3
ak.is_valid(array_is_valid)
```

```{code-cell} ipython3
ak.is_valid(array_is_invalid)
```

But the {func}`ak.validity_error` function tells us what the error was (if any).

```{code-cell} ipython3
ak.validity_error(array_is_valid)
```

```{code-cell} ipython3
ak.validity_error(array_is_invalid)
```

If you suspect that an array is invalid or becomes invalid in the course of your program, you can either use these functions to check or construct arrays with `check_valid=True` in the {obj}`ak.Array` constructor.

```{code-cell} ipython3
ak.Array(array_is_valid, check_valid=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception]
---
ak.Array(array_is_invalid, check_valid=True)
```
