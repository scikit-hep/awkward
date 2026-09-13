# How to ensure that an array is valid

Awkward Arrays are complex data structures with their own rules for internal consistency. In principle, all data sources should serve valid array structures and all operations on valid structures should return valid structures. However, errors sometimes happen.

Awkward Array’s compiled routines check for validity in the course of computation, so that errors are reported as Python exceptions, rather than undefined behavior or segmentation faults. However, those errors can be hard to understand because the invalid structure might have been constructed much earlier in a program than the point where it is discovered.

For that reason, you have tools to check an Awkward Array’s internal validity: [`ak.is_valid()`](sphinx-llm:cb72d589ba0e433d9cedf3f12cc0da1c#ak.is_valid), [`ak.validity_error()`](sphinx-llm:ccbf3feebdb343f8ae3e23ce953f394a#ak.validity_error), and the `check_valid` argument to constructors like [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array).

```ipython3
import awkward as ak
```

To demonstrate, here’s a valid array:

```ipython3
array_is_valid = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
array_is_valid
```

and here is a copy of it that I will make invalid.

```ipython3
array_is_invalid = ak.copy(array_is_valid)
```

```ipython3
array_is_invalid.layout
```

```ipython3
array_is_invalid.layout.offsets.data
```

```ipython3
array_is_invalid.layout.offsets.data[3] = 100

array_is_invalid.layout
```

The [`ak.is_valid()`](sphinx-llm:cb72d589ba0e433d9cedf3f12cc0da1c#ak.is_valid) function only tells us whether an array is valid or not:

```ipython3
ak.is_valid(array_is_valid)
```

```ipython3
ak.is_valid(array_is_invalid)
```

But the [`ak.validity_error()`](sphinx-llm:ccbf3feebdb343f8ae3e23ce953f394a#ak.validity_error) function tells us what the error was (if any).

```ipython3
ak.validity_error(array_is_valid)
```

```ipython3
ak.validity_error(array_is_invalid)
```

If you suspect that an array is invalid or becomes invalid in the course of your program, you can either use these functions to check or construct arrays with `check_valid=True` in the [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) constructor.

```ipython3
ak.Array(array_is_valid, check_valid=True)
```

```ipython3
ak.Array(array_is_invalid, check_valid=True)
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[12], line 1
----> 1 ak.Array(array_is_invalid, check_valid=True)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:367, in Array.__init__(self, data, behavior, with_name, check_valid, backend, attrs, named_axis)
    364 self._update_class()
    366 if check_valid:
--> 367     ak.operations.validity_error(self, exception=True)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_dispatch.py:40, in named_high_level_function.<locals>.dispatch(*args, **kwargs)
     37 @wraps(func)
     38 def dispatch(*args, **kwargs):
     39     # NOTE: this decorator assumes that the operation is exposed under `ak.`
---> 40     with OperationErrorContext(name, args, kwargs):
     41         gen_or_result = func(*args, **kwargs)
     42         if isgenerator(gen_or_result):

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_errors.py:79, in ErrorContext.__exit__(self, exception_type, exception_value, traceback)
     77     self._slate.__dict__.clear()
     78     # Handle caught exception
---> 79     raise self.decorate_exception(exception_type, exception_value)
     80 else:
     81     # Step out of the way so that another ErrorContext can become primary.
     82     if self.primary() is self:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_dispatch.py:66, in named_high_level_function.<locals>.dispatch(*args, **kwargs)
     64 # Failed to find a custom overload, so resume the original function
     65 try:
---> 66     next(gen_or_result)
     67 except StopIteration as err:
     68     return err.value

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_validity_error.py:32, in validity_error(array, exception)
     29 yield (array,)
     31 # Implementation
---> 32 return _impl(array, exception)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_validity_error.py:42, in _impl(array, exception)
     39 out = ak._do.validity_error(layout, path="highlevel")
     41 if out not in (None, "") and exception:
---> 42     raise ValueError(out)
     43 else:
     44     return out

ValueError: at highlevel ("<class 'awkward.contents.listoffsetarray.ListOffsetArray'>"): stop[i] > len(content) at i=2 (in compiled code: https://github.com/scikit-hep/awkward/blob/awkward-cpp-56/awkward-cpp/src/cpu-kernels/awkward_ListArray_validity.cpp#L24)

This error occurred while calling

    ak.validity_error(
        <Array [[0, 1, 2], [], ..., [], [6, 7, 8, 9]] type='5 * var * int64'>
        exception = True
    )
```
