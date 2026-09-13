# How to use Awkward Arrays in C++ with cppyy

#### WARNING
Awkward Array can only work with `cppyy` 3.1 or later.

#### WARNING
`cppyy` must be in a different venv or conda environment from ROOT, if you have installed ROOT, because the two packages define modules with conflicting names.

The [cppyy](https://cppyy.readthedocs.io/en/latest/index.html) is an automatic, run-time, Python-C++ bindings generator, for calling C++ from Python and Python from C++. `cppyy` is based on the C++ interpreter `Cling`.

`cppyy` can understand Awkward Arrays. When an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) type is passed to a C++ function defined in `cppyy`, a `__cast_cpp__` magic function of an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) is invoked. The function dynamically generates a C++ type and a view of the array, if it has not been generated yet.

The view is a lightweight 40-byte C++ object dynamically allocated on the stack. This view is generated on demand - and only once per Awkward Array, the data are not copied.

```ipython3
import awkward as ak
ak.__version__
```

```ipython3
import awkward._connect.cling
```

```ipython3
import cppyy
cppyy.__version__
```

```myst-ansi
(Re-)building pre-compiled headers (options: -O2 -march=native); this may take a minute ...
ERROR: cannot find etc/dictpch/allHeaders.h file here ./etc/dictpch/allHeaders.h nor here etc/dictpch/allHeaders.h
```

```myst-ansi
/opt/hostedtoolcache/Python/3.11.0/x64/lib/python3.11/site-packages/cppyy_backend/loader.py:139: UserWarning: No precompiled header available (failed to build); this may impact performance.
  warnings.warn('No precompiled header available (%s); this may impact performance.' % msg)
input_line_10:2:45: error: explicit instantiation of '_M_use_local_data' does not refer to a function template, variable template, member function, member class, or static data member
template std::string::pointer  std::string::_M_use_local_data();
                                            ^
input_line_10:3:46: error: explicit instantiation of '_M_use_local_data' does not refer to a function template, variable template, member function, member class, or static data member
template std::wstring::pointer std::wstring::_M_use_local_data();
                                             ^
/opt/hostedtoolcache/Python/3.11.0/x64/lib/python3.11/site-packages/cppyy/__init__.py:356: UserWarning: CPyCppyy API not found (tried: /opt/hostedtoolcache/Python/3.11.0/x64/lib/python3.11/site-packages/../../../include/python3.11); set CPPYY_API_PATH envar to the 'CPyCppyy' API directory to fix
  warnings.warn("CPyCppyy API not found (tried: %s); set CPPYY_API_PATH envar to the 'CPyCppyy' API directory to fix" % apipath_extra)
```

Let’s define an Awkward Array as a list of records:

```ipython3
array = ak.Array(
    [
        [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
        [],
        [{"x": 3, "y": [3.0, 0.3, 3.3]}],
    ]
)
array
```

This example shows a templated C++ function that takes an Awkward Array and iterates over the list of records:

```ipython3
source_code = """
template<typename T>
double go_fast_cpp(T& awkward_array) {
    double out = 0.0;

    for (auto list : awkward_array) {
        for (auto record : list) {
            for (auto item : record.y()) {
                out += item;
            }
        }
    }

    return out;
}
"""

cppyy.cppdef(source_code)
```

The C++ type of an Awkward Array is a made-up type;
`awkward::ListArray_hyKwTH3lk1A`.

```ipython3
array.cpp_type
```

Awkward Arrays are dynamically typed, so in a C++ context, the type name is hashed. In practice, there is no need to know the type. The C++ code should use a placeholder type specifier `auto`. The type of the variable that is being declared will be automatically deduced from its initializer.

In a Python contexts, when a templated function requires a C++ type as a Python string, it can use the `ak.Array.cpp_type` property:

```ipython3
out = cppyy.gbl.go_fast_cpp[array.cpp_type](array)
```

```ipython3
%%timeit

out = cppyy.gbl.go_fast_cpp[array.cpp_type](array)
```

```myst-ansi
5.56 μs ± 31.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
```

```ipython3
%%timeit

ak.sum(array["y"])
```

```myst-ansi
151 μs ± 536 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

But the result is the same.

```ipython3
assert out == ak.sum(array["y"])
```
