<a href="https://github.com/scikit-hep/awkward-1.0#readme"><img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/logo/logo-300px.png"></a>

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they're not.

<table>
  <tr>
    <td width="66%" valign="top">
      <a href="https://awkward-array.org">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-tutorials.png" width="570">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.org">
        How-to tutorials
        </a>
      </b></p>
    </td>
    <td width="33%" valign="top">
      <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-sphinx.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        Python API reference
        </a>
      </b></p>
      <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-doxygen.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        C++ API reference
        </a>
      </b></p>
    </td>
  </tr>
</table>

# Motivating example

Given an array of objects with `x`, `y` fields and variable-length nested lists like

```python
array = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": {1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

the following slices out the `y` values, drops the first element from each inner list, and runs NumPy's `np.square` function on everything that is left:

```python
output = np.square(array["y", ..., 1:])
```

The result is

```python
[
    [[], [4], [4, 9]],
    [],
    [[4, 9, 16], [4, 9, 16, 25]]
]
```

The equivalent using only Python is

```python
output = []
for sublist in array:
    tmp1 = []
    for record in sublist:
        tmp2 = []
        for number in record["y"][1:]:
            tmp2.append(np.square(number))
        tmp1.append(tmp2)
    output.append(tmp1)
```

Not only is the expression using Awkward Arrays more concise, using idioms familiar from NumPy, but it's much faster and uses less memory.

For a similar problem 10 million times larger than the one above (on a single-threaded 2.2 GHz processor),

   * the Awkward Array one-liner takes **4.6 seconds** to run and uses **2.1 GB** of memory,
   * the equivalent using Python lists and dicts takes **138 seconds** to run and uses **22 GB** of memory.

Speed and memory factors in the double digits are common because we're replacing Python's dynamically typed, pointer-chasing virtual machine with type-specialized, precompiled routines on contiguous data. (In other words, for the same reasons as NumPy.) Even higher speedups are possible when Awkward Array is paired with [Numba](https://numba.pydata.org/).

Our [presentation at SciPy 2020](https://youtu.be/WlnUF3LRBj4) provides a good introduction, showing how to use these arrays in a real analysis.

# Installation

Awkward Array can be installed [from PyPI](https://pypi.org/project/awkward1/) using pip:

```bash
pip install awkward1
```

Most users will get a precompiled binary (wheel) for your operating system and Python version. If not, the above attempts to compile from source.

   * Report bugs, request features, and ask for additional documentation on [GitHub Issues](https://github.com/scikit-hep/awkward-1.0/issues). If you have a general "How do I…?" question, we'll answer it as a new [example in the tutorial](https://awkward-array.org/how-to.html).
   * If you have a problem that's too specific to be new documentation or it isn't exclusively related to Awkward Array, it might be more appropriate to ask on [StackOverflow with the [awkward-array] tag](https://stackoverflow.com/questions/tagged/awkward-array). Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.
   * The [Gitter Scikit-HEP/community](https://gitter.im/Scikit-HEP/community) is a way to get in touch with all Scikit-HEP developers and users.
