# Getting started

## Installation

:::::{card} If you use pip, rip, pixi, or uv to install packages,

```bash
pip install awkward
```
:::::
:::::{card} If you use conda/mamba, it's in the conda-forge channel.

```bash
conda install -c conda-forge awkward
```
:::::

If you're installing as a developer or testing updates that haven't been released in a package manager yet, see the [developer installation instructions](https://github.com/scikit-hep/awkward/blob/main/CONTRIBUTING.md#building-and-testing-locally) in the [Contributor guide](https://github.com/scikit-hep/awkward/blob/main/CONTRIBUTING.md).

## Frequently asked questions

::::::::{grid} 1

:::::::{grid-item} 
::::::{dropdown} What is Awkward Array for? How does it compare to other libraries?

Python's builtin lists, dicts, and classes can be used to analyze arbitrary data structures, but at a cost in speed and memory. Therefore, they can't be used (easily) with large datasets.

[Pandas](https://pandas.pydata.org/) DataFrames (as well as [Polars](https://pola.rs/), [cuDF](https://docs.rapids.ai/api/cudf/stable/), and [Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html)) are well-suited to tabular data, including tables with relational indexes, but not arbitrary data structures. If a DataFrame is filled with Python's builtin types, then it offers no speed or memory advantage over Python itself.

[NumPy](https://numpy.org/) is ideal for rectangular arrays of numbers, but not arbitrary data structures. If a NumPy array is filled with Python's builtin types, then it offers no speed or memory advantage over Python itself.

[Apache Arrow](https://arrow.apache.org/) ([pyarrow](https://arrow.apache.org/docs/python/)) manages arrays of arbitrary data structures (including those in [Polars](https://pola.rs/), [cuDF](https://docs.rapids.ai/api/cudf/stable/), and to some extent, [Pandas](https://pandas.pydata.org/)), with great language interoperability and interprocess communication, but without manipulation functions oriented toward data analysts.

Awkward Array is a data analyst-friendly extension of NumPy-like idioms for arbitrary data structures. It is intended to be used interchangeably with NumPy and share data with Arrow and DataFrames. Like NumPy, it simplifies and accelerates computations that transform arrays into arrays—all computations over elements in an array are compiled. Also like NumPy, imperative-style computations can be accelerated with [Numba](https://numba.pydata.org/).

Note that there is also a [ragged](https://github.com/scikit-hep/ragged) array library with simpler (but still non-rectangular) data types that more closely adheres to [array APIs](https://data-apis.org/array-api/latest/API_specification).

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} Where is an Awkward Array's `shape` and `dtype`?

Since Awkward Arrays can contain arbitrary data structures, their type can't be separated into a `shape` and a `dtype`, the way a NumPy array can.

For an array of records like

```python
import awkward as ak

array = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

the `x` field contains floating point numbers and the `y` field contains lists of integers. They would have different `dtypes`, as well as different numbers of dimensions. This array also can't be separated into `x` and `y` columns with different `dtypes`, as in a DataFrame, since both fields are inside of records in a variable-length list.

Instead, Awkward Arrays have a `type`, which looks like

```python
3 * var * {x: float64, y: var * int64}
```

for the above. This combines `shape` and `dtype` information in the following way: the length of the array is `3`, the first dimension has `var` or variable length, it contains records with `x` and `y` field names in `{` `}`, the `x` field has `float64` primitive type and the `y` field is a `var` variable length list of `int64`. You can `print(array.type)` or `array.type.show()` to see the type of any `array`.

See the [ragged](https://github.com/scikit-hep/ragged) array library for variable-length dimensions that are nevertheless separable into a `shape` and `dtype`, like a conventional array.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} QUESTION

ANSWER

::::::
:::::::

::::::::

<br><br><br><br><br>








<!-- ### What are the most common questions new users have about the Awkward Array library in Python? -->

<!-- New users of the Awkward Array library in Python often have questions related to the following areas: -->

<!-- 1. **Basics and Getting Started**: -->
<!--    - What is Awkward Array, and how is it different from NumPy arrays? -->
<!--    - How do I install Awkward Array? -->

<!-- 2. **Creating and Manipulating Arrays**: -->
<!--    - How do I create an Awkward Array? -->
<!--    - How can I convert a NumPy array or a list into an Awkward Array? -->
<!--    - How do I access or modify elements in an Awkward Array? -->

<!-- 3. **Multidimensional and Nested Data**: -->
<!--    - How do I deal with nested data or arrays of variable length? -->
<!--    - How can I perform operations on nested data, like filtering or mapping? -->

<!-- 4. **Performance and Optimization**: -->
<!--    - Are there best practices for optimizing the performance of Awkward Array computations? -->
<!--    - How does Awkward Array handle memory management, especially with large datasets? -->

<!-- 5. **Interoperability**: -->
<!--    - How do I use Awkward Array with other libraries like pandas, NumPy, or PyTorch? -->
<!--    - Can I use Awkward Array with data formats like JSON or Parquet? -->

<!-- 6. **Saving and Loading**: -->
<!--    - How do I save and load Awkward Arrays to and from disk? -->
<!--    - What file formats are supported for serialization? -->

<!-- 7. **Advanced Features**: -->
<!--    - What are the special functions and capabilities of Awkward Array for complex data manipulation (like jagged arrays, etc.)? -->
<!--    - How do I work with high-dimensional or complex nested structures? -->

<!-- 8. **Troubleshooting and Error Handling**: -->
<!--    - What common errors might I encounter, and how can I troubleshoot them? -->
<!--    - How do I interpret error messages related to type mismatches or unsupported operations? -->

<!-- 9. **Community and Support**: -->
<!--    - Where can I find documentation and tutorials? -->
<!--    - How do I contribute to the Awkward Array project, or where do I report bugs? -->

<!-- Understanding these topics can significantly smooth the learning curve for new Awkward Array users, enabling them to leverage the library effectively for complex data manipulation and analysis tasks. -->

