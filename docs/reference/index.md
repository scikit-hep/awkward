# API reference

See the left side-bar (or bring it into view by clicking on the upper-left `≡`) for a detailed description of every public class and function in Awkward Array. You can use your browser's find-in-page to find a particular function.

In a nutshell, the `awkward` library consists of

* a high-level {obj}`ak.Array` class, as well as {obj}`ak.Record` for scalar records,
* a suite of functions in the `ak.*` and `ak.str.*` namespaces, which operate on arrays,
* high-level data {obj}`ak.types.Type` classes, a generalization of NumPy's shape and dtype,
* low-level array {obj}`ak.contents.Content`, which describe the memory layout of arrays, as well as their {obj}`ak.forms.Form` (low-level types),
* an {obj}`ak.behavior` dict to add functionality to arrays and records.

For details about array slicing, see {func}`ak.Array.__getitem__`.

For details about adding record fields to an array of records, see {func}`ak.Array.__setitem__`.

To get a low-level {obj}`ak.contents.Content` from an array or record, see {obj}`ak.Array.layout` and {obj}`ak.Record.layout`.

If you're looking for "how to..." guides arranged by task, rather than function, see the user guide instead.

You can test any of these functions in a new window/tab by clicking on [![Try It! ⭷](https://img.shields.io/badge/-Try%20It%21%20%E2%86%97-orange?style=for-the-badge)](https://awkward-array.org/doc/main/_static/try-it.html).

<br><br><br><br><br>

:::{card} C++ Documentation {fas}`external-link-alt`
:link: ../_static/doxygen/index.html

The C++ code implementing the `awkward-cpp` helper library are documented separately. Click here to go to the C++ API reference.
:::

:::{card} dask-awkward {fas}`external-link-alt`
:link: https://dask-awkward.readthedocs.io/

Although many of the functions have the same names and interfaces, the `dask-awkward` library is documented separately. Click here to learn about Awkward Arrays in Dask.
:::

```{eval-rst}
.. include:: toctree.txt
```
