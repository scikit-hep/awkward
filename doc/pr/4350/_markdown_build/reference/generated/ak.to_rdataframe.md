# ak.to_rdataframe

Defined in [awkward.operations.ak_to_rdataframe](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_rdataframe.py) on [line 16](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_rdataframe.py#L16).

#### ak.to_rdataframe(arrays, \*, flatlist_as_rvec=True)

Converts an Awkward Array into ROOT RDataFrame columns.

See also [`ak.from_rdataframe`](sphinx-llm:aeb870e103be45fc91e3cba7692297ff#ak.from_rdataframe).

* **Parameters:**
  * **arrays** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *of* *arrays*) – Each value in this dict can be any array-like data
    that [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes, but they must all have the same length.
  * **flatlist_as_rvec** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, lists of primitive types (numbers, booleans, etc.)
    are presented to C++ as `ROOT::RVec<primitive>`, but all other types use
    Awkward Array’s custom C++ classes. If False, even these “flat” lists use
    Awkward Array’s custom C++ classes.
* **Returns:**
  A ROOT RDataFrame whose columns are the fields of `array`.

### Examples

```pycon
>>> x = ak.Array([
...     [1.1, 2.2, 3.3],
...     [],
...     [4.4, 5.5],
... ])
>>> y = ak.Array([
...     {"a": 1.1, "b": [1]},
...     {"a": 2.2, "b": [2, 1]},
...     {"a": 3.3, "b": [3, 2, 1]},
... ])
```

```pycon
>>> rdf = ak.to_rdataframe({"x": x, "y": y})
>>> rdf.Define("z", "ROOT::VecOps::Sum(x) + y.a() + y.b()[0]").AsNumpy(["z"])
{'z': ndarray([ 8.7,  4.2, 16.2])}
```

```pycon
>>> ak.sum(x, axis=-1) + y.a + y.b[:, 0]
<Array [8.7, 4.2, 16.2] type='3 * float64'>
```
