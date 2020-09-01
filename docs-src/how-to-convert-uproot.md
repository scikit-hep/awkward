---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.10'
    jupytext_version: 1.5.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to convert to/from ROOT with Uproot
=======================================

Uproot defaults to reading data as Awkward Arrays, so there usually isn't any extra work to do. But there are caveats, mostly with the legacy version of Uproot (Uproot 3).

To find out which version you're using:

   * if you `import uproot` and `uproot.__version__` starts with `"3."`, then it's Uproot 3;
   * if you `import uproot` and `uproot.__version__` starts with `"4."`, then it's Uproot 4;
   * if you `import uproot3`, then it's Uproot 3;
   * if you `import uproot4`, then it's Uproot 4.

```{code-cell} ipython3
import awkward1 as ak
import numpy as np
import uproot
import uproot4
```

From ROOT to Awkward with Uproot 4
----------------------------------

By default, Uproot 4 delivers data from ROOT files as Awkward 1 arrays, even though the Awkward library isn't one of Uproot's formal dependencies. (If you try to use Uproot 4 without having Awkward, you'll quickly be presented with an ImportError and suggestions about how to proceed.)

To start, open a file and look at the objects it contains.

```{code-cell} ipython3
up4_file = uproot4.open("http://scikit-hep.org/uproot/examples/HZZ.root")
up4_file.classnames()
```

From the above, we learn that `"events"` is a TTree, so we read its metadata with:

```{code-cell} ipython3
up4_events = up4_file["events"]
up4_events
```

And then look at its branches and their types.

```{code-cell} ipython3
up4_events.show()
```

Some of these branches have a single value per event (e.g. `"MET_px"` has type `float`) and some have multiple values per event (e.g. `"Muon_Px"` has type  `float[]`).

Regardless of type, they would all be returned as Awkward Arrays:

```{code-cell} ipython3
array = up4_events["MET_px"].array()
array
```

```{code-cell} ipython3
type(array)
```

```{code-cell} ipython3
array = up4_events["Muon_Px"].array()
array
```

```{code-cell} ipython3
type(array)
```

Because `library="ak"` is the default. Setting it to another value, like `library="np"`, returns non-Awkward arrays.

```{code-cell} ipython3
array = up4_events["MET_px"].array(library="np")
array
```

```{code-cell} ipython3
type(array)
```

```{code-cell} ipython3
array = up4_events["Muon_Px"].array(library="np")
array
```

```{code-cell} ipython3
type(array)
```

Uproot's `arrays` method (plural) returns a "package" of related arrays, which for the Awkward library means arrays presented as records.

```{code-cell} ipython3
arrays = up4_events.arrays(["Muon_Px", "Muon_Py", "Muon_Pz"])
arrays
```

```{code-cell} ipython3
ak.type(arrays)
```

```{code-cell} ipython3
arrays["Muon_Px"]
```

With `arrays`, the `how="zip"` option attempts to [ak.zip](https://awkward-array.readthedocs.io/en/latest/_auto/ak.zip.html) lists with common list lengths.

Note that the [ak.type](https://awkward-array.readthedocs.io/en/latest/_auto/ak.type.html) below is `var * {all fields}`, rather than `{field: var, field: var, ...}`.

```{code-cell} ipython3
arrays = up4_events.arrays(["Muon_Px", "Muon_Py", "Muon_Pz"], how="zip")
arrays
```

```{code-cell} ipython3
ak.type(arrays)
```

```{code-cell} ipython3
arrays.Muon.Px
```

If some of the branches cannot be combined because they have different multiplicities, they are kept separate.

```{code-cell} ipython3
arrays = up4_events.arrays(["Muon_Px", "Muon_Py", "Muon_Pz", "Jet_Px", "Jet_Py", "Jet_Pz"], how="zip")
arrays
```

```{code-cell} ipython3
ak.type(arrays)
```

```{code-cell} ipython3
arrays.Muon.Px
```

```{code-cell} ipython3
arrays.Jet.Px
```

From Awkward to ROOT with Uproot 4
----------------------------------

**Not implemented yet:** see the bottom of this page for writing files with Uproot 3.

+++

From ROOT to Awkward with Uproot 3
----------------------------------

Some of the arrays returned by Uproot 3 are NumPy arrays and others are Awkward 0 (i.e. "old library") arrays, depending on whether Awkward is needed.

Uproot 4 is recommended unless you're dealing with legacy software built on Uproot 3.

To start, open a file and look at the objects it contains.

```{code-cell} ipython3
up3_file = uproot.open("http://scikit-hep.org/uproot/examples/HZZ.root")
up3_file.classnames()
```

From the above, we learn that `"events"` is a TTree, so we read its metadata with:

```{code-cell} ipython3
up3_events = up3_file["events"]
up3_events
```

And then look at its branches and their types.

```{code-cell} ipython3
up3_events.show()
```

Some of these branches have a single value per event (e.g. `"MET_px"` has interpretation `asdtype('>f4')`) and some have multiple values per event (e.g. `"Muon_Px"` has interpretation `asjagged(asdtype('>f4'))`).

Data that can be interpreted `asdtype` are returned as NumPy arrays:

```{code-cell} ipython3
array = up3_events.array("MET_px")
array
```

```{code-cell} ipython3
type(array)
```

NumPy arrays can be converted to Awkward 1 (i.e. "new library") by passing them to the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) constructor or the [ak.from_numpy](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_numpy.html) function.

```{code-cell} ipython3
ak.Array(array)
```

And data that require `asjagged` or other specialized interpretations are returned as Awkward Arrays:

```{code-cell} ipython3
array = up3_events.array("Muon_Px")
array
```

```{code-cell} ipython3
type(array)
```

Awkward 0 arrays can be converted to Awkward 1 by passing them to the [ak.from_awkward0](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_awkward0.html) function. (There's also an [ak.to_awkward0](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_awkward0.html) for the other direction; conversions are usually zero-copy and quick.)

```{code-cell} ipython3
ak.from_awkward0(array)
```

(Unlike Uproot 4, there isn't a way to specify which library you want for returning output.)

+++

Uproot 3's `arrays` method (plural) returns "packages" of related arrays as Python dicts:

```{code-cell} ipython3
arrays = up3_events.arrays(["Muon_Px", "Muon_Py", "Muon_Pz"])
arrays
```

Be careful of the bytestring keys (dict key type is `bytes`, rather than `str`) and note that you can only convert all of the arrays with a loop: they are separate entities.

```{code-cell} ipython3
{name.decode(): ak.from_awkward0(array) for name, array in arrays.items()}
```

From Awkward to ROOT with Uproot 3
----------------------------------

Since ROOT file-writing is only implemented in Uproot 3, you'll need to take into consideration whether an array is flat, and therefore NumPy, or jagged, and therefore Awkward 0 (i.e. "old library").

To open a flie for writing, use `uproot.recreate`, rather than `uproot.open`.

```{code-cell} ipython3
file = uproot.recreate("/tmp/example.root")
file
```

The `uproot.newtree` function creates a tree that can be written. The data types for each branch have to be specified.

```{code-cell} ipython3
file["tree1"] = uproot.newtree({"branch1": int, "branch2": np.float32})
```

The method for writing is `extend`, which can be called as many times as needed to write array chunks to the file.

The chunks should be large (each represents a ROOT TBasket) and must include equal-length arrays for each branch.

```{code-cell} ipython3
file["tree1"].extend({"branch1": np.array([0, 1, 2, 3, 4]),
                      "branch2": np.array([0.0, 1.1, 2.2, 3.3, 4.4], dtype=np.float32)})
```

```{code-cell} ipython3
file["tree1"].extend({"branch1": np.array([5, 6, 7, 8, 9]),
                      "branch2": np.array([5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float32)})
```

To write a jagged array, it must be in Awkward 0 format. You may need to use [ak.to_awkward0](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_awkward0.html).

```{code-cell} ipython3
ak0_array = ak.to_awkward0(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
ak0_array
```

And you will need its `counts`. (This is the Awkward 0 equivalent of [ak.num](https://awkward-array.readthedocs.io/en/latest/_auto/ak.num.html).

```{code-cell} ipython3
ak0_array.counts
```

The branch's type has to be constructed with the `uproot.newbranch` function and has to include a `size`, into which the counts will be written.

```{code-cell} ipython3
file["tree2"] = uproot.newtree({"branch3": uproot.newbranch(np.dtype("f8"), size="n")})
```

Fill each chunk by assigning the branch data and the counts in each `extend`.

```{code-cell} ipython3
file["tree2"].extend({"branch3": ak0_array, "n": ak0_array.counts})
```

File-closure could also be enforced by putting `uproot.recreate` in a context manager (Python `with` statement).

```{code-cell} ipython3
file.close()
```
