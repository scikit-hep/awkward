# Thinking in arrays

*Originally presented as [part](https://github.com/jpivarski-talks/2023-12-18-hsf-india-tutorial-bhubaneswar/blob/main/lesson-3-awkward/lecture-slides.ipynb) of [HSF-India training on December 18, 2023](https://indico.cern.ch/event/1328624/).*

<br><br><br>

So far, all the arrays we’ve dealt with have been rectangular (in $n$ dimensions; “rectilinear”).

![](getting-started/8-layer_cube.jpg)

What if we had data like this?

```json
[
  [[1.84, 0.324]],
  [[-1.609, -0.713, 0.005], [0.953, -0.993, 0.011, 0.718]],
  [[0.459, -1.517, 1.545], [0.33, 0.292]],
  [[-0.376, -1.46, -0.206], [0.65, 1.278]],
  [[], [], [1.617]],
  []
]
[
  [[-0.106, 0.611]],
  [[0.118, -1.788, 0.794, 0.658], [-0.105]]
]
[
  [[-0.384], [0.697, -0.856]],
  [[0.778, 0.023, -1.455, -2.289], [-0.67], [1.153, -1.669, 0.305, 1.517, -0.292]]
]
[
  [[0.205, -0.355], [-0.265], [1.042]],
  [[-0.004], [-1.167, -0.054, 0.726, 0.213]],
  [[1.741, -0.199, 0.827]]
]
```

What if we had data like this?

```json
[
  {"fill": "#b1b1b1", "stroke": "none", "points": [{"x": 5.27453, "y": 1.03276},
    {"x": -3.51280, "y": 1.74849}]},
  {"fill": "#b1b1b1", "stroke": "none", "points": [{"x": 8.21630, "y": 4.07844},
    {"x": -0.79157, "y": 3.49478}, {"x": 16.38932, "y": 5.29399},
    {"x": 10.38641, "y": 0.10832}, {"x": -2.07070, "y": 14.07140},
    {"x": 9.57021, "y": -0.94823}, {"x": 1.97332, "y": 3.62380},
    {"x": 5.66760, "y": 11.38001}, {"x": 0.25497, "y": 3.39276},
    {"x": 3.86585, "y": 6.22051}, {"x": -0.67393, "y": 2.20572}]},
  {"fill": "#d0d0ff", "stroke": "none", "points": [{"x": 3.59528, "y": 7.37191},
    {"x": 0.59192, "y": 2.91503}, {"x": 4.02932, "y": -1.13601},
    {"x": -1.01593, "y": 1.95894}, {"x": 1.03666, "y": 0.05251}]},
  {"fill": "#d0d0ff", "stroke": "none", "points": [{"x": -8.78510, "y": -0.00497},
    {"x": -15.22688, "y": 3.90244}, {"x": 5.74593, "y": 4.12718}]},
  {"fill": "none", "stroke": "#000000", "points": [{"x": 4.40625, "y": -6.953125},
    {"x": 4.34375, "y": -7.09375}, {"x": 4.3125, "y": -7.140625},
    {"x": 4.140625, "y": -7.140625}]},
  {"fill": "none", "stroke": "#808080", "points": [{"x": 0.46875, "y": -0.09375},
    {"x": 0.46875, "y": -0.078125}, {"x": 0.46875, "y": 0.53125}]}
]
```

What if we had data like this?

```json
[
  {"movie": "Evil Dead", "year": 1981, "actors":
    ["Bruce Campbell", "Ellen Sandweiss", "Richard DeManincor", "Betsy Baker"]
  },
  {"movie": "Darkman", "year": 1900, "actors":
    ["Liam Neeson", "Frances McDormand", "Larry Drake", "Bruce Campbell"]
  },
  {"movie": "Army of Darkness", "year": 1992, "actors":
    ["Bruce Campbell", "Embeth Davidtz", "Marcus Gilbert", "Bridget Fonda",
     "Ted Raimi", "Patricia Tallman"]
  },
  {"movie": "A Simple Plan", "year": 1998, "actors":
    ["Bill Paxton", "Billy Bob Thornton", "Bridget Fonda", "Brent Briscoe"]
  },
  {"movie": "Spider-Man 2", "year": 2004, "actors":
    ["Tobey Maguire", "Kristen Dunst", "Alfred Molina", "James Franco",
     "Rosemary Harris", "J.K. Simmons", "Stan Lee", "Bruce Campbell"]
  },
  {"movie": "Drag Me to Hell", "year": 2009, "actors":
    ["Alison Lohman", "Justin Long", "Lorna Raver", "Dileep Rao", "David Paymer"]
  }
]
```

What if we had data like this?

```json
[
  {"run": 1, "luminosityBlock": 156, "event": 46501,
   "PV": {"x": 0.243, "y": 0.393, "z": 1.451},
   "electron": [],
   "muon": [
     {"pt": 63.043, "eta": -0.718, "phi": 2.968, "mass": 0.105, "charge": 1},
     {"pt": 38.120, "eta": -0.879, "phi": -1.032, "mass": 0.105, "charge": -1},
     {"pt": 4.048, "eta": -0.320, "phi": 1.038, "mass": 0.105, "charge": 1}
   ],
   "MET": {"pt": 21.929, "phi": -2.730}
  },
  {"run": 1, "luminosityBlock": 156, "event": 46502,
   "PV": {"x": 0.244, "y": 0.395, "z": -2.879},
   "electron": [
     {"pt": 21.902, "eta": -0.702, "phi": 0.133, "mass": 0.005, "charge": 1},
     {"pt": 42.632, "eta": -0.979, "phi": -1.863, "mass": 0.008, "charge": 1},
     {"pt": 78.012, "eta": -0.933, "phi": -2.207, "mass": 0.018, "charge": -1},
     {"pt": 23.835, "eta": -1.362, "phi": -0.621, "mass": 0.008, "charge": -1}
   ],
   "muon": [],
   "MET": {"pt": 16.972, "phi": 2.866}},
  ...
]
```

It might be possible to turn these datasets into tabular form using surrogate keys and database normalization, but

* they could be inconvenient or less efficient in that form, depending on what we want to do,
* they were very likely *given* in a ragged/untidy form. You can’t ignore the data-cleaning step!

<br>

Dealing with these datasets as JSON or Python objects is inefficient for the same reason as for lists of numbers.

<br>

We want arbitrary data structure with array-oriented interface and performance…

![](getting-started/awkward-motivation-venn-diagram.svg)

## Libraries for irregular arrays

<br>

![](getting-started/logo-arrow.svg)

```ipython3
import pyarrow as pa
```

<br>
```ipython3
arrow_array = pa.array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

<br>
```ipython3
arrow_array.type
```

<br>
```ipython3
arrow_array
```

<br>

![](getting-started/logo-awkward.svg)

```ipython3
import awkward as ak
```

<br>
```ipython3
awkward_array = ak.from_arrow(arrow_array)
awkward_array
```

<br>

![](getting-started/logo-parquet.svg)

```ipython3
ak.to_parquet(awkward_array, "/tmp/file.parquet")
```

<br>
```ipython3
ak.from_parquet("/tmp/file.parquet")
```

## Awkward Array

```ipython3
ragged = ak.Array([
    [
      [[1.84, 0.324]],
      [[-1.609, -0.713, 0.005], [0.953, -0.993, 0.011, 0.718]],
      [[0.459, -1.517, 1.545], [0.33, 0.292]],
      [[-0.376, -1.46, -0.206], [0.65, 1.278]],
      [[], [], [1.617]],
      []
    ],
    [
      [[-0.106, 0.611]],
      [[0.118, -1.788, 0.794, 0.658], [-0.105]]
    ],
    [
      [[-0.384], [0.697, -0.856]],
      [[0.778, 0.023, -1.455, -2.289], [-0.67], [1.153, -1.669, 0.305, 1.517, -0.292]]
    ],
    [
      [[0.205, -0.355], [-0.265], [1.042]],
      [[-0.004], [-1.167, -0.054, 0.726, 0.213]],
      [[1.741, -0.199, 0.827]]
    ]
])
```

**Multidimensional indexing**

```ipython3
ragged[3, 1, -1, 2]
```

<br>

**Basic slicing**

```ipython3
ragged[3, 1:, -1, 1:3]
```

<br>

**Advanced slicing**

```ipython3
ragged[[False, False, True, True], [0, -1, 0, -1], 0, -1]
```

**Awkward slicing**

```ipython3
ragged > 0
```

<br>
```ipython3
ragged[ragged > 0]
```

**Reductions**

```ipython3
ak.sum(ragged)
```

<br>
```ipython3
ak.sum(ragged, axis=-1)
```

<br>
```ipython3
ak.sum(ragged, axis=0)
```

How do we even define reductions on an array with variable length lists?

![](getting-started/example-reducer-2d.svg)

How do we even define reductions on an array with variable length lists?

![](getting-started/example-reducer-ragged.svg)

```ipython3
array = ak.Array([[   1,    2,    3,    4],
                  [  10, None,   30      ],
                  [ 100,  200            ]])
```

<br>
```ipython3
ak.sum(array, axis=0).tolist()
```

<br>
```ipython3
ak.sum(array, axis=1).tolist()
```

<br>

(You almost always want the deepest/maximum `axis`, which you can get with `axis=-1`.)

<br>

### Awkward Arrays in particle physics

```ipython3
import uproot

file = uproot.open("https://github.com/jpivarski-talks/2023-12-18-hsf-india-tutorial-bhubaneswar/raw/main/data/SMHiggsToZZTo4L.root")
file
```

<br>
```ipython3
tree = file["Events"]
tree
```

<br>
```ipython3
tree.arrays(entry_stop=100)
```

The same data fits into Parquet files (a little more easily).

```ipython3
events = ak.from_parquet("https://github.com/jpivarski-talks/2023-12-18-hsf-india-tutorial-bhubaneswar/raw/main/data/SMHiggsToZZTo4L.parquet")
events
```

View the first event as Python lists and dicts (like JSON).

```ipython3
events[0].to_list()
```

Get one numeric field (also known as “column”).

```ipython3
events.electron.pt
```

Compute something ($p_z = p_T \\sinh\\eta$).

```ipython3
import numpy as np

events.electron.pt * np.sinh(events.electron.eta)
```

Note that the Vector library works with Awkward Arrays, if it is imported this way:

```ipython3
import vector
vector.register_awkward()
```

<br>

Records with `name="Momentum4D"` and fields with coordinate names (`px`, `py`, `pz`, `E` or `pt`, `phi`, `eta`, `m`) automatically get Vector properties and methods.

<br>
```ipython3
events.electron.type.show()
```

```myst-ansi
299973 * var * Momentum4D[
    pt: float32,
    eta: float32,
    phi: float32,
    mass: float32,
    charge: int32,
    pfRelIso03_all: float32,
    dxy: float32,
    dxyErr: float32,
    dz: float32,
    dzErr: float32
]
```

<br>
```ipython3
# implicitly computes pz = pt * sinh(eta)
events.electron.pz
```

To make histograms or other plots, we need numbers without structure, so [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten) the array.

```ipython3
from hist import Hist

Hist.new.Regular(100, 0, 100, name=" ").Double().fill(
    ak.flatten(events.electron.pt)
).plot();
```

Each event has a different number of electrons and muons ([`ak.num()`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num) to check).

```ipython3
ak.num(events.electron), ak.num(events.muon)
```

<br>

So what happens if we try to compute something with the electrons’ $p_T$ and the muons’ $\\eta$?

```ipython3
events.electron.pt * np.sinh(events.muon.eta)
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[33], line 1
----> 1 events.electron.pt * np.sinh(events.muon.eta)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_operators.py:52, in _binary_method.<locals>.func(self, other)
     49 if _disables_array_ufunc(other):
     50     return NotImplemented
---> 52 return ufunc(self, other)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1644, in Array.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)
   1579 """
   1580 Intercepts attempts to pass this Array to a NumPy
   1581 [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
   (...)   1641 See also #__array_function__.
   1642 """
   1643 name = f"{type(ufunc).__module__}.{ufunc.__name__}.{method!s}"
-> 1644 with ak._errors.OperationErrorContext(name, inputs, kwargs):
   1645     return ak._connect.numpy.array_ufunc(ufunc, method, inputs, kwargs)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_errors.py:79, in ErrorContext.__exit__(self, exception_type, exception_value, traceback)
     77     self._slate.__dict__.clear()
     78     # Handle caught exception
---> 79     raise self.decorate_exception(exception_type, exception_value)
     80 else:
     81     # Step out of the way so that another ErrorContext can become primary.
     82     if self.primary() is self:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1645, in Array.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)
   1643 name = f"{type(ufunc).__module__}.{ufunc.__name__}.{method!s}"
   1644 with ak._errors.OperationErrorContext(name, inputs, kwargs):
-> 1645     return ak._connect.numpy.array_ufunc(ufunc, method, inputs, kwargs)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_connect/numpy.py:484, in array_ufunc(ufunc, method, inputs, kwargs)
    476         raise TypeError(
    477             "no {}.{} overloads for custom types: {}".format(
    478                 type(ufunc).__module__, ufunc.__name__, ", ".join(error_message)
    479             )
    480         )
    482     return None
--> 484 out = ak._broadcasting.broadcast_and_apply(
    485     inputs,
    486     action,
    487     depth_context=depth_context,
    488     lateral_context=lateral_context,
    489     allow_records=False,
    490     function_name=ufunc.__name__,
    491 )
    493 out_named_axis = functools.reduce(
    494     _unify_named_axis, lateral_context[NAMED_AXIS_KEY].named_axis
    495 )
    496 if len(out) == 1:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1223, in broadcast_and_apply(inputs, action, depth_context, lateral_context, allow_records, left_broadcast, right_broadcast, numpy_to_regular, regular_to_jagged, function_name, broadcast_parameters_rule)
   1221 backend = backend_of(*inputs, coerce_to_common=False)
   1222 isscalar = []
-> 1223 out = apply_step(
   1224     backend,
   1225     broadcast_pack(inputs, isscalar),
   1226     action,
   1227     0,
   1228     depth_context,
   1229     lateral_context,
   1230     {
   1231         "allow_records": allow_records,
   1232         "left_broadcast": left_broadcast,
   1233         "right_broadcast": right_broadcast,
   1234         "numpy_to_regular": numpy_to_regular,
   1235         "regular_to_jagged": regular_to_jagged,
   1236         "function_name": function_name,
   1237         "broadcast_parameters_rule": broadcast_parameters_rule,
   1238     },
   1239 )
   1240 assert isinstance(out, tuple)
   1241 return tuple(broadcast_unpack(x, isscalar) for x in out)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1201, in apply_step(backend, inputs, action, depth, depth_context, lateral_context, options)
   1199     return result
   1200 elif result is None:
-> 1201     return continuation()
   1202 else:
   1203     raise AssertionError(result)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1170, in apply_step.<locals>.continuation()
   1168 # Any non-string list-types?
   1169 elif any(x.is_list and not is_string_like(x) for x in contents):
-> 1170     return broadcast_any_list()
   1172 # Any RecordArrays?
   1173 elif any(x.is_record for x in contents):

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:663, in apply_step.<locals>.broadcast_any_list()
    660         nextinputs.append(x)
    661         nextparameters.append(NO_PARAMETERS)
--> 663 outcontent = apply_step(
    664     backend,
    665     nextinputs,
    666     action,
    667     depth + 1,
    668     copy.copy(depth_context),
    669     lateral_context,
    670     options,
    671 )
    672 assert isinstance(outcontent, tuple)
    673 parameters = parameters_factory(nextparameters, len(outcontent))

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1201, in apply_step(backend, inputs, action, depth, depth_context, lateral_context, options)
   1199     return result
   1200 elif result is None:
-> 1201     return continuation()
   1202 else:
   1203     raise AssertionError(result)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1170, in apply_step.<locals>.continuation()
   1168 # Any non-string list-types?
   1169 elif any(x.is_list and not is_string_like(x) for x in contents):
-> 1170     return broadcast_any_list()
   1172 # Any RecordArrays?
   1173 elif any(x.is_record for x in contents):

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:714, in apply_step.<locals>.broadcast_any_list()
    710 for i, ((named_axis, ndim), x, x_is_string) in enumerate(
    711     zip(named_axes_with_ndims, inputs, input_is_string, strict=True)
    712 ):
    713     if isinstance(x, listtypes) and not x_is_string:
--> 714         next_content = broadcast_to_offsets_avoiding_carry(x, offsets)
    715         nextinputs.append(next_content)
    716         nextparameters.append(x._parameters)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:373, in broadcast_to_offsets_avoiding_carry(list_content, offsets)
    371         return list_content.content[:next_length]
    372     else:
--> 373         return list_content._broadcast_tooffsets64(offsets).content
    374 elif isinstance(list_content, ListArray):
    375     # Is this list contiguous?
    376     if nplike.array_equal(
    377         list_content.starts.data[1:], list_content.stops.data[:-1]
    378     ):
    379         # Does this list match the offsets?

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/listoffsetarray.py:439, in ListOffsetArray._broadcast_tooffsets64(self, offsets)
    434     next_content = self._content[this_start:]
    436 if nplike.known_data and not nplike.array_equal(
    437     this_zero_offsets, offsets.data
    438 ):
--> 439     raise ValueError("cannot broadcast nested list")
    441 return ListOffsetArray(
    442     offsets, next_content[: offsets[-1]], parameters=self._parameters
    443 )

ValueError: cannot broadcast nested list

This error occurred while calling

    numpy.multiply.__call__(
        <Array [[], [21.9, ...], ..., [48.1, 38.7]] type='299973 * var * fl...'>
        <Array [[-0.782, -0.997, -0.326], ..., []] type='299973 * var * flo...'>
    )
```

This is data structure-aware, array-oriented programming.

**Application:** Filtering events with an array of booleans.

```ipython3
events.MET.pt, events.MET.pt > 20
```

<br>
```ipython3
len(events), len(events[events.MET.pt > 20])
```

<br>

**Application:** Filtering particles with an array of lists of booleans.

```ipython3
events.electron.pt, events.electron.pt > 30
```

<br>
```ipython3
ak.num(events.electron), ak.num(events.electron[events.electron.pt > 30])
```

**Quizlet:** Using the reducer [`ak.any()`](sphinx-llm:df7c468b43ca4558ab741d771ffbf0dc#ak.any), how would we select *events* in which any electron has $p_T > 30$ GeV/c$^2$?

```ipython3
events.electron[events.electron.pt > 30]
```

<br>

Awkward Array has two combinatorial primitives:

[`ak.cartesian()`](sphinx-llm:e515482efbae4eb28bb9f06e625d21c4#ak.cartesian) takes a [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) of lists from $N$ different arrays, producing an array of lists of $N$-tuples.

[`ak.combinations()`](sphinx-llm:e2edae471dd3475f90dd9ae86f4e03a3#ak.combinations) takes $N$ [samples without replacement](http://prob140.org/sp18/textbook/notebooks-md/5_04_Sampling_Without_Replacement.html) of lists from a single array, producing an array of lists of $N$-tuples.

```ipython3
numbers = ak.Array([[1, 2, 3], [], [4]])
letters = ak.Array([["a", "b"], ["c"], ["d", "e"]])
```

<br>
```ipython3
ak.cartesian([numbers, letters])
```

<br>
```ipython3
values = ak.Array([[1.1, 2.2, 3.3, 4.4], [], [5.5, 6.6]])
```

<br>
```ipython3
ak.combinations(values, 2)
```

Often, it’s useful to separate the separate the left-hand sides and right-hand sides of these pairs with [`ak.unzip()`](sphinx-llm:a253bd50406941cf8e244c3a95347059#ak.unzip), so they can be used in mathematical expressions.

<br>
```ipython3
electron_muon_pairs = ak.cartesian([events.electron, events.muon])
electron_muon_pairs.type.show()
```

```myst-ansi
299973 * var * (
    Momentum4D[
        pt: float32,
        eta: float32,
        phi: float32,
        mass: float32,
        charge: int32,
        pfRelIso03_all: float32,
        dxy: float32,
        dxyErr: float32,
        dz: float32,
        dzErr: float32
    ],
    Momentum4D[
        pt: float32,
        eta: float32,
        phi: float32,
        mass: float32,
        charge: int32,
        pfRelIso03_all: float32,
        pfRelIso04_all: float32,
        dxy: float32,
        dxyErr: float32,
        dz: float32,
        dzErr: float32
    ]
)
```

<br>
```ipython3
electron_in_pair, muon_in_pair = ak.unzip(electron_muon_pairs)
electron_in_pair.type.show()
```

```myst-ansi
299973 * var * Momentum4D[
    pt: float32,
    eta: float32,
    phi: float32,
    mass: float32,
    charge: int32,
    pfRelIso03_all: float32,
    dxy: float32,
    dxyErr: float32,
    dz: float32,
    dzErr: float32
]
```

<br>
```ipython3
electron_in_pair.pt, muon_in_pair.pt
```

<br>
```ipython3
ak.num(electron_in_pair), ak.num(muon_in_pair)
```

To use Vector’s `deltaR` method ($\\Delta R = \\sqrt{\\Delta\\phi^2 + \\Delta\\eta^2}$), we need to have the electrons and muons in separate arrays.

```ipython3
electron_in_pair, muon_in_pair = ak.unzip(ak.cartesian([events.electron, events.muon]))
```

<br>
```ipython3
electron_in_pair.deltaR(muon_in_pair)
```

```ipython3
first_electron_in_pair, second_electron_in_pair = ak.unzip(ak.combinations(events.electron, 2))
```

<br>
```ipython3
first_electron_in_pair.deltaR(second_electron_in_pair)
```

**Quizlet:** What’s this?

```ipython3
(first_electron_in_pair + second_electron_in_pair).mass
```

```ipython3
Hist.new.Reg(120, 0, 120, name="mass (GeV)").Double().fill(
    ak.flatten((first_electron_in_pair + second_electron_in_pair).mass, axis=-1)
).plot();
```
