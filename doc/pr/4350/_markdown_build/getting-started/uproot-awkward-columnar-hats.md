# Uproot Awkward Columnar HATS

*Originally presented as [part](https://github.com/jpivarski-talks/2021-06-14-uproot-awkward-columnar-hats/blob/main/3-awkward-array.ipynb) of [CMS HATS training on June 14, 2021](https://indico.cern.ch/event/1042866/).*

<br><br><br><br><br>

## What about an array of lists?

```ipython3
import skhep_testdata
import awkward as ak
import numpy as np
import uproot
```

```ipython3
events = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"]
events.show()
```

```myst-ansi
name                 | typename                 | interpretation                
---------------------+--------------------------+-------------------------------
NJet                 | int32_t                  | AsDtype('>i4')
Jet_Px               | float[]                  | AsJagged(AsDtype('>f4'))
Jet_Py               | float[]                  | AsJagged(AsDtype('>f4'))
Jet_Pz               | float[]                  | AsJagged(AsDtype('>f4'))
Jet_E                | float[]                  | AsJagged(AsDtype('>f4'))
Jet_btag             | float[]                  | AsJagged(AsDtype('>f4'))
Jet_ID               | bool[]                   | AsJagged(AsDtype('bool'))
NMuon                | int32_t                  | AsDtype('>i4')
Muon_Px              | float[]                  | AsJagged(AsDtype('>f4'))
Muon_Py              | float[]                  | AsJagged(AsDtype('>f4'))
Muon_Pz              | float[]                  | AsJagged(AsDtype('>f4'))
Muon_E               | float[]                  | AsJagged(AsDtype('>f4'))
Muon_Charge          | int32_t[]                | AsJagged(AsDtype('>i4'))
Muon_Iso             | float[]                  | AsJagged(AsDtype('>f4'))
NElectron            | int32_t                  | AsDtype('>i4')
Electron_Px          | float[]                  | AsJagged(AsDtype('>f4'))
Electron_Py          | float[]                  | AsJagged(AsDtype('>f4'))
Electron_Pz          | float[]                  | AsJagged(AsDtype('>f4'))
Electron_E           | float[]                  | AsJagged(AsDtype('>f4'))
Electron_Charge      | int32_t[]                | AsJagged(AsDtype('>i4'))
Electron_Iso         | float[]                  | AsJagged(AsDtype('>f4'))
NPhoton              | int32_t                  | AsDtype('>i4')
Photon_Px            | float[]                  | AsJagged(AsDtype('>f4'))
Photon_Py            | float[]                  | AsJagged(AsDtype('>f4'))
Photon_Pz            | float[]                  | AsJagged(AsDtype('>f4'))
Photon_E             | float[]                  | AsJagged(AsDtype('>f4'))
Photon_Iso           | float[]                  | AsJagged(AsDtype('>f4'))
MET_px               | float                    | AsDtype('>f4')
MET_py               | float                    | AsDtype('>f4')
MChadronicBottom_px  | float                    | AsDtype('>f4')
MChadronicBottom_py  | float                    | AsDtype('>f4')
MChadronicBottom_pz  | float                    | AsDtype('>f4')
MCleptonicBottom_px  | float                    | AsDtype('>f4')
MCleptonicBottom_py  | float                    | AsDtype('>f4')
MCleptonicBottom_pz  | float                    | AsDtype('>f4')
MChadronicWDecayQ... | float                    | AsDtype('>f4')
MChadronicWDecayQ... | float                    | AsDtype('>f4')
MChadronicWDecayQ... | float                    | AsDtype('>f4')
MChadronicWDecayQ... | float                    | AsDtype('>f4')
MChadronicWDecayQ... | float                    | AsDtype('>f4')
MChadronicWDecayQ... | float                    | AsDtype('>f4')
MClepton_px          | float                    | AsDtype('>f4')
MClepton_py          | float                    | AsDtype('>f4')
MClepton_pz          | float                    | AsDtype('>f4')
MCleptonPDGid        | int32_t                  | AsDtype('>i4')
MCneutrino_px        | float                    | AsDtype('>f4')
MCneutrino_py        | float                    | AsDtype('>f4')
MCneutrino_pz        | float                    | AsDtype('>f4')
NPrimaryVertices     | int32_t                  | AsDtype('>i4')
triggerIsoMu24       | bool                     | AsDtype('bool')
EventWeight          | float                    | AsDtype('>f4')
```

```ipython3
events["Muon_Px"].array()
```

```ipython3
events["Muon_Px"].array(entry_stop=20).tolist()
```

This is what Awkward Array was made for. NumPy’s equivalent is cumbersome and inefficient.

```ipython3
jagged_numpy = events["Muon_Px"].array(entry_stop=20, library="np")
jagged_numpy
```

What if I want the first item in each list as an array?

```ipython3
np.array([x[0] for x in jagged_numpy])
```

This violates the rule from [1-python-performance.ipynb](https://github.com/jpivarski-talks/2021-06-14-uproot-awkward-columnar-hats/blob/main/1-python-performance.ipynb): don’t iterate in Python.

```ipython3
jagged_awkward = events["Muon_Px"].array(entry_stop=20, library="ak")
jagged_awkward
```

```ipython3
jagged_awkward[:, 0]
```

<br><br><br><br><br>

## Awkward Array is a general-purpose library: NumPy-like idioms on JSON-like data

![](getting-started/pivarski-one-slide-summary.svg)

<br><br><br><br><br>

## Main idea: slicing through structure is computationally inexpensive

Slicing by field name doesn’t modify any large buffers and [ak.zip](https://awkward-array.readthedocs.io/en/latest/_auto/ak.zip.html) only scans them to ensure they’re compatible (not even that if `depth_limit=1`).

```ipython3
array = events.arrays()
array
```

Think of this as zero-cost:

```ipython3
array.Muon_Px, array.Muon_Py, array.Muon_Pz
```

Think of this as zero-cost:

```ipython3
ak.zip({"px": array.Muon_Px, "py": array.Muon_Py, "pz": array.Muon_Pz})
```

(The above is a manual version of `how="zip"`.)

<br><br><br>

NumPy ufuncs work on these arrays (if they’re “[broadcastable](https://awkward-array.readthedocs.io/en/latest/_auto/ak.broadcast_arrays.html)”).

```ipython3
np.sqrt(array.Muon_Px**2 + array.Muon_Py**2)
```

<br><br><br>

And there are specialized operations that only make sense in a variable-length context.

[`ak.cartesian()`](sphinx-llm:e515482efbae4eb28bb9f06e625d21c4#ak.cartesian)

![](getting-started/cartoon-cartesian.png)

[`ak.combinations()`](sphinx-llm:e2edae471dd3475f90dd9ae86f4e03a3#ak.combinations)

![](getting-started/cartoon-combinations.png)

```ipython3
ak.cartesian((array.Muon_Px, array.Jet_Px))
```

```ipython3
ak.combinations(array.Muon_Px, 2)
```

<br><br><br><br><br>

## Arrays can have custom [behavior](https://awkward-array.readthedocs.io/en/latest/ak.behavior.html)

The following come from the new [Vector](https://github.com/scikit-hep/vector#readme) library.

```ipython3
import vector
vector.register_awkward()
```

```ipython3
muons = ak.zip({"px": array.Muon_Px, "py": array.Muon_Py, "pz": array.Muon_Pz, "E": array.Muon_E}, with_name="Momentum4D")
muons
```

This is an array of lists of vectors, and methods like `pt`, `eta`, `phi` apply through the whole array.

```ipython3
muons.pt
```

```ipython3
muons.eta
```

```ipython3
muons.phi
```

<br><br><br>

Let’s try an example: ΔR(muons, jets)

```ipython3
jets = ak.zip({"px": array.Jet_Px, "py": array.Jet_Py, "pz": array.Jet_Pz, "E": array.Jet_E}, with_name="Momentum4D")
jets
```

```ipython3
ak.num(muons), ak.num(jets)
```

```ipython3
ms, js = ak.unzip(ak.cartesian((muons, jets)))
ms, js
```

```ipython3
ak.num(ms), ak.num(js)
```

```ipython3
ms.deltaR(js)
```

<br><br><br>

And another: muon pairs (all combinations, not just the first two per event).

```ipython3
ak.num(muons)
```

```ipython3
m1, m2 = ak.unzip(ak.combinations(muons, 2))
m1, m2
```

```ipython3
ak.num(m1), ak.num(m2)
```

```ipython3
m1 + m2
```

```ipython3
(m1 + m2).mass
```

```ipython3
import hist

hist.Hist.new.Reg(120, 0, 120, name="mass").Double().fill(
    ak.flatten((m1 + m2).mass)
).plot()

None
```

<br><br><br>

### It doesn’t matter which coordinates were used to construct it

```ipython3
array2 = uproot.open(
    "https://github.com/jpivarski-talks/2023-12-18-hsf-india-tutorial-bhubaneswar/raw/main/data/SMHiggsToZZTo4L.root:Events"
).arrays(["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge"], entry_stop=100000)
```

```ipython3
import particle

muons2 = ak.zip({"pt": array2.Muon_pt, "eta": array2.Muon_eta, "phi": array2.Muon_phi, "q": array2.Muon_charge}, with_name="Momentum4D")
muons2["mass"] = particle.Particle.findall("mu-")[0].mass / 1000.0
muons2
```

As long as you use properties (dots, not strings in brackets), you don’t need to care what coordinates it’s based on.

```ipython3
muons2.px
```

```ipython3
muons2.py
```

```ipython3
muons2.pz
```

```ipython3
muons2.E
```

```ipython3
m1, m2 = ak.unzip(ak.combinations(muons2, 2))
hist.Hist.new.Log(200, 0.1, 120, name="mass").Double().fill(
    ak.flatten((m1 + m2).mass)
).plot()

None
```

<br><br><br>

## Awkward Arrays and Vector in Numba

Remember Numba, the JIT-compiler from [1-python-performance.ipynb](https://github.com/jpivarski-talks/2021-06-14-uproot-awkward-columnar-hats/blob/main/1-python-performance.ipynb)? Awkward Array and Vector have been implemented in Numba’s compiler.

```ipython3
import numba as nb

@nb.njit
def first_big_dimuon(events):
    for event in events:
        for i in range(len(event)):
            mu1 = event[i]
            for j in range(i + 1, len(event)):
                mu2 = event[j]
                dimuon = mu1 + mu2
                if dimuon.mass > 10:
                    return dimuon
```

```ipython3
first_big_dimuon(muons2)
```
