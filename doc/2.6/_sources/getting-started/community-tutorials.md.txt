# Community tutorials

## Jagged, ragged, Awkward Arrays
An HSF-provided tutorial aimed at High Energy Physics (HEP) researchers on using Awkward Array to obtain a dimuon-mass spectrum.

### Format
- [{fas}`external-link-alt` Webpage](https://hsf-training.github.io/hsf-training-scikit-hep-webpage/04-awkward/index.html) with example code snippets.

### Objectives
- Filter ragged arrays using innermost lists.
- Compute quantities on combinations of fields with {func}`ak.combinations`.
- Unzip arrays with {func}`ak.unzip`.
- Ravel (flatten) ragged arrays with {func}`ak.ravel`.

## Loopy and unloopy programming techniques (SciPy 2022)
A tutorial presented at the SciPy conference on July 11, 2022.

### Format
- [{fab}`github` GitHub repository](https://github.com/jpivarski-talks/2022-07-11-scipy-loopy-tutorial)
 with Jupyter Notebooks that can be run on [MyBinder](https://mybinder.org/v2/gh/jpivarski-talks/2022-07-11-scipy-loopy-tutorial/v1.0?urlpath=lab/tree/narrative.ipynb).
- [{fab}`youtube` YouTube](https://www.youtube.com/watch?v=Dovyd72eD70) recording of presentation.

### Objectives
- Load data from a remote Parquet source with {func}`ak.from_parquet`.
- Explore a complex dataset.
- Mask and slice ragged array with {func}`ak.mask`.
- Perform ragged reduction and broadcasting.
- Flatten ragged arrays with {func}`ak.flatten`.

## Columnar data analysis (CoDaS-HEP 2022)
A tutorial aimed at HEP researchers, given at CODAS-HEP, to reconstruct Z masses and the Higgs mass from four leptons (4μ, 4e, 2μ2e) using Awkward Array and uproot.

### Format
- [{fab}`github` GitHub repository](https://github.com/jpivarski-talks/2022-08-03-codas-hep-columnar-tutorial)  with Jupyter Notebooks that can be run on [MyBinder](https://mybinder.org/).
 
### Objectives
- Restructure/reformat arrays with {func}`ak.zip`.
- Compute kinematic quantities with [`vector`](https://github.com/scikit-hep/vector).
- Add new fields to an array.
- Explore combinatorics with {func}`ak.cartesian` and {func}`ak.combinations`.

## Uproot-Awkward columnar HATS (2020)
Tutorials for Uproot Awkward Columnar HATS, a hands-on tutorial hosted by the [Fermilab LPC](https://lpc.fnal.gov/).

### Format
- [{fab}`github` GitHub repository](https://github.com/jpivarski-talks/2020-06-08-uproot-awkward-columnar-hats)  with Jupyter Notebooks that can be run on [MyBinder](https://mybinder.org/).
 
### Objectives
- Index nested record arrays.
- Perform ragged reduction and broadcasting.
- Restructure/reformat arrays with {func}`ak.zip`.
- Write high-performance imperative routines that operate upon Awkward Arrays with Numba.
- Build Awkward Arrays imperatively with {class}`ak.ArrayBuilder`.
- Flatten ragged arrays with {func}`ak.flatten`.
- Unzip arrays with {func}`ak.unzip`.
- Explore combinatorics with {func}`ak.cartesian` and {func}`ak.combinations`.
- Mask and slice ragged array with {func}`ak.mask`.
- Explode Awkward Arrays into DataFrames with {func}`ak.pandas.df`.
