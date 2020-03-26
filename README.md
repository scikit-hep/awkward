<img src="docs/content/images/logo/logo-300px.png">

<table>
  <tr>
    <td width="33%" valign="top">
      <a href="https://scikit-hep.org/awkward-1.0/index.html">
        <img src="docs/content/images/logo/panel-data-analysts.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://scikit-hep.org/awkward-1.0/index.html">
        How-to documentation<br>for data analysts
        </a>
      </b></p>
    </td>
    <td width="33%" valign="top">
      <a href="https://scikit-hep.org/awkward-1.0/index.html">
        <img src="docs/content/images/logo/panel-developers.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://scikit-hep.org/awkward-1.0/index.html">
        How-it-works tutorials<br>for developers
        </a>
      </b></p>
    </td>
    <td width="33%" valign="top">
      <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        <img src="docs/content/images/logo/panel-doxygen.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        C++<br>API reference
        </a>
      </b></p>
      <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        <img src="docs/content/images/logo/panel-sphinx.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        Python<br>API reference
        </a>
      </b></p>
    </td>
  </tr>
</table>

# Installation

## Normal installation

Normally, you would install Awkward [from PyPI](https://pypi.org/project/awkward1/) using pip

```bash
pip install awkward1
```

to get the latest release of Awkward 1.0 as a precompiled wheel. If a wheel does not exist for your combination of operating system and Python version, the above command attempts to compile from source, downloading any dependencies it needs to do that.

## Manually installing from source

If you need to force an installation from source, get it from GitHub via

```bash
git clone --recursive https://github.com/scikit-hep/awkward-1.0.git
```

(note the recursive git-clone; it is required to get C++ dependencies) and compile+install with dependencies via

```bash
pip install .
```

or

```bash
pip install .[test,dev]
```

to perform tests (`[test]`) on all optional dependencies (`[dev]`).

## Development workflow

If you are developing Awkward Array, manually installing from source will work, but it doesn't cache previously compiled code for rapid recompilation. Instead, get it from GitHub via

```bash
git clone --recursive https://github.com/scikit-hep/awkward-1.0.git
```

(note the recursive git-clone; it is required to get C++ dependencies) and compile+install with dependencies via

```bash
python localbuild.py --pytest tests
```

The `--pytest tests` optionally runs selected tests. See 

```bash
python localbuild.py --help
```

for more information. The build is based on CMake; see [localbuild.py](./localbuild.py) if you need to run CMake directly.

Continuous integration (CI) and continuous deployment (CD) are hosted by Azure Pipelines:

<p align="center"><b><a href="https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=3&_a=summary">buildtest-awkward</a></b> (CI) and <b><a href="https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=4&_a=summary">deploy-awkward</a></b> (CD)</p>

## Building projects that depend on Awkward

Python projects can simply use `awkward1` as a Python library.

C++ projects can either link against the shared libraries `libawkward-cpu-kernels.so` and `libawkward.so` or the static libraries, whose names end in `-static`. All four libraries, as well as their C++ header files, are shipped with the Python library. Even if you installed Awkward Array with pip, you'll have everything you need to build an Awkward C++ program.

If you also want to bind your C++ to Python and share Awkward Arrays between modules in Python, see the [dependent-project](https://github.com/scikit-hep/awkward-1.0/tree/master/dependent-project) example. This is a small CMake project bound to Python with pybind11 that can produce and consume Awkward Arrays in Python. Such projects depend on a specific version of Awkward Array, but we intend to stabilize the ABI for more flexibility.

# Papers and talks on Awkward Array

Development of Awkward 1.0, to replace [scikit-hep/awkward-array](https://github.com/scikit-hep/awkward-array#readme) in 2020.

   * The [original motivations document](https://docs.google.com/document/d/1lj8ARTKV1_hqGTh0W_f01S6SsmpzZAXz9qqqWnEB3j4/edit?usp=sharing) from July 2019, now a little out-of-date.
   * My [StrangeLoop talk](https://youtu.be/2NxWpU7NArk) on September 14, 2019.
   * My [PyHEP talk](https://indico.cern.ch/event/833895/contributions/3577882) on October 17, 2019.
   * My [CHEP talk](https://indico.cern.ch/event/773049/contributions/3473258) on November 7, 2019.
   * My [CHEP 2019 proceedings](https://arxiv.org/abs/2001.06307) (to be published in _EPJ Web of Conferences_).
   * [Demo for Coffea developers](https://github.com/scikit-hep/awkward-1.0/blob/master/docs-demo-notebooks/2019-12-20-coffea-demo.ipynb) on December 20, 2019.
   * [Demo for Numba developers](https://github.com/scikit-hep/awkward-1.0/blob/master/docs-demo-notebooks/2020-01-22-numba-demo-EVALUATED.ipynb) on January 22, 2020.

