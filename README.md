<img src="docs-img/logo/logo-300px.png">

[![Scikit-HEP](https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg)](https://scikit-hep.org/)
[![Build Status](https://dev.azure.com/jpivarski/Scikit-HEP/_apis/build/status/buildtest-awkward?branchName=master)](https://dev.azure.com/jpivarski/Scikit-HEP/_build/latest?definitionId=3&branchName=master)
[![DOI](https://zenodo.org/badge/137079949.svg)](https://zenodo.org/badge/latestdoi/137079949)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they're not.

<table>
  <tr>
    <td width="33%" valign="top">
      <a href="https://awkward-array.org/how-to.html">
        <img src="docs-img/panel-data-analysts.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.org/how-to.html">
        How-to documentation<br>for data analysts
        </a>
      </b></p>
    </td>
    <td width="33%" valign="top">
      <a href="https://awkward-array.org/how-it-works.html">
        <img src="docs-img/panel-developers.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.org/how-it-works.html">
        How-it-works tutorials<br>for developers
        </a>
      </b></p>
    </td>
    <td width="33%" valign="top">
      <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        <img src="docs-img/panel-sphinx.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        Python<br>API reference
        </a>
      </b></p>
      <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        <img src="docs-img/panel-doxygen.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        C++<br>API reference
        </a>
      </b></p>
    </td>
  </tr>
</table>

# Installation

Awkward Array can be installed [from PyPI](https://pypi.org/project/awkward1/) using pip:

```bash
pip install awkward1
```

Most users will get a precompiled binary (wheel) for your operating system and Python version. If not, the above attempts to compile from source.

   * Report bugs, request features, and ask for additional documentation on [GitHub Issues](https://github.com/scikit-hep/awkward-1.0/issues). If you have a general "How do I…?" question, we'll answer it as a new [example in the tutorial](https://awkward-array.org/how-to.html).
   * If you have a problem that's too specific to be new documentation or it isn't exclusively related to Awkward Array, it might be more appropriate to ask on [StackOverflow with the [awkward-array] tag](https://stackoverflow.com/questions/tagged/awkward-array). Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.
   * The [Gitter Scikit-HEP/community](https://gitter.im/Scikit-HEP/community) is a way to get in touch with all Scikit-HEP developers and users.

## Installation for developers

Be sure to clone this repository recursively to get the header-only C++ dependencies.

```bash
git clone --recursive https://github.com/scikit-hep/awkward-1.0.git
```

You can install it on your system with pip, which uses exactly the same procedure as deployment. This is recommended if you **do not** expect to change the code.

```bash
pip install .[test,dev]
```

Or you can build it locally for incremental development. The following reuses a local directory so that you only recompile what you've changed. This is recommended if you **do** expect to change the code.

```bash
python localbuild.py --pytest tests
```

   * [Continuous integration](https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=3&_a=summary) and [continuous deployment](https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=4&_a=summary) are hosted by Azure Pipelines.
   * [Release history](https://awkward-array.readthedocs.io/en/latest/_auto/changelog.html) (changelog) is hosted by ReadTheDocs.
   * [CONTRIBUTING.md](CONTRIBUTING.md) provides more information on how to contribute.
   * The [LICENSE](https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE) is BSD-3.

## Using Awkward Array as a dependency

Python projects can simply `import awkward1`.

C++ projects can link against the shared libraries `libawkward-cpu-kernels.so` and `libawkward.so` or their static library equivalents. These libraries are shipped, along with the include files, as part of pip's installation.

   * See the [dependent-project](https://github.com/scikit-hep/awkward-1.0/tree/master/dependent-project) directory for examples.

# Papers and talks about Awkward Array

   * [Original motivations document](https://docs.google.com/document/d/1lj8ARTKV1_hqGTh0W_f01S6SsmpzZAXz9qqqWnEB3j4/edit?usp=sharing) from July 2019, now out-of-date.
   * [StrangeLoop talk](https://youtu.be/2NxWpU7NArk) on September 14, 2019.
   * [PyHEP talk](https://indico.cern.ch/event/833895/contributions/3577882) on October 17, 2019.
   * [CHEP talk](https://indico.cern.ch/event/773049/contributions/3473258) on November 7, 2019.
   * [CHEP 2019 proceedings](https://arxiv.org/abs/2001.06307) (to be published in _EPJ Web of Conferences_).
   * [Demo for Coffea developers](https://github.com/scikit-hep/awkward-1.0/blob/master/docs-demo-notebooks/2019-12-20-coffea-demo.ipynb) on December 20, 2019.
   * [Demo for Numba developers](https://github.com/scikit-hep/awkward-1.0/blob/master/docs-demo-notebooks/2020-01-22-numba-demo-EVALUATED.ipynb) on January 22, 2020.
   * [Summary poster](https://github.com/jpivarski/2020-02-27-irishep-poster/blob/master/pivarski-irishep-poster.pdf) on February 27, 2020.
   * [Demo for Electron Ion Collider users](https://github.com/jpivarski/2020-04-08-eic-jlab#readme) ([video](https://www.youtube.com/watch?v=FoxNS6nlbD0)) on April 8, 2020.

# Acknowledgements

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

Thanks especially to the gracious help of awkward-array contributors (including the [original repository](https://github.com/scikit-hep/awkward-array)).

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/jpivarski"><img src="https://avatars0.githubusercontent.com/u/1852447?v=4" width="100px;" alt=""/><br /><sub><b>Jim Pivarski</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=jpivarski" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward-1.0/commits?author=jpivarski" title="Documentation">📖</a> <a href="#infra-jpivarski" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-jpivarski" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/ianna"><img src="https://avatars0.githubusercontent.com/u/1390682?v=4" width="100px;" alt=""/><br /><sub><b>Ianna Osborne</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=ianna" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nsmith-"><img src="https://avatars2.githubusercontent.com/u/6587412?v=4" width="100px;" alt=""/><br /><sub><b>Nicholas Smith</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=nsmith-" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward-1.0/commits?author=nsmith-" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/lgray"><img src="https://avatars0.githubusercontent.com/u/1068089?v=4" width="100px;" alt=""/><br /><sub><b>Lindsey Gray</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=lgray" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward-1.0/commits?author=lgray" title="Tests">⚠️</a></td>
    <td align="center"><a href="http://iscinumpy.gitlab.io"><img src="https://avatars1.githubusercontent.com/u/4616906?v=4" width="100px;" alt=""/><br /><sub><b>Henry Schreiner</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=henryiii" title="Code">💻</a> <a href="#infra-henryiii" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://github.com/reikdas"><img src="https://avatars0.githubusercontent.com/u/11775615?v=4" width="100px;" alt=""/><br /><sub><b>Pratyush Das</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=reikdas" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/trickarcher"><img src="https://avatars3.githubusercontent.com/u/39878675?v=4" width="100px;" alt=""/><br /><sub><b>Anish Biswas</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=trickarcher" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/Ellipse0934"><img src="https://avatars3.githubusercontent.com/u/7466364?v=4" width="100px;" alt=""/><br /><sub><b>Ellipse0934</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=Ellipse0934" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://gitlab.com/veprbl"><img src="https://avatars1.githubusercontent.com/u/245573?v=4" width="100px;" alt=""/><br /><sub><b>Dmitry Kalinkin</b></sub></a><br /><a href="#infra-veprbl" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://github.com/glass-ships"><img src="https://avatars2.githubusercontent.com/u/26975530?v=4" width="100px;" alt=""/><br /><sub><b>glass-ships</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=glass-ships" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward-1.0/commits?author=glass-ships" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/charles-c-escott/"><img src="https://avatars3.githubusercontent.com/u/48469669?v=4" width="100px;" alt=""/><br /><sub><b>Charles Escott</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=EscottC" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/masonproffitt"><img src="https://avatars3.githubusercontent.com/u/32773304?v=4" width="100px;" alt=""/><br /><sub><b>Mason Proffitt</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=masonproffitt" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/mhedges"><img src="https://avatars3.githubusercontent.com/u/18672512?v=4" width="100px;" alt=""/><br /><sub><b>Michael Hedges</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=mhedges" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/guitargeek"><img src="https://avatars2.githubusercontent.com/u/6578603?v=4" width="100px;" alt=""/><br /><sub><b>Jonas Rembser</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=guitargeek" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/Jayd-1234"><img src="https://avatars0.githubusercontent.com/u/34567389?v=4" width="100px;" alt=""/><br /><sub><b>Jaydeep Nandi</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=Jayd-1234" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/benkrikler"><img src="https://avatars0.githubusercontent.com/u/4083697?v=4" width="100px;" alt=""/><br /><sub><b>benkrikler</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward-1.0/commits?author=benkrikler" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

💻: code, 📖: documentation, 🚇: infrastructure, 🚧: maintainance, ⚠: tests that drive development.
