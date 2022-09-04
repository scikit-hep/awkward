<img src="docs-img/logo/logo-300px.png">

[![PyPI version](https://badge.fury.io/py/awkward.svg)](https://pypi.org/project/awkward)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/awkward)](https://github.com/conda-forge/awkward-feedstock)
[![Python 3.7‒3.11](https://img.shields.io/badge/python-3.7%E2%80%923.11-blue)](https://www.python.org)
[![BSD-3 Clause License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Test](https://github.com/scikit-hep/awkward/actions/workflows/build-test.yml/badge.svg?branch=main)](https://github.com/scikit-hep/awkward/actions/workflows/build-test.yml)

[![Scikit-HEP](https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg)](https://scikit-hep.org/)
[![NSF-1836650](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4341376.svg)](https://doi.org/10.5281/zenodo.4341376)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://awkward-array.org)
[![Gitter](https://img.shields.io/badge/chat-online-success)](https://gitter.im/Scikit-HEP/awkward-array)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they're not.

# Motivating example

Given an array of objects with `x`, `y` fields and variable-length nested lists like

```python
array = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
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

Awkward Array can be installed [from PyPI](https://pypi.org/project/awkward) using pip:

```bash
pip install awkward
```

You will likely get a precompiled binary (wheel), depending on your operating system and Python version. If not, pip attempts to compile from source (which requires a C++ compiler, make, and CMake).

Awkward Array is also available using [conda](https://anaconda.org/conda-forge/awkward), which always installs a binary:
```bash
conda install -c conda-forge awkward
```

If you have already added `conda-forge` as a channel, the `-c conda-forge` is unnecessary. Adding the channel is recommended because it ensures that all of your packages use compatible versions:

```bash
conda config --add channels conda-forge
conda update --all
```

# Getting help

<table>
  <tr>
    <td width="66%" valign="top">
      <a href="https://awkward-array.org">
        <img src="docs-img/panel-tutorials.png" width="570">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.org">
        How-to tutorials
        </a>
      </b></p>
    </td>
    <td width="33%" valign="top">
      <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        <img src="docs-img/panel-sphinx.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        Python API reference
        </a>
      </b></p>
      <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        <img src="docs-img/panel-doxygen.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        C++ API reference
        </a>
      </b></p>
    </td>
  </tr>
</table>

   * Report bugs, request features, and ask for additional documentation on [GitHub Issues](https://github.com/scikit-hep/awkward/issues).
   * If you have a "How do I...?" question, start a [GitHub Discussion](https://github.com/scikit-hep/awkward/discussions) with category "Q&A".
   * Alternatively, ask about it on [StackOverflow with the [awkward-array] tag](https://stackoverflow.com/questions/tagged/awkward-array). Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.
   * To ask questions in real time, try the Gitter [Scikit-HEP/awkward-array](https://gitter.im/Scikit-HEP/awkward-array) chat room.

# Installation for developers

Be sure to clone this repository recursively to get the header-only C++ dependencies.

```bash
git clone --recursive https://github.com/scikit-hep/awkward.git
```

Also be aware that the default branch is named `main`, not `master`, which could be important for pull requests from forks.

You can install it on your system with pip, which uses exactly the same procedure as deployment. This is recommended if you **do not** expect to change the code.

```bash
pip install .[test,dev]
```

Or you can build it locally for incremental development. The following reuses a local directory so that you only recompile what you've changed. This is recommended if you **do** expect to change the code.

```bash
python localbuild.py --pytest tests
```

The `--pytest tests` runs the integration tests from the `tests` directory (drop it to build only).

For more fine-grained testing, we also have tests of the low-level kernels, which can be invoked with

```bash
python dev/generate-tests.py
python -m pytest -vv -rs tests-spec
python -m pytest -vv -rs tests-cpu-kernels
```

   * [Continuous integration](https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=3&_a=summary) and [continuous deployment](https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=4&_a=summary) are hosted by [Azure Pipelines](https://azure.microsoft.com/en-us/services/devops/pipelines/).
   * [Release history](https://awkward-array.readthedocs.io/en/latest/_auto/changelog.html) (changelog) is hosted by [ReadTheDocs](https://readthedocs.org).
   * [awkward-array.org](https://awkward-array.org) is hosted by [Netlify](https://www.netlify.com).
   * [CONTRIBUTING.md](CONTRIBUTING.md) for technical information on how to contribute.
   * [Code of conduct](https://scikit-hep.org/code-of-conduct) for how we work together.
   * The [LICENSE](LICENSE) is BSD-3.

# Release notes and Roadmap

The [release notes/history/changelog](https://awkward-array.readthedocs.io/en/latest/_auto/changelog.html) is generated as part of the documentation.

The [roadmap, future version planning, issue prioritization, deprecation schedule, etc.](https://github.com/scikit-hep/awkward/wiki) are kept up-to-date on the wiki.

# Papers and talks about Awkward Array

   * [Original motivations for Awkward 1.0](https://docs.google.com/document/d/1lj8ARTKV1_hqGTh0W_f01S6SsmpzZAXz9qqqWnEB3j4/edit?usp=sharing) from July 2019, now out-of-date.
   * [StrangeLoop talk](https://youtu.be/2NxWpU7NArk) (video) on September 14, 2019.
   * [PyHEP talk](https://indico.cern.ch/event/833895/contributions/3577882) on October 17, 2019.
   * [CHEP talk](https://indico.cern.ch/event/773049/contributions/3473258) on November 7, 2019.
   * [CHEP 2019 proceedings](https://arxiv.org/abs/2001.06307) (published in _EPJ Web of Conferences_).
   * [Demo for Coffea developers](https://github.com/scikit-hep/awkward/blob/main/docs-jupyter/2019-12-20-coffea-demo.ipynb) on December 20, 2019.
   * [Demo for Numba developers](https://github.com/scikit-hep/awkward/blob/main/docs-jupyter/2020-01-22-numba-demo-EVALUATED.ipynb) on January 22, 2020.
   * [Summary poster](https://github.com/jpivarski/2020-02-27-irishep-poster/blob/master/pivarski-irishep-poster.pdf) on February 27, 2020.
   * [Demo for Electron Ion Collider users](https://github.com/jpivarski/2020-04-08-eic-jlab#readme) ([video](https://www.youtube.com/watch?v=FoxNS6nlbD0)) on April 8, 2020.
   * [Presentation at SciPy 2020](https://youtu.be/WlnUF3LRBj4) (video) on July 5, 2020.
   * [Tutorial at PyHEP 2020](https://youtu.be/ea-zYLQBS4U) (video with [interactive notebook on Binder](https://mybinder.org/v2/gh/jpivarski/2020-07-13-pyhep2020-tutorial.git/1.1?urlpath=lab/tree/tutorial.ipynb)) on July 13, 2020.
   * [Tutorial at PyHEP 2021](https://youtu.be/5aWAxvdrszw?t=9189) (video with [interactive notebook on Binder](https://mybinder.org/v2/gh/jpivarski-talks/2021-07-06-pyhep-uproot-awkward-tutorial/v1.2?urlpath=lab/tree/uproot-awkward-tutorial.ipynb) on July 6, 2021.
   * [Tutorial for STAR collaboration meeting](https://youtu.be/NnU_zp5s1MY) on September 13, 2021 (video with [notebooks on GitHub](https://github.com/jpivarski-talks/2021-09-13-star-uproot-awkward-tutorial#readme)). This is the first tutorial with extensive exercises to test your understanding.
   * [Lessons learned in Python-C++ integration](https://indico.cern.ch/event/855454/contributions/4605044/) ([video](https://videos.cern.ch/record/2295164) and [slides](https://indico.cern.ch/event/855454/contributions/4605044/attachments/2349193/4006676/main.pdf)) on December 1, 2021. This talk describes the motivation for Awkward version 2.0.
   * [Awkward Array updates](https://indico.cern.ch/event/1140031/) on April 6, 2022. A demonstration and overview of Awkward version 2.0.
   * [Summary poster](https://github.com/jpivarski-talks/2022-07-25-cssi-meeting-poster/blob/main/pivarski-awkward-cssi-poster.pdf) on July 5, 2022.
   * [Tutorial at SciPy 2022](https://youtu.be/Dovyd72eD70) (video with [notebooks on GitHub](https://github.com/jpivarski-talks/2022-07-11-scipy-loopy-tutorial#readme)) on July 11, 2022. This is aimed at a general scientific audience, not particle physicists in particular. Features three long exercises.
   * [Tutorial at CoDaS-HEP 2022](https://github.com/jpivarski-talks/2022-08-03-codas-hep-columnar-tutorial#readme) on August 3, 2022. Aimed at particle physicists; features a long exercise on Higgs decay combinatorics.

### Citing Awkward Array in a publication

On the [GitHub README](https://github.com/scikit-hep/awkward), see the "Cite this repository" drop-down menu on the top-right of the page.

# Acknowledgements

Support for this work was provided by NSF cooperative agreement [OAC-1836650](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1836650) (IRIS-HEP), grant [OAC-1450377](https://nsf.gov/awardsearch/showAward?AWD_ID=1450377) (DIANA/HEP), [PHY-1520942](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1520942) (US-CMS LHC Ops), and [OAC-2103945](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103945) (Awkward Array).

We also thank [Erez Shinan](https://github.com/erezsh) and the developers of the [Lark standalone parser](https://github.com/lark-parser/lark), which is used to parse type strings as type objects.

Thanks especially to the gracious help of Awkward Array contributors (including the [original repository](https://github.com/scikit-hep/awkward-0.x)).

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/jpivarski"><img src="https://avatars0.githubusercontent.com/u/1852447?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jim Pivarski</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=jpivarski" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=jpivarski" title="Documentation">📖</a> <a href="#infra-jpivarski" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-jpivarski" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/ianna"><img src="https://avatars0.githubusercontent.com/u/1390682?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ianna Osborne</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=ianna" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/reikdas"><img src="https://avatars0.githubusercontent.com/u/11775615?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Pratyush Das</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=reikdas" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/trickarcher"><img src="https://avatars3.githubusercontent.com/u/39878675?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Anish Biswas</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=trickarcher" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/glass-ships"><img src="https://avatars2.githubusercontent.com/u/26975530?v=4?s=100" width="100px;" alt=""/><br /><sub><b>glass-ships</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=glass-ships" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=glass-ships" title="Tests">⚠️</a></td>
    <td align="center"><a href="http://iscinumpy.gitlab.io"><img src="https://avatars1.githubusercontent.com/u/4616906?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Henry Schreiner</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=henryiii" title="Code">💻</a> <a href="#infra-henryiii" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://github.com/nsmith-"><img src="https://avatars2.githubusercontent.com/u/6587412?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nicholas Smith</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=nsmith-" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=nsmith-" title="Tests">⚠️</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/lgray"><img src="https://avatars0.githubusercontent.com/u/1068089?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lindsey Gray</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=lgray" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=lgray" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/Ellipse0934"><img src="https://avatars3.githubusercontent.com/u/7466364?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ellipse0934</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Ellipse0934" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://gitlab.com/veprbl"><img src="https://avatars1.githubusercontent.com/u/245573?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Dmitry Kalinkin</b></sub></a><br /><a href="#infra-veprbl" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/charles-c-escott/"><img src="https://avatars3.githubusercontent.com/u/48469669?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Charles Escott</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=EscottC" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/masonproffitt"><img src="https://avatars3.githubusercontent.com/u/32773304?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mason Proffitt</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=masonproffitt" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/mhedges"><img src="https://avatars3.githubusercontent.com/u/18672512?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Michael Hedges</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=mhedges" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/guitargeek"><img src="https://avatars2.githubusercontent.com/u/6578603?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jonas Rembser</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=guitargeek" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/Jayd-1234"><img src="https://avatars0.githubusercontent.com/u/34567389?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jaydeep Nandi</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Jayd-1234" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/benkrikler"><img src="https://avatars0.githubusercontent.com/u/4083697?v=4?s=100" width="100px;" alt=""/><br /><sub><b>benkrikler</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=benkrikler" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/bfis"><img src="https://avatars0.githubusercontent.com/u/15651150?v=4?s=100" width="100px;" alt=""/><br /><sub><b>bfis</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=bfis" title="Code">💻</a></td>
    <td align="center"><a href="https://ddavis.io/"><img src="https://avatars2.githubusercontent.com/u/3202090?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Doug Davis</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=douglasdavis" title="Code">💻</a></td>
    <td align="center"><a href="http://twitter: @JoosepPata"><img src="https://avatars0.githubusercontent.com/u/69717?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Joosep Pata</b></sub></a><br /><a href="#ideas-jpata" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="http://martindurant.github.io/"><img src="https://avatars1.githubusercontent.com/u/6042212?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Martin Durant</b></sub></a><br /><a href="#ideas-martindurant" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="http://gordonwatts.wordpress.com"><img src="https://avatars2.githubusercontent.com/u/1778366?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Gordon Watts</b></sub></a><br /><a href="#ideas-gordonwatts" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://gitlab.com/nikoladze"><img src="https://avatars0.githubusercontent.com/u/3707225?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nikolai Hartmann</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=nikoladze" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/sjperkins"><img src="https://avatars3.githubusercontent.com/u/3530212?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Simon Perkins</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=sjperkins" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/drahnreb"><img src="https://avatars.githubusercontent.com/u/25883607?v=4?s=100" width="100px;" alt=""/><br /><sub><b>.hard</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=drahnreb" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=drahnreb" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/HenryDayHall"><img src="https://avatars.githubusercontent.com/u/12996763?v=4?s=100" width="100px;" alt=""/><br /><sub><b>HenryDayHall</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=HenryDayHall" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/agoose77"><img src="https://avatars.githubusercontent.com/u/1248413?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Angus Hollands</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=agoose77" title="Tests">⚠️</a> <a href="https://github.com/scikit-hep/awkward/commits?author=agoose77" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ioanaif"><img src="https://avatars.githubusercontent.com/u/9751871?v=4?s=100" width="100px;" alt=""/><br /><sub><b>ioanaif</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=ioanaif" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=ioanaif" title="Tests">⚠️</a></td>
    <td align="center"><a href="http://lizards.opensuse.org/author/bmwiedemann/"><img src="https://avatars.githubusercontent.com/u/637990?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Bernhard M. Wiedemann</b></sub></a><br /><a href="#maintenance-bmwiedemann" title="Maintenance">🚧</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://www.matthewfeickert.com/"><img src="https://avatars.githubusercontent.com/u/5142394?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Matthew Feickert</b></sub></a><br /><a href="#maintenance-matthewfeickert" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/SantamRC"><img src="https://avatars.githubusercontent.com/u/52635773?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Santam Roy Choudhury</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=SantamRC" title="Tests">⚠️</a></td>
    <td align="center"><a href="http://jeroen.vangoey.be"><img src="https://avatars.githubusercontent.com/u/59344?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jeroen Van Goey</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=BioGeek" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/Ahmad-AlSubaie"><img src="https://avatars.githubusercontent.com/u/32343365?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ahmad-AlSubaie</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Ahmad-AlSubaie" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ManasviGoyal"><img src="https://avatars.githubusercontent.com/u/55101825?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Manasvi Goyal</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=ManasviGoyal" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/aryan26roy"><img src="https://avatars.githubusercontent.com/u/50577809?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Aryan Roy</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=aryan26roy" title="Code">💻</a></td>
    <td align="center"><a href="https://saransh-cpp.github.io/"><img src="https://avatars.githubusercontent.com/u/74055102?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Saransh</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Saransh-cpp" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

💻: code, 📖: documentation, 🚇: infrastructure, 🚧: maintenance, ⚠: tests and feedback, 🤔: foundational ideas.
