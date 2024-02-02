<!-- begin-logo -->
![](docs-img/logo/logo-300px.png)
<!-- end-logo -->

[![PyPI version](https://badge.fury.io/py/awkward.svg)](https://pypi.org/project/awkward)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/awkward)](https://github.com/conda-forge/awkward-feedstock)
[![Python 3.8‒3.12](https://img.shields.io/badge/python-3.8%E2%80%923.12-blue)](https://www.python.org)
[![BSD-3 Clause License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Test](https://github.com/scikit-hep/awkward/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/scikit-hep/awkward/actions/workflows/test.yml)

[![Scikit-HEP](https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg)](https://scikit-hep.org/)
[![NSF-1836650](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4341376.svg)](https://doi.org/10.5281/zenodo.4341376)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://awkward-array.org/)
[![Gitter](https://img.shields.io/badge/chat-online-success)](https://gitter.im/Scikit-HEP/awkward-array)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they're not.

# Motivating example

Given an array of lists of objects with `x`, `y` fields (with nested lists in the `y` field),

```python
import awkward as ak

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

The expression using Awkward Arrays is more concise, using idioms familiar from NumPy, and it also has NumPy-like performance. For a similar problem 10 million times larger than the one above (single-threaded on a 2.2 GHz processor),

   * the Awkward Array one-liner takes **1.5 seconds** to run and uses **2.1 GB** of memory,
   * the equivalent using Python lists and dicts takes **140 seconds** to run and uses **22 GB** of memory.

Awkward Array is even faster when used in [Numba](https://numba.pydata.org/)'s JIT-compiled functions.

See the [Getting started](https://awkward-array.org/doc/main/getting-started/index.html) documentation on [awkward-array.org](https://awkward-array.org) for an introduction, including a [no-install demo](https://awkward-array.org/doc/main/getting-started/try-awkward-array.html) you can try in your web browser.

# Getting help

   * View the documentation on [awkward-array.org](https://awkward-array.org/).
   * Report bugs, request features, and ask for additional documentation on [GitHub Issues](https://github.com/scikit-hep/awkward/issues).
   * If you have a "How do I...?" question, start a [GitHub Discussion](https://github.com/scikit-hep/awkward/discussions) with category "Q&A".
   * Alternatively, ask about it on [StackOverflow with the [awkward-array] tag](https://stackoverflow.com/questions/tagged/awkward-array). Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.
   * To ask questions in real time, try the Gitter [Scikit-HEP/awkward-array](https://gitter.im/Scikit-HEP/awkward-array) chat room.

# Installation

Awkward Array can be installed from [PyPI](https://pypi.org/project/awkward) using pip:

```bash
pip install awkward
```

The `awkward` package is pure Python, and it will download the `awkward-cpp` compiled components as a dependency. If there is no `awkward-cpp` binary package (wheel) for your platform and Python version, pip will attempt to compile it from source (which has additional dependencies, such as a C++ compiler).

Awkward Array is also available on [conda-forge](https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge):

```bash
conda install -c conda-forge awkward
```

<!-- readme-pypi-ignore-after -->

Because of the two packages (`awkward-cpp` may be updated in GitHub but not on PyPI), pip install through git (`pip install git+https://...`) will not work. Instead, use the [Installation for developers](#installation-for-developers) section below.

# Installation for developers

Clone this repository _recursively_ to get the header-only C++ dependencies, then generate sources with [nox](https://nox.thea.codes/), compile and install `awkward-cpp`, and finally install `awkward` as an editable installation:

```bash
git clone --recursive https://github.com/scikit-hep/awkward.git
cd awkward

nox -s prepare
python -m pip install -v ./awkward-cpp
python -m pip install -e .
```

Tests can be run in parallel with [pytest](https://docs.pytest.org/):

```bash
python -m pytest -n auto tests
```

For more details, see [CONTRIBUTING.md](https://github.com/scikit-hep/awkward/blob/main/CONTRIBUTING.md), or one of the links below.

   * [Continuous integration](https://github.com/scikit-hep/awkward/actions/workflows/test.yml) and [continuous deployment](https://github.com/scikit-hep/awkward/actions/workflows/wheels.yml) are hosted by [GitHub Actions](https://github.com/features/actions/).
   * [Code of conduct](https://scikit-hep.org/code-of-conduct) for how we work together.
   * The [LICENSE](LICENSE) is BSD-3.

# Documentation, Release notes, Roadmap, Citations

The documentation is on [awkward-array.org](https://awkward-array.org), including

   * [Getting started](https://awkward-array.org/doc/main/getting-started/index.html)
   * [User guide](https://awkward-array.org/doc/main/user-guide/index.html)
   * [API reference](https://awkward-array.org/doc/main/reference/index.html)
   * [Tutorials (with videos)](https://awkward-array.org/doc/main/getting-started/community-tutorials.html)
   * [Papers and talks](https://awkward-array.org/doc/main/getting-started/papers-and-talks.html) about Awkward Array

The Release notes for each version are in the [GitHub Releases tab](https://github.com/scikit-hep/awkward/releases).

The Roadmap, Plans, and Deprecation Schedule are in the [GitHub Wiki](https://github.com/scikit-hep/awkward/wiki).

To cite Awkward Array in a paper, see the "Cite this repository" drop-down menu on the top-right of the [GitHub front page](https://github.com/scikit-hep/awkward). The BibTeX is

```bibtex
@software{Pivarski_Awkward_Array_2018,
author = {Pivarski, Jim and Osborne, Ianna and Ifrim, Ioana and Schreiner, Henry and Hollands, Angus and Biswas, Anish and Das, Pratyush and Roy Choudhury, Santam and Smith, Nicholas and Goyal, Manasvi},
doi = {10.5281/zenodo.4341376},
month = {10},
title = {{Awkward Array}},
year = {2018}
}
```

# Acknowledgements

Support for this work was provided by NSF cooperative agreements [OAC-1836650](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1836650) and [PHY-2323298](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2323298) (IRIS-HEP), grant [OAC-1450377](https://nsf.gov/awardsearch/showAward?AWD_ID=1450377) (DIANA/HEP), [PHY-2121686](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2121686) (US-CMS LHC Ops), and [OAC-2103945](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103945) (Awkward Array).

We also thank [Erez Shinan](https://github.com/erezsh) and the developers of the [Lark standalone parser](https://github.com/lark-parser/lark), which is used to parse type strings as type objects.

Thanks especially to the gracious help of Awkward Array contributors (including the [original repository](https://github.com/scikit-hep/awkward-0.x)).

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jpivarski"><img src="https://avatars0.githubusercontent.com/u/1852447?v=4?s=100" width="100px;" alt="Jim Pivarski"/><br /><sub><b>Jim Pivarski</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=jpivarski" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=jpivarski" title="Documentation">📖</a> <a href="#infra-jpivarski" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-jpivarski" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ianna"><img src="https://avatars0.githubusercontent.com/u/1390682?v=4?s=100" width="100px;" alt="Ianna Osborne"/><br /><sub><b>Ianna Osborne</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=ianna" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/reikdas"><img src="https://avatars0.githubusercontent.com/u/11775615?v=4?s=100" width="100px;" alt="Pratyush Das"/><br /><sub><b>Pratyush Das</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=reikdas" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/trickarcher"><img src="https://avatars3.githubusercontent.com/u/39878675?v=4?s=100" width="100px;" alt="Anish Biswas"/><br /><sub><b>Anish Biswas</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=trickarcher" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/glass-ships"><img src="https://avatars2.githubusercontent.com/u/26975530?v=4?s=100" width="100px;" alt="glass-ships"/><br /><sub><b>glass-ships</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=glass-ships" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=glass-ships" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://iscinumpy.gitlab.io"><img src="https://avatars1.githubusercontent.com/u/4616906?v=4?s=100" width="100px;" alt="Henry Schreiner"/><br /><sub><b>Henry Schreiner</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=henryiii" title="Code">💻</a> <a href="#infra-henryiii" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nsmith-"><img src="https://avatars2.githubusercontent.com/u/6587412?v=4?s=100" width="100px;" alt="Nicholas Smith"/><br /><sub><b>Nicholas Smith</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=nsmith-" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=nsmith-" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lgray"><img src="https://avatars0.githubusercontent.com/u/1068089?v=4?s=100" width="100px;" alt="Lindsey Gray"/><br /><sub><b>Lindsey Gray</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=lgray" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=lgray" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ellipse0934"><img src="https://avatars3.githubusercontent.com/u/7466364?v=4?s=100" width="100px;" alt="Ellipse0934"/><br /><sub><b>Ellipse0934</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Ellipse0934" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://gitlab.com/veprbl"><img src="https://avatars1.githubusercontent.com/u/245573?v=4?s=100" width="100px;" alt="Dmitry Kalinkin"/><br /><sub><b>Dmitry Kalinkin</b></sub></a><br /><a href="#infra-veprbl" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/charles-c-escott/"><img src="https://avatars3.githubusercontent.com/u/48469669?v=4?s=100" width="100px;" alt="Charles Escott"/><br /><sub><b>Charles Escott</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=EscottC" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/masonproffitt"><img src="https://avatars3.githubusercontent.com/u/32773304?v=4?s=100" width="100px;" alt="Mason Proffitt"/><br /><sub><b>Mason Proffitt</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=masonproffitt" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mhedges"><img src="https://avatars3.githubusercontent.com/u/18672512?v=4?s=100" width="100px;" alt="Michael Hedges"/><br /><sub><b>Michael Hedges</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=mhedges" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/guitargeek"><img src="https://avatars2.githubusercontent.com/u/6578603?v=4?s=100" width="100px;" alt="Jonas Rembser"/><br /><sub><b>Jonas Rembser</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=guitargeek" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Jayd-1234"><img src="https://avatars0.githubusercontent.com/u/34567389?v=4?s=100" width="100px;" alt="Jaydeep Nandi"/><br /><sub><b>Jaydeep Nandi</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Jayd-1234" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/benkrikler"><img src="https://avatars0.githubusercontent.com/u/4083697?v=4?s=100" width="100px;" alt="benkrikler"/><br /><sub><b>benkrikler</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=benkrikler" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bfis"><img src="https://avatars0.githubusercontent.com/u/15651150?v=4?s=100" width="100px;" alt="bfis"/><br /><sub><b>bfis</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=bfis" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://ddavis.io/"><img src="https://avatars2.githubusercontent.com/u/3202090?v=4?s=100" width="100px;" alt="Doug Davis"/><br /><sub><b>Doug Davis</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=douglasdavis" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://twitter: @JoosepPata"><img src="https://avatars0.githubusercontent.com/u/69717?v=4?s=100" width="100px;" alt="Joosep Pata"/><br /><sub><b>Joosep Pata</b></sub></a><br /><a href="#ideas-jpata" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://martindurant.github.io/"><img src="https://avatars1.githubusercontent.com/u/6042212?v=4?s=100" width="100px;" alt="Martin Durant"/><br /><sub><b>Martin Durant</b></sub></a><br /><a href="#ideas-martindurant" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://gordonwatts.wordpress.com"><img src="https://avatars2.githubusercontent.com/u/1778366?v=4?s=100" width="100px;" alt="Gordon Watts"/><br /><sub><b>Gordon Watts</b></sub></a><br /><a href="#ideas-gordonwatts" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://gitlab.com/nikoladze"><img src="https://avatars0.githubusercontent.com/u/3707225?v=4?s=100" width="100px;" alt="Nikolai Hartmann"/><br /><sub><b>Nikolai Hartmann</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=nikoladze" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sjperkins"><img src="https://avatars3.githubusercontent.com/u/3530212?v=4?s=100" width="100px;" alt="Simon Perkins"/><br /><sub><b>Simon Perkins</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=sjperkins" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/drahnreb"><img src="https://avatars.githubusercontent.com/u/25883607?v=4?s=100" width="100px;" alt=".hard"/><br /><sub><b>.hard</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=drahnreb" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=drahnreb" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/HenryDayHall"><img src="https://avatars.githubusercontent.com/u/12996763?v=4?s=100" width="100px;" alt="HenryDayHall"/><br /><sub><b>HenryDayHall</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=HenryDayHall" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/agoose77"><img src="https://avatars.githubusercontent.com/u/1248413?v=4?s=100" width="100px;" alt="Angus Hollands"/><br /><sub><b>Angus Hollands</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=agoose77" title="Tests">⚠️</a> <a href="https://github.com/scikit-hep/awkward/commits?author=agoose77" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ioanaif"><img src="https://avatars.githubusercontent.com/u/9751871?v=4?s=100" width="100px;" alt="ioanaif"/><br /><sub><b>ioanaif</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=ioanaif" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=ioanaif" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://lizards.opensuse.org/author/bmwiedemann/"><img src="https://avatars.githubusercontent.com/u/637990?v=4?s=100" width="100px;" alt="Bernhard M. Wiedemann"/><br /><sub><b>Bernhard M. Wiedemann</b></sub></a><br /><a href="#maintenance-bmwiedemann" title="Maintenance">🚧</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://www.matthewfeickert.com/"><img src="https://avatars.githubusercontent.com/u/5142394?v=4?s=100" width="100px;" alt="Matthew Feickert"/><br /><sub><b>Matthew Feickert</b></sub></a><br /><a href="#maintenance-matthewfeickert" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SantamRC"><img src="https://avatars.githubusercontent.com/u/52635773?v=4?s=100" width="100px;" alt="Santam Roy Choudhury"/><br /><sub><b>Santam Roy Choudhury</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=SantamRC" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://jeroen.vangoey.be"><img src="https://avatars.githubusercontent.com/u/59344?v=4?s=100" width="100px;" alt="Jeroen Van Goey"/><br /><sub><b>Jeroen Van Goey</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=BioGeek" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ahmad-AlSubaie"><img src="https://avatars.githubusercontent.com/u/32343365?v=4?s=100" width="100px;" alt="Ahmad-AlSubaie"/><br /><sub><b>Ahmad-AlSubaie</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Ahmad-AlSubaie" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ManasviGoyal"><img src="https://avatars.githubusercontent.com/u/55101825?v=4?s=100" width="100px;" alt="Manasvi Goyal"/><br /><sub><b>Manasvi Goyal</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=ManasviGoyal" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aryan26roy"><img src="https://avatars.githubusercontent.com/u/50577809?v=4?s=100" width="100px;" alt="Aryan Roy"/><br /><sub><b>Aryan Roy</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=aryan26roy" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://saransh-cpp.github.io/"><img src="https://avatars.githubusercontent.com/u/74055102?v=4?s=100" width="100px;" alt="Saransh"/><br /><sub><b>Saransh</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Saransh-cpp" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Laurits7"><img src="https://avatars.githubusercontent.com/u/30724920?v=4?s=100" width="100px;" alt="Laurits Tani"/><br /><sub><b>Laurits Tani</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Laurits7" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dsavoiu"><img src="https://avatars.githubusercontent.com/u/17005255?v=4?s=100" width="100px;" alt="Daniel Savoiu"/><br /><sub><b>Daniel Savoiu</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=dsavoiu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://sites.google.com/view/raybellwaves/home"><img src="https://avatars.githubusercontent.com/u/17162724?v=4?s=100" width="100px;" alt="Ray Bell"/><br /><sub><b>Ray Bell</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=raybellwaves" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://zonca.dev"><img src="https://avatars.githubusercontent.com/u/383090?v=4?s=100" width="100px;" alt="Andrea Zonca"/><br /><sub><b>Andrea Zonca</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=zonca" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chrisburr"><img src="https://avatars.githubusercontent.com/u/5220533?v=4?s=100" width="100px;" alt="Chris Burr"/><br /><sub><b>Chris Burr</b></sub></a><br /><a href="#infra-chrisburr" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zbilodea"><img src="https://avatars.githubusercontent.com/u/70441641?v=4?s=100" width="100px;" alt="Zoë Bilodeau"/><br /><sub><b>Zoë Bilodeau</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=zbilodea" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/raymondEhlers"><img src="https://avatars.githubusercontent.com/u/1571927?v=4?s=100" width="100px;" alt="Raymond Ehlers"/><br /><sub><b>Raymond Ehlers</b></sub></a><br /><a href="#maintenance-raymondEhlers" title="Maintenance">🚧</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.mloning.com/"><img src="https://avatars.githubusercontent.com/u/21020482?v=4?s=100" width="100px;" alt="Markus Löning"/><br /><sub><b>Markus Löning</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=mloning" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kkothari2001"><img src="https://avatars.githubusercontent.com/u/53650538?v=4?s=100" width="100px;" alt="Kush Kothari"/><br /><sub><b>Kush Kothari</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=kkothari2001" title="Code">💻</a> <a href="https://github.com/scikit-hep/awkward/commits?author=kkothari2001" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jrueb"><img src="https://avatars.githubusercontent.com/u/30041073?v=4?s=100" width="100px;" alt="Jonas Rübenach"/><br /><sub><b>Jonas Rübenach</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=jrueb" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://blog.jling.dev"><img src="https://avatars.githubusercontent.com/u/5306213?v=4?s=100" width="100px;" alt="Jerry Ling"/><br /><sub><b>Jerry Ling</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=Moelf" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lobis"><img src="https://avatars.githubusercontent.com/u/35803280?v=4?s=100" width="100px;" alt="Luis Antonio Obis Aparicio"/><br /><sub><b>Luis Antonio Obis Aparicio</b></sub></a><br /><a href="https://github.com/scikit-hep/awkward/commits?author=lobis" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

💻: code, 📖: documentation, 🚇: infrastructure, 🚧: maintenance, ⚠: tests and feedback, 🤔: foundational ideas.
