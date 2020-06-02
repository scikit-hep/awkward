---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: 1.4.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

![](https://raw.githubusercontent.com/scikit-hep/awkward-1.0/master/docs-img/logo/logo-300px.png)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they’re not.

Quickstart
==========

<table style="margin-top: 30px">
  <tr>
    <td width="50%" valign="top" align="center">
      <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-sphinx.png" width="80%">
      </a>
      <p align="center" style="margin-top: 10px"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        Python<br>API reference
        </a>
      </b></p>
    </td>
    <td width="50%" valign="top" align="center">
      <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-doxygen.png" width="80%">
      </a>
      <p align="center" style="margin-top: 10px"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        C++<br>API reference
        </a>
      </b></p>
    </td>
  </tr>
  <tr style="margin-top: 20px">
    <td width="50%" valign="top" align="center">
      <a href="how-to.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-data-analysts.png" width="80%">
      </a>
      <p align="center" style="margin-top: 10px"><b>
        <a href="how-to.html">
        How-to documentation<br>for data analysts
        </a>
      </b></p>
    </td>
    <td width="50%" valign="top" align="center">
      <a href="how-it-works.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-developers.png" width="80%">
      </a>
      <p align="center" style="margin-top: 10px"><b>
        <a href="how-it-works.html">
        How-it-works tutorials<br>for developers
        </a>
      </b></p>
    </td>
  </tr>
</table>

Installation
------------

Awkward Array can be installed [from PyPI](https://pypi.org/project/awkward1/) using pip:

```bash
pip install awkward1
```

Most users will get a precompiled binary (wheel) for your operating system and Python version. If not, the above attempts to compile from source.

Getting help
------------

Report bugs, request features, and ask for additional documentation on [GitHub Issues](https://github.com/scikit-hep/awkward-1.0/issues). If you have a general "How do I…?" question, we'll answer it as a new [example in the tutorial](how-to).

If you have a problem that's too specific to be new documentation or it isn't exclusively related to Awkward Array, it might be more appropriate to ask on [StackOverflow with the [awkward-array] tag](https://stackoverflow.com/questions/tagged/awkward-array). Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.

The [Gitter Scikit-HEP/community](https://gitter.im/Scikit-HEP/community) is a way to get in touch with all Scikit-HEP developers and users.

For developers
--------------

See Awkward Array's [GitHub page](https://github.com/scikit-hep/awkward-1.0) for more on the following.

   * [Insatllation for developers](https://github.com/scikit-hep/awkward-1.0#installation-for-developers).
   * [Using Awkward as a dependency](https://github.com/scikit-hep/awkward-1.0/tree/master/dependent-project) (example projects).
   * [Continuous integration](https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=3&_a=summary) and [continuous deployment](https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=4&_a=summary) are hosted by [Azure Pipelines](https://azure.microsoft.com/en-us/services/devops/pipelines/).
   * [Release history](https://awkward-array.readthedocs.io/en/latest/_auto/changelog.html) (changelog) is hosted by [ReadTheDocs](https://readthedocs.org).
   * [awkward-array.org](https://awkward-array.org) (this site) is hosted by [Netlify](https://www.netlify.com).
   * [CONTRIBUTING.md](https://github.com/scikit-hep/awkward-1.0/blob/master/CONTRIBUTING.md) for technical information on how to contribute.
   * [Code of conduct](https://scikit-hep.org/code-of-conduct) for how we work together.
   * The [LICENSE](https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE) is BSD-3.

See also [papers and talks about Awkward Array](https://github.com/scikit-hep/awkward-1.0#papers-and-talks-about-awkward-array) and a [table of contributors](https://github.com/scikit-hep/awkward-1.0#acknowledgements).
