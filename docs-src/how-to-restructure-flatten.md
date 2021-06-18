---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to flatten arrays, especially for plotting
==============================================

In a data analysis, it is important to plot your data frequently, and the interactive nature of array-at-a-time functions facilitate that.

However, plotting views your data as a generic set or sequence—the structure of nested lists and records can't be captured by standard plots. Histograms (including 2-dimensional heatmaps) take input data to be an unordered set, as do scatter plots. Connected-line plots, such as time-series, use the sequential order of the data, but there aren't many visualizations that show nestedness. (Maybe there will be, in the future.)

As such, these standard plotting routines expect simple structures, either a single flat array (in which the order may be relevant or irrelevant) or several same-length arrays (in which the relative or absolute order is relevant). Encountering an Awkward Array, they may try to call `np.asarray` on it, which only works if the array can be made rectilinear or they may try to iterate over it in Python, which can be prohibitively slow if the dataset is large.

+++
