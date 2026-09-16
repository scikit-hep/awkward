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

How to compute statistics on dimensions (mean/var/std)
======================================================

Awkward Array provides several functions for statistical analysis that operate on ragged arrays. These are dimensional reducers, like {func}`ak.sum`, {func}`ak.min`, {func}`ak.any`, and {func}`ak.all` in the {doc}`previous section <how-to-math-reducing>`, but they compute quantities such as mean, variance, standard deviation, and higher moments, as well as functions for linear regression and correlation.

```{code-cell}
import awkward as ak
import numpy as np
```

## Basic statistical functions

### Mean, variance, and standard deviation

To compute the [mean](https://en.wikipedia.org/wiki/Mean), [variance](https://en.wikipedia.org/wiki/Variance), and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) of an array, use {func}`ak.mean`, {func}`ak.var`, and {func}`ak.std`. Unlike the NumPy functions with the same names, these functions apply to arrays with variable-length dimensions and missing values (but not heterogeneous dimensionality or records; see the last section of {doc}`reducing <how-to-math-reducing>`.

```{code-cell}
array = ak.Array([[0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
```

```{code-cell}
ak.mean(array, axis=-1)
```

```{code-cell}
ak.var(array, axis=-1)
```

```{code-cell}
ak.std(array, axis=-1)
```

These functions also have counterparts that ignore `nan` values: {func}`ak.nanmean`, {func}`ak.nanvar`, and {func}`ak.nanstd`.

```{code-cell}
array_with_nan = ak.Array([[0, 1.1, np.nan], [3.3, 4.4], [np.nan], [6.6, np.nan, 8.8, 9.9]])
```

```{code-cell}
ak.nanmean(array_with_nan, axis=-1)
```

```{code-cell}
ak.nanvar(array_with_nan, axis=-1)
```

```{code-cell}
ak.nanstd(array_with_nan, axis=-1)
```

Note that floating-point `nan` is different from missing values (`None`). Unlike `nan`, integer arrays can have missing values, and whole lists can be missing as well. For both types of functions, missing values are ignored if they are in the dimension being reduced or pass through a function to the output otherwise, just as the `nan`-ignoring functions ignore `nan`.

```{code-cell}
array_with_None = ak.Array([[0, 1.1, 2.2], None, [None, 4.4], [5.5], [6.6, np.nan, 8.8, 9.9]])
```

```{code-cell}
ak.mean(array_with_None, axis=-1)
```

```{code-cell}
ak.nanmean(array_with_None, axis=-1)
```

### Moments

For higher moments, use {func}`ak.moment`. For example, to calculate the third [moment](https://en.wikipedia.org/wiki/Moment_(mathematics)) (skewness), you would do the following:

```{code-cell}
ak.moment(array, 3, axis=-1)
```

## Correlation and covariance

For [correlation](https://en.wikipedia.org/wiki/Correlation) and [covariance](https://en.wikipedia.org/wiki/Covariance) between two arrays, use {func}`ak.corr` and {func}`ak.covar`.

```{code-cell}
array_x = ak.Array([[0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
array_y = ak.Array([[0, 1, 2], [3, 4], [5], [6, 7, 8, 9]])
```

```{code-cell}
ak.corr(array_x, array_y, axis=-1)
```

```{code-cell}
ak.covar(array_x, array_y, axis=-1)
```

## Linear fits

To perform [linear fits](https://en.wikipedia.org/wiki/Linear_regression), use {func}`ak.linear_fit`. Instead of reducing each list to a number, it reduces each list to a record that has `intercept`, `slope`, `intercept_error`, and `slope_error` fields. (These "errors" are uncertainty estimates of the intercept and slope parameters, assuming that the underlying generator of data is truly linear.)

```{code-cell}
ak.linear_fit(array_x, array_y, axis=-1)
```

[Ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) linear fits can be computed by a formula, without approximation or iteration, so it can be thought of like computing the mean or other moments, but with greater fidelity to the data because it models a general correlation. For example, some statistical models achieve high granularity by segmenting a dataset in some meaningful way and then summarizing the data in each segment (such as a regression [decision tree](https://en.wikipedia.org/wiki/Decision_tree)). Performing linear fits on each segment fine-tunes the model more than performing just taking the average of data in each segment.

+++

## Peak to peak

The peak-to-peak function {func}`ak.ptp` can be used to find the range (maximum - minimum) of data along an axis. It's more convenient than calling {func}`ak.min` and {func}`ak.max` separately.

```{code-cell}
ak.ptp(array, axis=-1)
```

## Softmax

The [softmax](https://en.wikipedia.org/wiki/Softmax_function) function is useful in machine learning, particularly in the context of logistic regression and neural networks. Awkward Array provides {func}`ak.softmax` to compute softmax values of an array.

Note that this function does not _reduce_ a dimension; it computes one output value for each input value, but each output value is normalized by all the other values in the same list.

Also note that only `axis=-1` (innermost lists) is supported by {func}`ak.softmax`.

```{code-cell}
ak.softmax(array, axis=-1)
```

## Example uses in data analysis

Here is an example that normalizes an input array to have an overall mean of 0 and standard deviation of 1:

```{code-cell}
array = ak.Array([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
```

```{code-cell}
(array - ak.mean(array)) / ak.std(array)
```

And here's another example that normalizes each _list_ within the array to each have a mean of 0 and a standard deviation of 1:

```{code-cell}
(array - ak.mean(array, axis=-1)) / ak.std(array, axis=-1)
```
