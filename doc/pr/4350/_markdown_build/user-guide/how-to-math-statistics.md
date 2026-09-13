# How to compute statistics on dimensions (mean/var/std)

Awkward Array provides several functions for statistical analysis that operate on ragged arrays. These are dimensional reducers, like [`ak.sum()`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum), [`ak.min()`](sphinx-llm:d40aa9e4100d4897a291c711bc7a7042#ak.min), [`ak.any()`](sphinx-llm:df7c468b43ca4558ab741d771ffbf0dc#ak.any), and [`ak.all()`](sphinx-llm:6d63d32372de42f69a3faba14948813c#ak.all) in the [previous section](sphinx-llm:be92daa510d04afcba9a0da0b0119a60), but they compute quantities such as mean, variance, standard deviation, and higher moments, as well as functions for linear regression and correlation.

```ipython3
import awkward as ak
import numpy as np
```

## Basic statistical functions

### Mean, variance, and standard deviation

To compute the [mean](https://en.wikipedia.org/wiki/Mean), [variance](https://en.wikipedia.org/wiki/Variance), and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) of an array, use [`ak.mean()`](sphinx-llm:9dfbdca7e4104165863b3b3f4391b21e#ak.mean), [`ak.var()`](sphinx-llm:a102203a874241f493f04760ce9ca52a#ak.var), and [`ak.std()`](sphinx-llm:072c087e1e3f4eabab83ac68c00b886b#ak.std). Unlike the NumPy functions with the same names, these functions apply to arrays with variable-length dimensions and missing values (but not heterogeneous dimensionality or records; see the last section of [reducing](sphinx-llm:be92daa510d04afcba9a0da0b0119a60).

```ipython3
array = ak.Array([[0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
```

```ipython3
ak.mean(array, axis=-1)
```

```ipython3
ak.var(array, axis=-1)
```

```ipython3
ak.std(array, axis=-1)
```

These functions also have counterparts that ignore `nan` values: [`ak.nanmean()`](sphinx-llm:d58141c9dfb5425b9cfedecf70a020e5#ak.nanmean), [`ak.nanvar()`](sphinx-llm:13e7f76a3f70493093b4c1274ee0cb21#ak.nanvar), and [`ak.nanstd()`](sphinx-llm:04d355ec5f954c2496e15e883ce67410#ak.nanstd).

```ipython3
array_with_nan = ak.Array([[0, 1.1, np.nan], [3.3, 4.4], [np.nan], [6.6, np.nan, 8.8, 9.9]])
```

```ipython3
ak.nanmean(array_with_nan, axis=-1)
```

```ipython3
ak.nanvar(array_with_nan, axis=-1)
```

```ipython3
ak.nanstd(array_with_nan, axis=-1)
```

Note that floating-point `nan` is different from missing values (`None`). Unlike `nan`, integer arrays can have missing values, and whole lists can be missing as well. For both types of functions, missing values are ignored if they are in the dimension being reduced or pass through a function to the output otherwise, just as the `nan`-ignoring functions ignore `nan`.

```ipython3
array_with_None = ak.Array([[0, 1.1, 2.2], None, [None, 4.4], [5.5], [6.6, np.nan, 8.8, 9.9]])
```

```ipython3
ak.mean(array_with_None, axis=-1)
```

```ipython3
ak.nanmean(array_with_None, axis=-1)
```

### Moments

For higher moments, use [`ak.moment()`](sphinx-llm:4af063f29db5434da52399f813a1d334#ak.moment). For example, to calculate the third [moment](https://en.wikipedia.org/wiki/Moment_(mathematics)) (skewness), you would do the following:

```ipython3
ak.moment(array, 3, axis=-1)
```

## Correlation and covariance

For [correlation](https://en.wikipedia.org/wiki/Correlation) and [covariance](https://en.wikipedia.org/wiki/Covariance) between two arrays, use [`ak.corr()`](sphinx-llm:e0d2b718f1254120ad3457fc17fb8b5c#ak.corr) and [`ak.covar()`](sphinx-llm:3ffb5fa606814cdeb900cb06a40e997c#ak.covar).

```ipython3
array_x = ak.Array([[0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
array_y = ak.Array([[0, 1, 2], [3, 4], [5], [6, 7, 8, 9]])
```

```ipython3
ak.corr(array_x, array_y, axis=-1)
```

```ipython3
ak.covar(array_x, array_y, axis=-1)
```

## Linear fits

To perform [linear fits](https://en.wikipedia.org/wiki/Linear_regression), use [`ak.linear_fit()`](sphinx-llm:2460991f798a4ae9b63748d7cebeca00#ak.linear_fit). Instead of reducing each list to a number, it reduces each list to a record that has `intercept`, `slope`, `intercept_error`, and `slope_error` fields. (These “errors” are uncertainty estimates of the intercept and slope parameters, assuming that the underlying generator of data is truly linear.)

```ipython3
ak.linear_fit(array_x, array_y, axis=-1)
```

[Ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) linear fits can be computed by a formula, without approximation or iteration, so it can be thought of like computing the mean or other moments, but with greater fidelity to the data because it models a general correlation. For example, some statistical models achieve high granularity by segmenting a dataset in some meaningful way and then summarizing the data in each segment (such as a regression [decision tree](https://en.wikipedia.org/wiki/Decision_tree)). Performing linear fits on each segment fine-tunes the model more than performing just taking the average of data in each segment.

## Peak to peak

The peak-to-peak function [`ak.ptp()`](sphinx-llm:8d4491031e5e4221b645d7275e922799#ak.ptp) can be used to find the range (maximum - minimum) of data along an axis. It’s more convenient than calling [`ak.min()`](sphinx-llm:d40aa9e4100d4897a291c711bc7a7042#ak.min) and [`ak.max()`](sphinx-llm:8d04431ada5f4197aee6b08d9430b68f#ak.max) separately.

```ipython3
ak.ptp(array, axis=-1)
```

## Softmax

The [softmax](https://en.wikipedia.org/wiki/Softmax_function) function is useful in machine learning, particularly in the context of logistic regression and neural networks. Awkward Array provides [`ak.softmax()`](sphinx-llm:011cd8890ef147e5b68eaff5571c15f4#ak.softmax) to compute softmax values of an array.

Note that this function does not *reduce* a dimension; it computes one output value for each input value, but each output value is normalized by all the other values in the same list.

Also note that only `axis=-1` (innermost lists) is supported by [`ak.softmax()`](sphinx-llm:011cd8890ef147e5b68eaff5571c15f4#ak.softmax).

```ipython3
ak.softmax(array, axis=-1)
```

## Example uses in data analysis

Here is an example that normalizes an input array to have an overall mean of 0 and standard deviation of 1:

```ipython3
array = ak.Array([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
```

```ipython3
(array - ak.mean(array)) / ak.std(array)
```

And here’s another example that normalizes each *list* within the array to each have a mean of 0 and a standard deviation of 1:

```ipython3
(array - ak.mean(array, axis=-1)) / ak.std(array, axis=-1)
```
