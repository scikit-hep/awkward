---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Splitting and joining strings

+++

Strings in Awkward Array can arbitrarily be joined together, and split into sublists. Let's start by creating an array of strings that we can later manipulate. The following `timestamps` array contains a list of timestamp-like strings

```{code-cell} ipython3
import awkward as ak
timestamp = ak.from_iter(
    [
        "12-17 19:31:36.263",
        "12-17 19:31:36.263",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.264",
        "12-17 19:31:36.265",
        "12-17 19:31:36.265",
        "12-17 19:31:36.265",
        "12-17 19:31:36.265",
        "12-17 19:31:36.265",
        "12-17 19:31:36.265",
        "12-17 19:31:36.267",
        "12-17 19:31:36.270",
        "12-17 19:31:36.271",
        "12-17 19:31:36.275",
        "12-17 19:31:36.275",
        "12-17 19:31:36.275",
        "12-17 19:31:36.276",
        "12-17 19:31:36.278",
        "12-17 19:31:36.279",
        "12-17 19:31:36.279",
        "12-17 19:31:36.279",
        "12-17 19:31:36.280",
        "12-17 19:31:36.280",
        "12-17 19:31:36.280",
        "12-17 19:31:36.280",
        "12-17 19:31:36.280",
        "12-17 19:31:36.280",
        "12-17 19:31:36.280",
        "12-17 19:31:36.281",
        "12-17 19:31:36.282",
        "12-17 19:31:36.283",
        "12-17 19:31:36.284",
        "12-17 19:31:36.285",
        "12-17 19:31:36.285",
        "12-17 19:31:36.289",
        "12-17 19:31:36.295",
        "12-17 19:31:36.297",
        "12-17 19:31:36.297",
        "12-17 19:31:36.298",
        "12-17 19:31:36.299",
        "12-17 19:31:36.300",
        "12-17 19:31:36.301",
        "12-17 19:31:36.301",
        "12-17 19:31:36.301",
        "12-17 19:31:36.301",
        "12-17 19:31:36.301",
        "12-17 19:31:36.301",
        "12-17 19:31:36.302",
        "12-17 19:31:36.304",
        "12-17 19:31:36.311",
        "12-17 19:31:36.311",
        "12-17 19:31:36.311",
        "12-17 19:31:36.311",
        "12-17 19:31:36.313",
    ]
)
```

## Joining strings together

Parsing datetimes in a performant manner is tricky. Pandas has such an ability, but it uses NumPy's fixed-width strings. Arrow provides `strptime`, but it does not handle fractional seconds or timedeltas and requires a full date. In order to use Arrow's {func}`pyarrow.compute.strptime` function, we can manipulate the string to prepend the date, operating only on the non-fraction part of the match.

+++

Let's assume that these timestamps were recorded in the year 2022. We can prepend the string "2022" with the "-" delimiter to complete the timestamp string

```{code-cell} ipython3
timestamp_with_year = ak.str.join_element_wise(["2022"], timestamp, ["-"])
timestamp_with_year
```

The `["2022"]` and `["-"]` arrays are broadcast with the `timestamp` array before joining element-wise.

+++

{func}`ak.str.join_element_wise` is useful for building new strings from separate arrays. It might also be the case that one has a single array of strings that they wish to join along the final axis (like a reducer). There exists a separate function {func}`ak.str.join` for such a purpose

```{code-cell} ipython3
ak.str.join(
    [
        ["do", "re", "me"],
        ["fa", "so"],
        ["la"],
        ["ti", "da"],
    ],
    separator="-🎵-",
)
```

## Splitting strings apart

+++

The timestamps above still cannot be parsed by Arrow; the fractional time component is not (at time of writing) yet supported. To fix this, we can split the fractional component from the timestamp, and add it as a `timedelta64[ms]` later on.

+++

Let's split the fractional time component into two parts using {func}`ak.str.split_pattern`.

```{code-cell} ipython3
timestamp_split = ak.str.split_pattern(timestamp_with_year, ".", max_splits=1)
timestamp_split
```

```{code-cell} ipython3
timestamp_non_fractional = timestamp_split[:, 0]
timestamp_fractional = timestamp_split[:, 1]
```

Now we can parse these timestamps using Arrow!

```{code-cell} ipython3
import pyarrow.compute

datetime = ak.from_arrow(
    pyarrow.compute.strptime(
        ak.to_arrow(timestamp_non_fractional, extensionarray=False),
        "%Y-%m-%d %H:%M:%S",
        "ms",
    )
)
datetime
```

Finally, we build an offset for the fractional component (in milliseconds) using {func}`ak.strings_astype`

```{code-cell} ipython3
import numpy as np

datetime_offset = ak.strings_astype(timestamp_fractional, np.dtype("timedelta64[ms]"))
datetime_offset
```

This offset is added to the absolute datetime to obtain a timestamp

```{code-cell} ipython3
timestamp = datetime + datetime_offset
timestamp
```

If we had a different parsing library that could only handle dates and times separately, then we could also split on the whitespace. Although {func}`ak.str.split_pattern` supports whitespace, it is more performant (and versatile) to use {func}`ak.str.split_whitespace`

```{code-cell} ipython3
ak.str.split_whitespace(timestamp_with_year)
```

If we also needed to split off the fractional component (and manually build the time delta), then we could have used {func}`ak.str.split_pattern_regex` to split on both whitespace *and* the period

```{code-cell} ipython3
ak.str.split_pattern_regex(timestamp_with_year, r"\.|\s")
```
