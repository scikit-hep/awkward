# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout

__all__ = ("from_cudf",)


@high_level_function()
def from_cudf(obj, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        obj (cudf.Series or cudf.DataFrame): The cuDF object to convert into an
            Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Converts a cuDF Series or DataFrame into an Awkward Array by recursively
    traversing pylibcudf columns and wrapping GPU buffers with CuPy.

    Note: DataFrame column names and order are preserved. Arrow/cuDF column
    metadata beyond field names (e.g. time-zone annotations,
    extension-type metadata) is not yet propagated and will be addressed in a
    follow-up.

    See also #ak.to_cudf, #ak.from_cupy, and #ak.from_dlpack.
    """
    # Dispatch
    yield (obj,)

    # Implementation
    return _impl(obj, highlevel, behavior, attrs)


def _impl(obj, highlevel, behavior, attrs):
    from awkward._connect.cudf import (
        _dataframe_to_layout,
        _ensure_deps,
        _series_to_layout,
    )

    cudf, _, _ = _ensure_deps()
    if isinstance(obj, cudf.Series):
        layout = _series_to_layout(obj)
    elif isinstance(obj, cudf.DataFrame):
        layout = _dataframe_to_layout(obj)
    else:
        raise TypeError(
            "ak.from_cudf accepts only cudf.Series or cudf.DataFrame, "
            f"not {type(obj).__name__!r}"
        )

    return wrap_layout(
        layout,
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
