# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout

__all__ = ("from_cudf",)


@high_level_function()
def from_cudf(series, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        series (cudf.Series): The cuDF Series to convert into an Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Converts a cuDF Series into an Awkward Array by recursively traversing
    the underlying pylibcudf column structure.

    Primitive, boolean, list, struct, string, and nullable columns are
    supported. The conversion is zero-copy on GPU for numeric buffers via
    CuPy. Other column types raise ``NotImplementedError``.

    See also #ak.to_cudf, #ak.from_cupy, and #ak.from_dlpack.
    """
    # Dispatch
    yield (series,)

    # Implementation
    return _impl(series, highlevel, behavior, attrs)


def _impl(series, highlevel, behavior, attrs):
    from awkward._connect.cudf import from_cudf as connect_from_cudf

    layout = connect_from_cudf(series)
    return wrap_layout(
        layout,
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
