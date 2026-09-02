# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from awkward.contents import (
    BitMaskedArray,
    IndexedArray,
    IndexedOptionArray,
    ListOffsetArray,
    NumpyArray,
    RecordArray,
)
from awkward.index import Index32, Index64, IndexU8

if TYPE_CHECKING:
    import cupy as cp
    import pylibcudf as plc

    from awkward.contents.content import Content


_ISSUE_URL = "https://github.com/scikit-hep/awkward/issues"


@cache
def _ensure_deps() -> tuple[Any, Any, Any]:
    """Return (cudf, pylibcudf, cupy) or raise ImportError."""
    try:
        import cudf
    except ImportError:
        raise ImportError(
            "ak.from_cudf requires the 'cudf' package.\n"
            "Install for CUDA 12:  pip install cudf-cu12\n"
            "Install for CUDA 13:  pip install cudf-cu13"
        ) from None

    try:
        import pylibcudf as plc
    except ImportError:
        raise ImportError(
            "ak.from_cudf requires pylibcudf >= 25.02, which ships with cudf >= 25.02."
        ) from None

    try:
        import cupy as cp
    except ImportError:
        raise ImportError("ak.from_cudf requires the 'cupy' package.") from None

    return cudf, plc, cp


def _get_attr_or_call(obj: Any, name: str) -> Any | None:
    """
    Return ``obj.name`` if it is an attribute, or ``obj.name()`` if it is a
    method.

    pylibcudf has changed some APIs between property and zero-argument method
    forms across releases, so this keeps compatibility logic in one place.
    """
    val = getattr(obj, name, None)
    if val is None:
        return None
    return val() if callable(val) else val


def _get_size(col: plc.Column) -> int:
    value = _get_attr_or_call(col, "size")
    return int(value) if value is not None else 0


def _get_offset(col: plc.Column) -> int:
    value = _get_attr_or_call(col, "offset")
    return int(value) if value is not None else 0


def _get_null_count(col: plc.Column) -> int:
    value = _get_attr_or_call(col, "null_count")
    return int(value) if value is not None else 0


def _get_num_children(col: plc.Column) -> int:
    value = _get_attr_or_call(col, "num_children")
    if value is not None:
        return int(value)

    children = _get_attr_or_call(col, "children")
    return len(children) if children is not None else 0


def _type_id_class(plc_module: Any) -> Any:
    return getattr(plc_module, "TypeId", None) or plc_module.types.TypeId


def _type_id(plc_module: Any, name: str) -> Any:
    return getattr(_type_id_class(plc_module), name, None)


def _primitive_dtypes(plc_module: Any) -> dict[Any, str]:
    names_and_dtypes = (
        ("INT8", "int8"),
        ("INT16", "int16"),
        ("INT32", "int32"),
        ("INT64", "int64"),
        ("UINT8", "uint8"),
        ("UINT16", "uint16"),
        ("UINT32", "uint32"),
        ("UINT64", "uint64"),
        ("FLOAT32", "float32"),
        ("FLOAT64", "float64"),
        ("BOOL8", "bool"),
        ("DURATION_DAYS", "timedelta64[D]"),
        ("DURATION_SECONDS", "timedelta64[s]"),
        ("DURATION_MILLISECONDS", "timedelta64[ms]"),
        ("DURATION_MICROSECONDS", "timedelta64[us]"),
        ("DURATION_NANOSECONDS", "timedelta64[ns]"),
        ("TIMESTAMP_DAYS", "datetime64[D]"),
        ("TIMESTAMP_SECONDS", "datetime64[s]"),
        ("TIMESTAMP_MILLISECONDS", "datetime64[ms]"),
        ("TIMESTAMP_MICROSECONDS", "datetime64[us]"),
        ("TIMESTAMP_NANOSECONDS", "datetime64[ns]"),
    )

    out = {}
    for name, dtype in names_and_dtypes:
        type_id = _type_id(plc_module, name)
        if type_id is not None:
            out[type_id] = dtype
    return out


def _to_pylibcudf_column(series: Any) -> plc.Column:
    try:
        result = series.to_pylibcudf()
    except AttributeError:
        cudf, _, _ = _ensure_deps()
        raise RuntimeError(
            "cudf.Series.to_pylibcudf() is not available.  "
            "ak.from_cudf requires cudf >= 25.02; your installed "
            f"version is {getattr(cudf, '__version__', 'unknown')}.  "
            "Please upgrade: pip install 'cudf-cu12>=25.02' or "
            "            pip install 'cudf-cu13>=25.02'."
        ) from None

    return result[0] if isinstance(result, tuple) else result


def _buf_to_cupy(buf: Any, dtype: str) -> cp.ndarray:
    """
    Wrap a pylibcudf gpumemoryview as a CuPy array without copying.

    Missing buffers represent empty columns in pylibcudf, so they become
    zero-length CuPy arrays of the requested dtype.
    """
    _, _, cp = _ensure_deps()

    # Return a zero-length array rather than None so every caller
    # gets a uniform cp.ndarray regardless of whether the column's
    # buffer is allocated.  None would force callers to
    # special-case empty columns and risks losing dtype information.
    if buf is None:
        return cp.empty(0, dtype=dtype)

    dtype = cp.dtype(dtype)

    # Invariant: a well-formed pylibcudf buffer must contain an integral
    # number of dtype-sized elements.  A mismatch indicates buffer metadata
    # that cannot be safely wrapped as a strided CuPy array.
    total_bytes = int(buf.size)
    if total_bytes % dtype.itemsize != 0:
        full_elements = total_bytes // dtype.itemsize
        expected_bytes = (full_elements + 1) * dtype.itemsize
        raise RuntimeError(
            f"Buffer size mismatch: expected a multiple of {dtype.itemsize} bytes "
            f"({expected_bytes} bytes for {full_elements + 1} elements) but got "
            f"{total_bytes} bytes.  This may indicate "
            f"a sliced column where offset metadata was not "
            f"propagated."
        )

    # owner=buf keeps a Python reference to the pylibcudf
    # gpumemoryview alive for the entire lifetime of the CuPy
    # allocation.  Without it, the underlying GPU memory could be
    # freed while the cp.ndarray is still live.
    mem = cp.cuda.UnownedMemory(ptr=int(buf.ptr), size=int(buf.size), owner=buf)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    length = int(buf.size) // dtype.itemsize
    # cp.ndarray accepts memptr= (unlike np.ndarray); shape=()
    # form silences linters that conflate the two APIs.
    return cp.ndarray(shape=(length,), dtype=dtype, memptr=memptr)  # type: ignore[call-arg]  # pylint: disable=unexpected-keyword-arg


def _data_to_cupy(col: plc.Column, dtype: str) -> cp.ndarray:
    _, _, cp = _ensure_deps()
    buffer = _get_attr_or_call(col, "data_buffer") or _get_attr_or_call(col, "data")
    offset = _get_offset(col)
    size = _get_size(col)
    length = offset + size

    if buffer is None or buffer.nbytes == 0:
        if size == 0:
            return cp.empty(0, dtype=dtype)
        if _get_null_count(col) == size:
            return cp.empty(length, dtype=dtype)
        raise RuntimeError("non-empty pylibcudf column is missing its data buffer")

    data = _buf_to_cupy(buffer, dtype)
    buf_length = int(data.size)
    dtype = data.dtype

    # Invariant: for a well-formed pylibcudf column the data buffer must
    # cover the logical slice described by offset + length.  A mismatch
    # indicates a sliced column whose offset metadata was not propagated
    # correctly; overrunning would silently corrupt data.
    if buf_length < length:
        raise RuntimeError(
            f"Buffer size mismatch: expected {length} elements "
            f"({length * dtype.itemsize} bytes) but got "
            f"{buf_length} ({int(buffer.size)} bytes).  This may indicate "
            f"a sliced column where offset metadata was not "
            f"propagated."
        )

    return data


def _offset_dtype(offsets_col: plc.Column) -> str:
    _, plc_module, _ = _ensure_deps()
    type_id = offsets_col.type().id()
    if type_id == _type_id(plc_module, "INT64"):
        return "int64"
    else:
        return "int32"


def _offsets_to_index(offsets_col: plc.Column | None, parent_col: plc.Column) -> Any:
    _, _, cp = _ensure_deps()
    fallback_length = max(_get_offset(parent_col) + _get_size(parent_col) + 1, 1)

    if offsets_col is None:
        offsets = cp.zeros(fallback_length, dtype="int32")
    else:
        buffer = _get_attr_or_call(offsets_col, "data_buffer") or _get_attr_or_call(
            offsets_col, "data"
        )
        if buffer is None:
            if _get_size(offsets_col) == 0:
                offsets = cp.zeros(fallback_length, dtype="int32")
            else:
                raise RuntimeError(
                    "non-empty pylibcudf offsets column is missing its data buffer"
                )
        else:
            offsets = _buf_to_cupy(buffer, _offset_dtype(offsets_col))
            if offsets.size == 0:
                offsets = cp.zeros(fallback_length, dtype="int32")

    if offsets.dtype == cp.dtype("int32"):
        return Index32(offsets)
    elif offsets.dtype == cp.dtype("int64"):
        return Index64(offsets)
    else:
        return Index64(offsets.astype(cp.int64))


def _struct_field_names(col: plc.Column) -> list[str]:
    """
    Return struct field names from a raw pylibcudf Column.

    pylibcudf Column objects carry no field-name metadata — names live only in
    the Python-level cudf dtype objects.  This function is therefore a pure
    positional fallback used when no cudf dtype is available.  Callers that
    have access to the Python-level cudf dtype should pass it via the ``dtype``
    parameter of ``_column_to_layout`` instead of relying on this function.
    """
    num_children = _get_num_children(col)
    return [str(i) for i in range(num_children)]


def _list_offsets_and_content(
    col: plc.Column,
) -> tuple[plc.Column | None, plc.Column]:
    if _get_num_children(col) >= 2:
        return col.child(0), col.child(1)
    else:
        return _get_attr_or_call(col, "offsets"), col.child(0)


def _string_offsets_and_chars(
    col: plc.Column,
) -> tuple[plc.Column | None, plc.Column]:
    """
    Return (offsets_col, chars_source) for a STRING column.

    pylibcudf STRING columns have exactly 1 child: the offsets column.
    The character bytes live in the parent column's own data buffer, so
    ``chars_source`` is the parent column ``col`` itself — callers extract
    chars via ``_string_chars_to_cupy(col, chars_source, offsets)``.

    The ``>= 2`` branch is defensive dead code against hypothetical future
    API changes; in all current pylibcudf versions ``num_children == 1``.
    If a degenerate column with 0 children is encountered (e.g. from an
    internal intermediate result) we return ``(None, col)`` so the caller
    still gets an empty chars buffer from the column's own data.
    """
    num_children = _get_num_children(col)
    if num_children >= 2:
        return col.child(0), col.child(1)
    elif num_children == 1:
        # Normal case: child(0) is offsets, chars are in col.data_buffer().
        return col.child(0), col
    else:
        # Degenerate: no children, no offsets — treat as empty.
        return None, col


def _string_chars_to_cupy(
    col: plc.Column, chars_col: plc.Column, offsets: cp.ndarray
) -> cp.ndarray:
    """
    Return the character bytes of a STRING column as a uint8 CuPy array.

    A STRING column's ``size`` counts strings rather than bytes, so its
    character buffer cannot go through ``_data_to_cupy``: ``["", "x", ""]``
    is three elements but only one byte, and ``["", ""]`` has no character
    buffer at all.  The number of bytes the strings actually occupy is
    recorded only in the final offset, so that is what is checked here.
    """
    _, _, cp = _ensure_deps()

    buffer = _get_attr_or_call(chars_col, "data_buffer") or _get_attr_or_call(
        chars_col, "data"
    )
    if buffer is None or buffer.nbytes == 0:
        # A column of empty strings has no character bytes at all.
        chars = cp.empty(0, dtype="uint8")
    else:
        chars = _buf_to_cupy(buffer, "uint8")

    # Reading the last offset costs one small device-to-host copy, but it is
    # the only place the required number of character bytes is recorded.
    stop = _get_offset(col) + _get_size(col)
    num_chars = int(offsets[stop]) if int(offsets.size) > stop else 0

    # Invariant: the character buffer must cover every byte the offsets
    # address, otherwise reading the strings would run past the allocation.
    if int(chars.size) < num_chars:
        raise RuntimeError(
            f"Buffer size mismatch: the string offsets address {num_chars} "
            f"character bytes but the column's character buffer holds "
            f"{int(chars.size)}.  This may indicate a sliced column where "
            f"offset metadata was not propagated."
        )

    return chars


def _finalize(layout: Content, col: plc.Column) -> Content:
    """
    Apply the logical column offset and nullable mask to a layout.

    libcudf uses Arrow-style packed validity bits, where 1 means valid.
    Awkward's BitMaskedArray can wrap these packed bits directly.
    """
    import cupy as cp

    offset = _get_offset(col)
    size = _get_size(col)
    stop = offset + size

    # Invariant: for a well-formed pylibcudf column the layout must cover the
    # logical slice implied by offset and size.  If it does not, slicing below
    # would overrun the available buffer and silently corrupt data.
    if layout.length < stop:
        raise RuntimeError(
            "pylibcudf column buffers have an unexpected shape for ak.from_cudf"
        )

    if _get_null_count(col) != 0:
        mask = _get_attr_or_call(col, "null_mask")
        if mask is not None:
            if not isinstance(mask, cp.ndarray):
                mask = _buf_to_cupy(mask, "uint8")
            layout = BitMaskedArray.simplified(
                IndexU8(mask),
                layout,
                valid_when=True,
                length=stop,
                lsb_order=True,
            )

    if offset != 0 or layout.length != size:
        layout = layout[offset:stop]

    return layout


def _column_to_layout(col: plc.Column, dtype: Any = None) -> Content:
    """
    Convert a pylibcudf Column into an Awkward layout.

    ``dtype`` is the Python-level cudf dtype for this column (e.g.
    ``cudf.StructDtype``, ``cudf.ListDtype``, or a numpy dtype).  When
    provided it is used to recover struct field names and list element dtypes,
    because ``pylibcudf.Column`` objects carry no field-name metadata — names
    live exclusively in the Python-level dtype objects.
    """
    import numpy as np

    _, plc_module, _ = _ensure_deps()
    type_id = col.type().id()
    primitive_dtypes = _primitive_dtypes(plc_module)

    if type_id in primitive_dtypes:
        np_dtype_str = primitive_dtypes[type_id]
        np_dtype = np.dtype(np_dtype_str)
        if np_dtype.kind in ("M", "m"):
            # CuPy does not support *creating* datetime64/timedelta64 arrays,
            # but for some reason, views are fine
            int64_cp = _data_to_cupy(col, "int64")
            layout = NumpyArray(int64_cp.view(np_dtype))
        else:
            layout = NumpyArray(_data_to_cupy(col, np_dtype_str))

    elif type_id == _type_id(plc_module, "LIST"):
        offsets_col, content_col = _list_offsets_and_content(col)
        # Propagate the element dtype (if known) so nested struct names survive.
        child_dtype = getattr(dtype, "element_type", None)
        layout = ListOffsetArray(
            _offsets_to_index(offsets_col, col),
            _column_to_layout(content_col, child_dtype),
        )

    elif type_id == _type_id(plc_module, "STRUCT"):
        # pylibcudf Column objects carry no field names; recover them from the
        # Python-level StructDtype when available, falling back to positional
        # string indices ("0", "1", ...) only when no dtype was supplied.
        if dtype is not None and hasattr(dtype, "fields"):
            field_names = list(dtype.fields.keys())
            field_dtypes = list(dtype.fields.values())
        else:
            field_names = _struct_field_names(col)
            field_dtypes = [None] * len(field_names)
        layout = RecordArray(
            [
                _column_to_layout(col.child(i), field_dtypes[i])
                for i in range(_get_num_children(col))
            ],
            field_names,
            length=_get_offset(col) + _get_size(col),
        )

    elif type_id == _type_id(plc_module, "STRING"):
        offsets_col, chars_col = _string_offsets_and_chars(col)
        offsets = _offsets_to_index(offsets_col, col)
        layout = ListOffsetArray(
            offsets,
            NumpyArray(
                _string_chars_to_cupy(col, chars_col, offsets.data),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        )

    elif type_id == _type_id(plc_module, "DICTIONARY32"):
        layout = IndexedArray(
            Index32(_data_to_cupy(col.child(0), "int32")),
            _column_to_layout(col.child(1)),
            parameters={"__array__": "categorical"},
        )

    else:
        raise NotImplementedError(
            f"pylibcudf type id {type_id!r} is not supported by ak.from_cudf yet. "
            f"Please open an issue at {_ISSUE_URL} if you need this type."
        )

    return _finalize(layout, col)


def _categorical_to_layout(series: Any) -> Content:
    """
    Convert a categorical cuDF Series into a categorical Awkward layout.

    cuDF does not use libcudf's DICTIONARY32 type for categoricals: the
    pylibcudf column of a categorical Series holds only the integer codes,
    while the categories live on the Python-level ``CategoricalDtype``.  The
    two halves are therefore converted separately and recombined here into
    the same ``"categorical"``-parameterised layout that #ak.from_arrow
    produces for Arrow dictionary types.
    """
    cudf, _, _ = _ensure_deps()

    content = _series_to_layout(cudf.Series(series.dtype.categories))

    # Awkward's Index cannot hold the narrow (e.g. int8) code dtypes cuDF
    # picks, so the codes are widened to int64 on the device.
    codes = series.cat.codes.astype("int64")

    if series.null_count != 0:
        # A null entry belongs to no category; -1 marks it missing to Awkward.
        return IndexedOptionArray.simplified(
            Index64(codes.fillna(-1).to_cupy()),
            content,
            parameters={"__array__": "categorical"},
        )

    return IndexedArray(
        Index64(codes.to_cupy()),
        content,
        parameters={"__array__": "categorical"},
    )


def _series_to_layout(series: Any) -> Content:
    cudf, _, _ = _ensure_deps()

    if isinstance(series.dtype, cudf.CategoricalDtype):
        return _categorical_to_layout(series)

    # Pass the Python-level cudf dtype so that _column_to_layout can recover
    # struct field names at every nesting level.  pylibcudf Column objects
    # carry no field-name metadata; all name information lives in the dtype.
    return _column_to_layout(_to_pylibcudf_column(series), series.dtype)


def _dataframe_to_layout(dataframe: Any) -> Content:
    """
    Convert a cuDF DataFrame into a top-level RecordArray.

    Note: Column names and order are preserved. Arrow/cuDF column metadata
    beyond field names (e.g. time-zone annotations, extension-type metadata)
    is not yet propagated and will be addressed in a follow-up.
    """
    fields = list(dataframe.columns)
    contents = [_series_to_layout(dataframe[name]) for name in fields]
    return RecordArray(contents, fields, length=len(dataframe))
