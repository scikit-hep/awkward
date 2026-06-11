# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from awkward.contents import (
    BitMaskedArray,
    IndexedArray,
    ListOffsetArray,
    NumpyArray,
    RecordArray,
)
from awkward.index import Index32, Index64, IndexU8

if TYPE_CHECKING:
    import cupy as cp
    import pylibcudf as plc

    from awkward.contents.content import Content


__all__ = ("from_cudf",)


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
    return cp.ndarray(shape=(length,), dtype=dtype, memptr=memptr)  # type: ignore[call-arg]


def _data_to_cupy(col: plc.Column, dtype: str) -> cp.ndarray:
    _, _, cp = _ensure_deps()
    buffer = _get_attr_or_call(col, "data_buffer")
    offset = _get_offset(col)
    size = _get_size(col)
    length = offset + size

    if buffer is None:
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
        buffer = _get_attr_or_call(offsets_col, "data_buffer")
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
    # pylibcudf >= 25.04 exposes child_names() directly.
    num_children = _get_num_children(col)
    if hasattr(col, "child_names"):
        names = _get_attr_or_call(col, "child_names")
        if names is not None:
            names = list(names)
            if len(names) < num_children:
                names.extend(str(i) for i in range(len(names), num_children))
            return names[:num_children]

    # Older versions: read from the Arrow-like schema when exposed.
    schema = col.type()
    names = []
    for i in range(num_children):
        child = getattr(schema, "child", None)
        if callable(child):
            field = child(i)
            name = _get_attr_or_call(field, "name")
        else:
            name = None
        names.append(str(i) if name is None else name)
    return names


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
    if _get_num_children(col) >= 2:
        return col.child(0), col.child(1)
    else:
        return col.child(0), col


def _finalize(layout: Content, col: plc.Column) -> Content:
    """
    Apply the logical column offset and nullable mask to a layout.

    libcudf uses Arrow-style packed validity bits, where 1 means valid.
    Awkward's BitMaskedArray can wrap these packed bits directly.
    """
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
            layout = BitMaskedArray.simplified(
                IndexU8(_buf_to_cupy(mask, "uint8")),
                layout,
                valid_when=True,
                length=stop,
                lsb_order=True,
            )

    if offset != 0 or layout.length != size:
        layout = layout[offset:stop]

    return layout


def _column_to_layout(col: plc.Column) -> Content:
    _, plc_module, _ = _ensure_deps()
    type_id = col.type().id()
    primitive_dtypes = _primitive_dtypes(plc_module)

    if type_id in primitive_dtypes:
        layout = NumpyArray(_data_to_cupy(col, primitive_dtypes[type_id]))

    elif type_id == _type_id(plc_module, "LIST"):
        offsets_col, content_col = _list_offsets_and_content(col)
        layout = ListOffsetArray(
            _offsets_to_index(offsets_col, col),
            _column_to_layout(content_col),
        )

    elif type_id == _type_id(plc_module, "STRUCT"):
        layout = RecordArray(
            [_column_to_layout(col.child(i)) for i in range(_get_num_children(col))],
            _struct_field_names(col),
            length=_get_offset(col) + _get_size(col),
        )

    elif type_id == _type_id(plc_module, "STRING"):
        offsets_col, chars_col = _string_offsets_and_chars(col)
        layout = ListOffsetArray(
            _offsets_to_index(offsets_col, col),
            NumpyArray(
                _data_to_cupy(chars_col, "uint8"),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        )

    elif type_id == _type_id(plc_module, "DICTIONARY32"):
        layout = IndexedArray(
            Index32(_data_to_cupy(col.child(0), "int32")),
            _column_to_layout(col.child(1)),
        )

    else:
        raise NotImplementedError(
            f"pylibcudf type id {type_id!r} is not supported by ak.from_cudf yet. "
            f"Please open an issue at {_ISSUE_URL} if you need this type."
        )

    return _finalize(layout, col)


def _series_to_layout(series: Any) -> Content:
    return _column_to_layout(_to_pylibcudf_column(series))


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


def from_cudf(obj: Any) -> Content:
    """
    Args:
        obj (cudf.Series or cudf.DataFrame): The cuDF object to convert into a
            low-level Awkward layout.

    Converts a cuDF Series or DataFrame into a low-level Awkward layout by
    recursively traversing pylibcudf columns and wrapping GPU buffers with
    CuPy.

    Primitive, boolean, list, struct, string, dictionary, and nullable columns
    are supported. Other column types raise ``NotImplementedError``.

    Note: DataFrame column names and order are preserved. Arrow/cuDF column
    metadata beyond field names (e.g. time-zone annotations,
    extension-type metadata) is not yet propagated and will be addressed in a
    follow-up.
    """
    cudf, _, _ = _ensure_deps()

    if isinstance(obj, cudf.Series):
        return _series_to_layout(obj)
    elif isinstance(obj, cudf.DataFrame):
        return _dataframe_to_layout(obj)
    else:
        raise TypeError(
            "ak.from_cudf accepts only cudf.Series or cudf.DataFrame, "
            f"not {type(obj).__name__!r}"
        )
