# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json

import hypothesis_awkward.strategies as st_ak
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis_awkward.util.dtype import SUPPORTED_DTYPES

import awkward as ak


def json_dtype(dtype: np.dtype) -> bool:
    if dtype.kind == "c":
        # json.dumps cannot serialize complex numbers (TypeError)
        return False
    if dtype.kind == "f" and dtype.itemsize == 16:
        # json.dumps cannot serialize the extended-precision numpy scalars
        # that `tolist` leaves behind on platforms that have them (e.g.
        # float128 on Linux)
        return False
    if dtype.kind in "Mm":
        # json.dumps cannot serialize the datetime/timedelta objects that
        # datetime64/timedelta64 values become (TypeError)
        return False
    return True


SUPPORTED_JSON_DTYPES = tuple(d for d in SUPPORTED_DTYPES if json_dtype(d))


def json_dtypes() -> st.SearchStrategy[np.dtype]:
    """Strategy for dtypes that `to_json` can serialize."""
    return st.sampled_from(SUPPORTED_JSON_DTYPES)


def inferred_dtype(dtype: np.dtype) -> bool:
    # Schemaless `from_json` types every JSON scalar as bool, int64, or
    # float64: any other width or signedness reads back as one of these
    return dtype in (np.dtype(np.bool_), np.dtype(np.int64), np.dtype(np.float64))


SUPPORTED_INFERRED_DTYPES = tuple(d for d in SUPPORTED_JSON_DTYPES if inferred_dtype(d))


def inferred_json_dtypes() -> st.SearchStrategy[np.dtype]:
    """Strategy for dtypes that survive the JSON roundtrip."""
    return st.sampled_from(SUPPORTED_INFERRED_DTYPES)


def json_writable(a: ak.Array) -> bool:
    """Arrays whose values `to_json` can serialize.

    JSON has no representation for infinities, and json.dumps refuses
    them (ValueError); `allow_nan=False` at the strategies keeps NaN,
    which JSON cannot represent either, out of the domains. This is a
    limit of the format and stays.
    """
    return not _has_infinity(a.layout)


def _has_infinity(layout: ak.contents.Content) -> bool:
    """An infinity is among the values `to_json` serializes.

    Only values the writer visits count: a missing value is written as
    null, so an infinity hidden behind an option mask is fine.
    """
    if layout.is_numpy:
        return layout.dtype.kind == "f" and bool(np.isinf(layout.data).any())
    if layout.is_union:
        contents = range(len(layout.contents))
        return any(_has_infinity(layout.project(i)) for i in contents)
    if layout.is_record:
        return any(_has_infinity(c[: layout.length]) for c in layout.contents)
    if layout.is_option or layout.is_indexed:
        return _has_infinity(layout.project())
    if layout.is_regular:
        return _has_infinity(layout.content[: layout.length * layout.size])
    if layout.is_list:
        layout = layout.to_ListOffsetArray64(False)
        return _has_infinity(layout.content[layout.offsets[0] : layout.offsets[-1]])
    return False


def json_compatible(a: ak.Array) -> bool:
    """Arrays whose type the JSON roundtrip preserves.

    The clauses in `_loses_type` are limits of the conversion itself and
    stay: schemaless `from_json` rebuilds the type purely from the values
    present in the JSON text.
    """
    return json_writable(a) and not _loses_type(a.layout)


def _loses_type(layout: ak.contents.Content) -> bool:
    """The roundtrip changes the array's type without corrupting values.

    Every part of the type needs at least one witness value in the JSON
    text: `from_json` reads `[]` as unknown, null as an option, and an
    object as a named record.
    """
    if layout.is_unknown:
        # unknown is the one type an absence of values witnesses
        return False
    if layout.length == 0:
        # no values witness the type: reads back as unknown
        return True
    if layout.is_option:
        projected = layout.project()
        if projected.length == layout.length:
            # no missing values: reads back without the option
            return True
        # the nulls witness the option; the valid values, its content
        return _loses_type(projected)
    if layout.is_indexed:
        return _loses_type(layout.project())
    if layout.is_record:
        if layout.is_tuple:
            # a tuple is written as an object with stringified indices
            # ({"0": ...}): reads back as a named record
            return True
        return any(_loses_type(c[: layout.length]) for c in layout.contents)
    if layout.parameter("__array__") == "string":
        return False
    if layout.is_list:
        # only the elements inside the lists reach the JSON text
        layout = layout.to_ListOffsetArray64(False)
        return _loses_type(layout.content[layout.offsets[0] : layout.offsets[-1]])
    return False


def has_issues(a: ak.Array) -> bool:
    """Arrays affected by a known issue in the conversion.

    One function per known issue, each named after the issue whose
    released fix deletes it; with all of them gone the test filters
    reduce to `json_compatible` and `json_writable`.
    """
    if _has_issue_4241(a.layout):
        return True
    if _has_issue_4242(a.layout):
        return True
    return False


def _has_issue_4241(layout: ak.contents.Content) -> bool:
    """`from_json` parses numbers with rapidjson's approximate default
    algorithm, not `kParseFullPrecisionFlag`: a shortest-repr double can
    read back off by one ULP, and re-parsing its new repr can drift
    further, so affected values neither roundtrip nor reach a fixed
    point (silent corruption, #4241).

    Only values the writer visits count, as in `_has_infinity`. Whether
    a value is affected depends on its digit string, so each float leaf
    is checked directly against a `from_json` readback.
    """
    if layout.is_numpy:
        if layout.dtype.kind != "f":
            return False
        # `to_json` serializes the repr of the exact widened double
        data = np.asarray(layout.data, dtype=np.float64)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return False
        returned = np.asarray(ak.from_json(json.dumps(data.tolist())))
        return not np.array_equal(returned, data)
    if layout.is_union:
        contents = range(len(layout.contents))
        return any(_has_issue_4241(layout.project(i)) for i in contents)
    if layout.is_record:
        return any(_has_issue_4241(c[: layout.length]) for c in layout.contents)
    if layout.is_option or layout.is_indexed:
        return _has_issue_4241(layout.project())
    if layout.is_regular:
        return _has_issue_4241(layout.content[: layout.length * layout.size])
    if layout.is_list:
        layout = layout.to_ListOffsetArray64(False)
        return _has_issue_4241(layout.content[layout.offsets[0] : layout.offsets[-1]])
    return False


def _has_issue_4242(layout: ak.contents.Content) -> bool:
    """`from_json` crashes on a null alongside heterogeneous values: the
    builder emits an option-of-union form that `from_buffers` rebuilds
    with the `IndexedOptionArray` constructor, which refuses a union
    content, instead of with `IndexedOptionArray.simplified` (TypeError,
    #4242).

    A union whose elements mix a visible null with two or more builder
    kinds (bool, number, string, list, record) is affected; with one
    kind the builder merges instead (numbers of any dtype are one kind,
    and records merge across field names).
    """
    if layout.is_union:
        kinds = set()
        has_null = False
        for i in range(len(layout.contents)):
            child = layout.project(i)
            while child.is_option or child.is_indexed:
                if child.is_option:
                    has_null = has_null or child.project().length < child.length
                child = child.project()
            if child.length == 0 or child.is_unknown:
                continue
            if _has_issue_4242(child):
                return True
            kinds.add(_builder_kind(child))
        return has_null and len(kinds) >= 2
    if layout.is_record:
        return any(_has_issue_4242(c[: layout.length]) for c in layout.contents)
    if layout.is_option or layout.is_indexed:
        return _has_issue_4242(layout.project())
    if layout.is_regular:
        return _has_issue_4242(layout.content[: layout.length * layout.size])
    if layout.parameter("__array__") == "string":
        return False
    if layout.is_list:
        layout = layout.to_ListOffsetArray64(False)
        return _has_issue_4242(layout.content[layout.offsets[0] : layout.offsets[-1]])
    return False


def _builder_kind(layout: ak.contents.Content) -> str:
    """The builder slot a value of this (non-option) layout lands in."""
    if layout.parameter("__array__") == "string":
        return "string"
    if layout.is_record:
        return "record"
    if layout.is_list or layout.is_regular:
        return "list"
    if layout.dtype == np.dtype(np.bool_):
        return "bool"
    return "number"


def _children(layout: ak.contents.Content) -> list[ak.contents.Content]:
    """The direct child layouts of a node."""
    if layout.is_record or layout.is_union:
        return list(layout.contents)
    if layout.is_option or layout.is_indexed or layout.is_list:
        return [layout.content]
    return []


@given(
    a=st_ak.constructors.arrays(
        dtypes=inferred_json_dtypes(),
        # json.dumps raises ValueError on NaN
        allow_nan=False,
        # json.dumps cannot serialize bytes (TypeError)
        allow_bytestring=False,
        # a fixed-size list is written as a plain JSON array: reads back
        # as a var-length list
        allow_regular=False,
        # `from_json` funnels JSON values of one shape into one builder
        # slot: union[int64, float64] reads back as float64, and a union
        # of records as a single record of option-type fields
        allow_union=False,
        allow_indexed_option=False,
        allow_byte_masked=False,
        allow_bit_masked=False,
        allow_unmasked=False,
    ).filter(lambda a: json_compatible(a) and not has_issues(a))
)
def test_roundtrip(a: ak.Array) -> None:
    """`to_json` followed by `from_json` reconstructs the array."""
    returned = ak.from_json(ak.to_json(a))
    assert ak.array_equal(a, returned)


@given(
    a=st_ak.constructors.arrays(
        dtypes=inferred_json_dtypes(),
        allow_nan=False,
        allow_bytestring=False,
        allow_regular=False,
        allow_union=False,
        # an UnmaskedArray never has a missing value, so a null can never
        # witness its option: it always reads back non-option
        allow_unmasked=False,
    ).filter(lambda a: json_compatible(a) and not has_issues(a))
)
def test_roundtrip_masked(a: ak.Array) -> None:
    """Option-type arrays roundtrip through JSON nulls.

    `from_json` returns every option layout as an `IndexedOptionArray`,
    so content classes are not compared.
    """
    returned = ak.from_json(ak.to_json(a))
    assert ak.array_equal(a, returned, same_content_types=False)


def tolist_comparable(a: ak.Array) -> bool:
    """Arrays whose `to_list` output equals the parsed JSON text.

    Like `json_writable`, which this extends, the clause here is a limit
    of the representation: `to_list` yields a Python tuple for a tuple
    record, but the JSON object it is written as parses as a dict.
    """
    return json_writable(a) and not _has_tuple(a.layout)


def _has_tuple(layout: ak.contents.Content) -> bool:
    """A tuple record is somewhere in the layout."""
    if layout.is_record and layout.is_tuple:
        return True
    return any(_has_tuple(x) for x in _children(layout))


@given(
    a=st_ak.constructors.arrays(
        dtypes=json_dtypes(),
        allow_nan=False,
        allow_bytestring=False,
    ).filter(tolist_comparable)
)
def test_to_json_matches_json_loads(a: ak.Array) -> None:
    """`json.loads` reads the JSON text back as exactly `to_list`'s output.

    An oracle check of the writer against Python's own JSON parser,
    independent of `from_json`'s type inference: any numeric dtype,
    regular lists, records, options, and unions are all in the domain.
    """
    assert json.loads(ak.to_json(a)) == ak.to_list(a)


@given(
    a=st_ak.constructors.arrays(
        dtypes=json_dtypes(),
        allow_nan=False,
        allow_bytestring=False,
    ).filter(lambda a: json_writable(a) and not has_issues(a))
)
def test_roundtrip_stable(a: ak.Array) -> None:
    """One roundtrip brings the conversion to a fixed point.

    The first `from_json` can change types (int32 reads back as int64, a
    fixed-size list as a var-length list, a tuple as a named record, an
    unwitnessed type as unknown) and even values (uint64 above int64 max
    wraps negative, #4243; integers beyond uint64 read as float64), so neither
    the array nor the first JSON text is reproducible in general. After
    one full roundtrip the builder's normalizations have happened:
    converting the reconstruction again yields the same text. Layouts
    affected by `has_issues` are still excluded: this test does not
    vouch for them.
    """
    r = ak.from_json(ak.to_json(a))
    m = ak.to_json(r)
    m2 = ak.to_json(ak.from_json(m))
    assert m2 == m
