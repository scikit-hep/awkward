# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward.types import (
    ArrayType,
    ListType,
    NumpyType,
    OptionType,
    RecordType,
    RegularType,
    ScalarType,
    UnionType,
)
from awkward.types.numpytype import _primitive_to_dtype_dict

PRIMITIVES = list(_primitive_to_dtype_dict)

CONTEXTS = [
    ("{}", lambda t: t),
    ("var * {}", ListType),
    ("3 * {}", lambda t: RegularType(t, 3)),
    ("?{}", OptionType),
    ("option[var * {}]", lambda t: OptionType(ListType(t))),
    ("{{x: {}}}", lambda t: RecordType([t], ["x"])),
    ("({}, int64)", lambda t: RecordType([t, NumpyType("int64")], None)),
    (
        "union[{}, var * int64]",
        lambda t: UnionType([t, ListType(NumpyType("int64"))]),
    ),
    (
        "Name[x: {}]",
        lambda t: RecordType([t], ["x"], parameters={"__record__": "Name"}),
    ),
]


@pytest.mark.parametrize("primitive", PRIMITIVES)
@pytest.mark.parametrize(("template", "build"), CONTEXTS)
def test_type_string_roundtrip(primitive, template, build):
    datashape = template.format(primitive)
    expected = build(NumpyType(primitive))
    assert str(expected) == datashape
    assert ak.types.from_datashape(datashape, highlevel=False) == expected


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_parameters_roundtrip(primitive):
    expected = NumpyType(primitive, parameters={"a": 1})
    assert ak.types.from_datashape(str(expected), highlevel=False) == expected


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_highlevel(primitive):
    content = NumpyType(primitive)
    assert ak.types.from_datashape(f"5 * var * {primitive}") == ArrayType(
        ListType(content), 5
    )
    assert ak.types.from_datashape(f"{{x: {primitive}}}") == ScalarType(
        RecordType([content], ["x"])
    )


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_from_numpy_type_reparses(primitive):
    array = ak.from_numpy(np.zeros((2, 3), dtype=_primitive_to_dtype_dict[primitive]))
    assert ak.types.from_datashape(str(array.type), highlevel=True) == array.type


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_enforce_type_string(primitive):
    array = ak.Array([[0, 1], []])
    out = ak.enforce_type(array, f"var * {primitive}")
    assert out.type == ArrayType(ListType(NumpyType(primitive)), 2)


@pytest.mark.parametrize("primitive", ["float128", "complex256"])
def test_platform_dependent_primitive_is_lexed(primitive):
    if primitive in _primitive_to_dtype_dict:
        assert ak.types.from_datashape(
            f"var * {primitive}", highlevel=False
        ) == ListType(NumpyType(primitive))
    else:
        with pytest.raises(TypeError, match="unrecognized primitive"):
            ak.types.from_datashape(f"var * {primitive}", highlevel=False)
