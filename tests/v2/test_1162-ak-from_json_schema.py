# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_boolean():
    result = ak._v2.operations.convert.from_json_schema(
        "[true, false, true, true, false]",
        {"type": "array", "items": {"type": "boolean"}},
    )
    assert result.tolist() == [True, False, True, True, False]


def test_integer():
    result = ak._v2.operations.convert.from_json_schema(
        "[1, 2, 3.0, 4, 5]",
        {"type": "array", "items": {"type": "integer"}},
    )
    assert result.tolist() == [1, 2, 3, 4, 5]


def test_number():
    result = ak._v2.operations.convert.from_json_schema(
        "[1, 2, 3.14, 4, 5]",
        {"type": "array", "items": {"type": "number"}},
    )
    assert result.tolist() == [1, 2, 3.14, 4, 5]


def test_option_boolean():
    result = ak._v2.operations.convert.from_json_schema(
        "[true, false, null, true, false]",
        {"type": "array", "items": {"type": ["boolean", "null"]}},
    )
    assert result.tolist() == [True, False, None, True, False]


def test_option_integer():
    result = ak._v2.operations.convert.from_json_schema(
        "[1, 2, null, 4, 5]",
        {"type": "array", "items": {"type": ["null", "integer"]}},
    )
    assert result.tolist() == [1, 2, None, 4, 5]
