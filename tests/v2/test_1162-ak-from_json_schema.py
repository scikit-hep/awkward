# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_integer():
    result = ak._v2.operations.convert.from_json_schema(
        "[1, 2, 3, 4, 5]",
        {"type": "array", "items": {"type": "integer"}},
    )
    assert result.tolist() == [1, 2, 3, 4, 5]
