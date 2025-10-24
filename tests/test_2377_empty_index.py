from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pytest

import awkward as ak
from awkward.types.numpytype import _primitive_to_dtype_dict, primitive_to_dtype


def test_issue() -> None:
    """Assert the GitHub issue #2377 is resolved."""
    empty = ak.Array([])  # <Array [] type='0 * unknown'>

    # No exception should be raised.
    result = ak.flatten(empty[empty])

    # The result should be an empty awkward array of the same type.
    expected_layout = ak.Array([]).layout
    assert result.layout.is_equal_to(expected_layout, all_parameters=True)


def _add_necessary_unit(dtype_name: str) -> str:
    """Completes datetime or timedelta dtype names with a unit if missing."""
    UNIT_LESS_DT_RE = re.compile(r"^(?:datetime|timedelta)\d*$")
    SAMPLE_UNIT = "15us"
    if UNIT_LESS_DT_RE.fullmatch(dtype_name):
        return f"{dtype_name}[{SAMPLE_UNIT}]"
    return dtype_name


# (dtype('bool'), dtype('int8'), dtype('<M8[15us]'), ...) Supported dtypes
DTYPES = tuple(
    primitive_to_dtype(_add_necessary_unit(k)) for k in _primitive_to_dtype_dict.keys()
)


# (dtype('bool'), dtype('int8'), ...) Only bool and integer types
INDEXABLE_DTYPES = tuple(d for d in DTYPES if d.kind in ("b", "i", "u"))


AWKWARD_ARRAYS = (
    # Empty array of unknown type
    ak.Array([]),
    # Empty arrays of various dtypes
    *[ak.from_numpy(np.array([], dtype=d)) for d in DTYPES],
    # Non-empty arrays of specific types
    ak.Array([1, 2, 3]),
    ak.Array([[1, 2], [], [3]]),
    ak.Array([[[1.1, 2.2], []], [[3.3]], []]),
    ak.Array([1 + 1j, 2 + 2j, 3 + 3j]),
    ak.Array([datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)]),
    # Non-empty arrays of mixed types
    ak.Array([[1, 2], [1.1, 2.2], [1 + 1j, 2 + 2j]]),
    ak.Array(
        [
            [1, 2],
            [1.1, 2.2],
            [1 + 1j, 2 + 2j],
            [datetime(2020, 1, 1), datetime(2021, 1, 1)],
        ]
    ),
)

EMPTY_INDEXES = (
    # (),  # NOTE: An empty tuple doesn't pass the test.
    [],
    ak.Array([]),
    *[ak.from_numpy(np.array([], dtype=d)) for d in INDEXABLE_DTYPES],
    *[np.array([], dtype=d) for d in INDEXABLE_DTYPES],
)


@pytest.mark.parametrize("a", AWKWARD_ARRAYS)
@pytest.mark.parametrize("idx", EMPTY_INDEXES)
def test_empty_index(a: ak.Array, idx: Iterable) -> None:
    """Assert indexing with an empty array preserves the type."""
    result = a[idx]

    # Assert an empty array.
    assert result.to_list() == []

    # Assert the type is preserved
    # e.g., "0 * complex128", "0 * var * union[complex128, datetime64[us]]"
    expected_typestr = str(ak.types.ArrayType(a.type.content, 0))
    assert result.typestr == expected_typestr
