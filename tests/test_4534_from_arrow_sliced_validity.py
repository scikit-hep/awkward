# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

pa = pytest.importorskip("pyarrow")

DATA = {
    "int64": [None if i % 3 == 0 else i for i in range(24)],
    "string": [None if i % 3 == 0 else "x" * (i % 4) for i in range(24)],
    "list": [None if i % 3 == 0 else [i] * (i % 3) for i in range(24)],
    "struct": [None if i % 3 == 0 else {"x": i} for i in range(24)],
}


@pytest.mark.parametrize("name", DATA)
@pytest.mark.parametrize("offset", [1, 3, 8, 9])
def test_sliced_array(name, offset):
    sliced = pa.array(DATA[name]).slice(offset, 7)
    assert ak.from_arrow(sliced).to_list() == DATA[name][offset : offset + 7]
