from __future__ import annotations

import pytest

import awkward as ak

pa = pytest.importorskip("pyarrow")


def test_tz_is_dropped():
    data = pa.Table.from_arrays(
        [
            pa.array(
                [
                    1,
                    2,
                    3,
                ],
                type=pa.timestamp("ns", tz="UTC"),
            )
        ],
        names=["a"],
    )
    ak.from_arrow(data)
    ak.from_arrow_schema(data.schema)
