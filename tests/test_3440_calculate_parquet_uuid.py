# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pathlib

import pytest

import awkward

pytest.importorskip("pyarrow.parquet")

metadata_from_parquet_submodule = pytest.importorskip(
    "awkward.operations.ak_metadata_from_parquet"
)
metadata_from_parquet = metadata_from_parquet_submodule.metadata_from_parquet

SAMPLES_DIR = pathlib.Path(__file__).parent / "samples"
input = SAMPLES_DIR / "nullable-record-primitives.parquet"


def test_parquet_uuid():
    meta = metadata_from_parquet(input)
    assert (
        meta["uuid"]
        == "582dabdb8c87bfa17bc930676ed26b8d4ab22a900f92357751dc380c41acb593"
    )


@pytest.mark.parametrize("calculate_uuid", [True, False])
def test_return_tuple_with_or_without_uuid(calculate_uuid):
    results = awkward.operations.ak_from_parquet.metadata(
        input,
        {},
        None,
        None,
        False,
        True,
        calculate_uuid=calculate_uuid,
    )
    if calculate_uuid:
        assert len(results) == 8, "Expected 8 items in the result tuple"
        (
            parquet_columns,
            subform,
            actual_paths,
            fs,
            subrg,
            col_counts,
            metadata,
            uuid,
        ) = results
        assert uuid is not None, "UUID should be present when calculate_uuid is True"
        print("uuid:", uuid)
    else:
        assert len(results) == 7, "Expected 7 items in the result tuple"
        parquet_columns, subform, actual_paths, fs, subrg, col_counts, metadata = (
            results
        )
