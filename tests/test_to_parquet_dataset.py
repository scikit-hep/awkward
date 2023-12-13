from __future__ import annotations

import os

import awkward as ak


def simple_test(path1, path2, path3):
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    array1 = ak.Array([[1.1, 2.2, 3.3, 4.4], [4.0], [4.4, 5.5]])
    array2 = ak.Array([[1.0, 3.0, 3.3, 4.4], [4.0], [4.4, 10.0], [11.11]])

    ak.to_parquet(array, path1, parquet_compliant_nested=True)
    ak.to_parquet(array1, path2, parquet_compliant_nested=True)
    ak.to_parquet(array2, path3, parquet_compliant_nested=True)
    try:
        os.remove(
            "/Users/zobil/Documents/awkward/tests/samples/to_parquet/_common_metadata"
        )
        os.remove("/Users/zobil/Documents/awkward/tests/samples/to_parquet/_metadata")
    except:
        print("not there")

    no_metadata = ak.from_parquet(
        "/Users/zobil/Documents/awkward/tests/samples/to_parquet"
    )
    assert no_metadata.tolist() == [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]

    ak.to_parquet_dataset("/Users/zobil/Documents/awkward/tests/samples/to_parquet")

    assert os.path.exists(
        "/Users/zobil/Documents/awkward/tests/samples/to_parquet/_common_metadata"
    )
    assert os.path.exists(
        "/Users/zobil/Documents/awkward/tests/samples/to_parquet/_metadata"
    )

    with_metadata = ak.from_parquet(
        "/Users/zobil/Documents/awkward/tests/samples/to_parquet"
    )
    assert with_metadata.tolist() == [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]

    os.remove(path1)
    os.remove(path2)
    os.remove(path3)
    os.remove(
        "/Users/zobil/Documents/awkward/tests/samples/to_parquet/_common_metadata"
    )
    os.remove("/Users/zobil/Documents/awkward/tests/samples/to_parquet/_metadata")
