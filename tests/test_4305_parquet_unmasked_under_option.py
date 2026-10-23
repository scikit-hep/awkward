# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

pa = pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")

VALID = np.arange(8) % 2 == 0


def unmasked():
    return ak.contents.UnmaskedArray(ak.contents.NumpyArray(np.arange(len(VALID))))


def record(content):
    return ak.contents.RecordArray([content], ["x"])


def tuple_(content):
    return ak.contents.RecordArray([content], None)


def bytemasked(content):
    return ak.contents.ByteMaskedArray(
        ak.index.Index8(VALID.astype(np.int8)), content, valid_when=True
    )


def bitmasked(content):
    return ak.contents.BitMaskedArray(
        ak.index.IndexU8(np.packbits(VALID, bitorder="little")),
        content,
        valid_when=True,
        length=len(VALID),
        lsb_order=True,
    )


def indexedoption(content):
    return ak.contents.IndexedOptionArray(
        ak.index.Index64(np.where(VALID, np.arange(len(VALID)), -1)), content
    )


CASES = {
    "tuple(option(record(unmasked)))": lambda option: tuple_(
        option(record(unmasked()))
    ),
    "record(option(record(unmasked)))": lambda option: record(
        option(record(unmasked()))
    ),
    "option(record(record(unmasked)))": lambda option: option(
        record(record(unmasked()))
    ),
    "option(tuple(unmasked))": lambda option: option(tuple_(unmasked())),
    "option(tuple(record(unmasked)))": lambda option: option(
        tuple_(record(unmasked()))
    ),
    "list(option(record(unmasked)))": lambda option: ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 8])), option(record(unmasked()))
    ),
    "record(option(record(indexed(record(unmasked)))))": lambda option: record(
        option(
            record(
                ak.contents.IndexedArray(
                    ak.index.Index64(np.arange(7, -1, -1)), record(unmasked())
                )
            )
        )
    ),
    "record(option(record(unmasked(record))))": lambda option: record(
        option(
            record(
                ak.contents.UnmaskedArray(
                    record(ak.contents.NumpyArray(np.arange(len(VALID))))
                )
            )
        )
    ),
}


def test_from_arrow_struct_with_nullable_field(tmp_path):
    table = pa.table(
        {
            "s": pa.array(
                [{"x": i} if i % 2 == 0 else None for i in range(8)],
                type=pa.struct([pa.field("x", pa.int64(), nullable=True)]),
            )
        }
    )
    array = ak.from_arrow(table)
    assert isinstance(
        array.layout.content("s").content.content("x"), ak.contents.UnmaskedArray
    )

    path = tmp_path / "test.parquet"
    ak.to_parquet(array, path)
    result = ak.from_parquet(path)

    assert result.to_list() == array.to_list()
    assert result.type == array.type
    assert isinstance(
        result.layout.content("s").content.content("x"), ak.contents.UnmaskedArray
    )


@pytest.mark.parametrize("option", [bytemasked, bitmasked, indexedoption])
@pytest.mark.parametrize("case", CASES)
def test_unmasked_below_option_with_nulls(tmp_path, case, option):
    array = ak.Array(CASES[case](option))

    path = tmp_path / "test.parquet"
    ak.to_parquet(array, path)
    result = ak.from_parquet(path)

    assert result.to_list() == array.to_list()
    assert result.type == array.type
    # parquet fills the unmasked validity with the enclosing nulls
    # but it should still read back like the arrow table does
    assert result.layout.form == ak.from_arrow(ak.to_arrow_table(array)).layout.form
