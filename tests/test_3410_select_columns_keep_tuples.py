# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test_select_columns():
    (records, records_tuple) = (
        ak.forms.RecordForm(
            [
                ak.forms.NumpyForm("int64"),
                ak.forms.NumpyForm("int64"),
            ],
            None if is_tuple else ["x", "y"],
        )
        for is_tuple in (False, True)
    )

    assert not records.is_tuple
    assert records_tuple.is_tuple

    assert not records.select_columns("*").is_tuple
    assert records_tuple.select_columns("*").is_tuple
