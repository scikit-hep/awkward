# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize(
    ("form", "length", "container", "expected"),
    [
        pytest.param(
            ak.forms.ListOffsetForm(
                "i64", ak.forms.RecordForm([], None), form_key="list"
            ),
            1,
            {"list-offsets": lambda: np.array([0, 3], dtype=np.int64)},
            [[(), (), ()]],
            id="ListOffsetArray-tuple",
        ),
        pytest.param(
            ak.forms.ListOffsetForm(
                "i64", ak.forms.RecordForm([], []), form_key="list"
            ),
            1,
            {"list-offsets": lambda: np.array([0, 3], dtype=np.int64)},
            [[{}, {}, {}]],
            id="ListOffsetArray-record",
        ),
        pytest.param(
            ak.forms.ListOffsetForm(
                "i64", ak.forms.RecordForm([], None), form_key="list"
            ),
            1,
            {"list-offsets": lambda: np.array([0, 0], dtype=np.int64)},
            [[]],
            id="ListOffsetArray-empty-list",
        ),
        pytest.param(
            ak.forms.ListOffsetForm(
                "i64", ak.forms.RecordForm([], None), form_key="list"
            ),
            0,
            {"list-offsets": lambda: np.array([0], dtype=np.int64)},
            [],
            id="ListOffsetArray-zero-length",
        ),
        pytest.param(
            ak.forms.ListForm(
                "i64", "i64", ak.forms.RecordForm([], None), form_key="list"
            ),
            1,
            {
                "list-starts": lambda: np.array([0], dtype=np.int64),
                "list-stops": lambda: np.array([2], dtype=np.int64),
            },
            [[(), ()]],
            id="ListArray",
        ),
        pytest.param(
            ak.forms.IndexedForm(
                "i64", ak.forms.RecordForm([], []), form_key="indexed"
            ),
            3,
            {"indexed-index": lambda: np.array([2, 0, 1], dtype=np.int64)},
            [{}, {}, {}],
            id="IndexedArray",
        ),
        pytest.param(
            ak.forms.ByteMaskedForm(
                "i8",
                ak.forms.ListOffsetForm(
                    "i64", ak.forms.RecordForm([], None), form_key="list"
                ),
                valid_when=True,
                form_key="mask",
            ),
            2,
            {
                "mask-mask": lambda: np.array([1, 0], dtype=np.int8),
                "list-offsets": lambda: np.array([0, 2, 3], dtype=np.int64),
            },
            [[(), ()], None],
            id="ByteMaskedArray-ListOffsetArray",
        ),
        pytest.param(
            ak.forms.ListOffsetForm(
                "i64",
                ak.forms.ListOffsetForm(
                    "i64", ak.forms.RecordForm([], None), form_key="inner"
                ),
                form_key="outer",
            ),
            1,
            {
                "outer-offsets": lambda: np.array([0, 2], dtype=np.int64),
                "inner-offsets": lambda: np.array([0, 1, 3], dtype=np.int64),
            },
            [[[()], [(), ()]]],
            id="nested-ListOffsetArray",
        ),
        pytest.param(
            ak.forms.ListOffsetForm(
                "i64",
                ak.forms.RecordForm([ak.forms.RecordForm([], None)], ["a"]),
                form_key="list",
            ),
            1,
            {"list-offsets": lambda: np.array([0, 2], dtype=np.int64)},
            [[{"a": ()}, {"a": ()}]],
            id="zero-field-record-as-field",
        ),
        pytest.param(
            ak.forms.RecordForm([], None),
            4,
            {},
            [(), (), (), ()],
            id="top-level",
        ),
        pytest.param(
            ak.forms.RegularForm(ak.forms.RecordForm([], []), 3),
            2,
            {},
            [[{}, {}, {}], [{}, {}, {}]],
            id="RegularArray",
        ),
    ],
)
def test_from_buffers(
    form: ak.forms.Form,
    length: int,
    container: dict[str, Callable[[], np.ndarray]],
    expected: list[Any],
) -> None:
    virtual = ak.from_buffers(form, length, container)
    assert len(virtual) == length
    assert ak.materialize(virtual).tolist() == expected
    assert virtual.tolist() == expected
