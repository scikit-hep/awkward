# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray


def forbidden():
    pytest.fail("trimming must not materialize the virtual buffer")


@pytest.mark.parametrize("fields", [None, ["x"]])
def test_concatenate_trims_overlong_record_fields(fields):
    left = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.array([1, 2, 999]))], fields, length=2
    )
    right = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.array([3, 999]))], fields, length=1
    )
    result = ak.concatenate([left, right], highlevel=False)
    assert result.contents[0].to_list() == [1, 2, 3]


def test_record_field_trim_preserves_virtual_buffer():
    field = ak.contents.NumpyArray(
        VirtualNDArray(
            Numpy.instance(), shape=(3,), dtype=np.dtype("int64"), generator=forbidden
        )
    )
    trimmed = ak.contents.RecordArray._trim_merge_field(field, 2)
    assert trimmed.length == 2
    assert not trimmed.data.is_materialized
