# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak

def test_flatten():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=np.int64))
    array = ak.layout.ListOffsetArray64(offsets, content)

    assert ak.to_list(array) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(array.flatten(axis=1)) == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert ak.to_list(array.flatten(axis=-1)) == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    with pytest.raises(ValueError) as err:
        assert ak.to_list(array.flatten(axis=-2))
    assert str(err.value).startswith("axis=0 not allowed for flatten")

    array2 = array[2:-1]
    assert ak.to_list(array2.flatten(axis=1)) == [3.3, 4.4, 5.5]
    assert ak.to_list(array2.flatten(axis=-1)) == [3.3, 4.4, 5.5]
