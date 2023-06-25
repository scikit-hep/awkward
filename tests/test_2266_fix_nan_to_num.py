# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np

import awkward as ak


def test():
    array = ak.Array([[2.0, 3.0], [1.0, 9.0, 8.0, np.nan]], backend="typetracer")
    result = ak.nan_to_num(array + 3)
    assert result.layout.form == array.layout.form
