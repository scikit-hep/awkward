# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
    array = ak.Array([{"x": [[1, 2, 3], [4, 5, 6]]}])
    assert array[:, []].fields == ["x"]
    assert ak.to_list(ak.Array([{"x": [[1, 2, 3], [4, 5, 6]]}])[:, []]) == [{"x": []}]
