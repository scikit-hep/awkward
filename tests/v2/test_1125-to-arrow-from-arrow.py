# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pyarrow = pytest.importorskip("pyarrow")


def test_numpyarray():
    akarray = ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3]))
    paarray = akarray.to_arrow()
    assert (
        ak.to_list(akarray) == paarray.to_numpy().tolist()
    )  # FIXME Arrow 6.0: paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type


def test_numpyarray_parameters():
    akarray = ak._v2.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3]), parameters={"which": "only"}
    )
    paarray = akarray.to_arrow()
    assert (
        ak.to_list(akarray) == paarray.to_numpy().tolist()
    )  # FIXME Arrow 6.0: paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type
    assert akarray2.parameter("which") == "only"


def test_unmaskedarray_numpyarray():
    akarray = ak._v2.contents.UnmaskedArray(
        ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3]))
    )
    paarray = akarray.to_arrow()
    assert (
        ak.to_list(akarray) == paarray.to_numpy().tolist()
    )  # FIXME Arrow 6.0: paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type


def test_unmaskedarray_numpyarray_parameters():
    akarray = ak._v2.contents.UnmaskedArray(
        ak._v2.contents.NumpyArray(
            np.array([1.1, 2.2, 3.3]), parameters={"which": "inner"}
        ),
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow()
    assert (
        ak.to_list(akarray) == paarray.to_numpy().tolist()
    )  # FIXME Arrow 6.0: paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type
    assert akarray2.parameter("which") == "outer"
    assert akarray2.content.parameter("which") == "inner"
