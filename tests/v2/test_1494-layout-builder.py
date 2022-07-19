# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401
import os

cppyy = pytest.importorskip("cppyy")


cppyy.add_include_path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.pardir,
            os.pardir,
            "awkward",
            "_v2",
            "cpp-headers",
        )
    )
)

cppyy.include("awkward/LayoutBuilder.h")


def test_numpy_layout_builder():
    NumpyBuilder = cppyy.gbl.awkward.LayoutBuilder.Numpy[1024, "double"]
    builder = NumpyBuilder()

    builder.append(1.1)
    builder.append(2.2)
    builder.append(3.3)
    builder.append(4.4)
    builder.append(5.5)

    # names_nbytes = map[str, int]
    # builder.buffer_nbytes(names_nbytes)
    # print(names_nbytes)

    form = builder.form()
    assert (
        str(form)
        == """{ "class": "NumpyArray", "primitive": "float64", "form_key": "node0" }"""
    )

    length = builder.length()
    ptr = ak.nplike.numpy.empty(length, np.float64)
    builder.to_buffers(ptr)

    array = ak._v2.operations.from_numpy(ptr)
    assert ak._v2.to_list(array) == [1.1, 2.2, 3.3, 4.4, 5.5]
