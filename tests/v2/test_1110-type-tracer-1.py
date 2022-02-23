# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

typetracer = ak._v2._typetracer.TypeTracer.instance()


def test_getitem_at():
    concrete = ak._v2.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5) * 0.1)
    abstract = ak._v2.contents.NumpyArray(concrete.raw(typetracer))

    assert concrete.shape == (2, 3, 5)
    assert abstract.shape[1:] == (3, 5)
    assert abstract[0].shape[1:] == (5,)
    assert abstract[0][0].shape[1:] == ()

    assert abstract.form == concrete.form
    assert abstract.form.type == concrete.form.type

    assert abstract[0].form == concrete[0].form
    assert abstract[0].form.type == concrete[0].form.type
