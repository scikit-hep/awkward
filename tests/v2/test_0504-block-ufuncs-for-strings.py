# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test():
    def _apply_ufunc(ufunc, method, inputs, kwargs):
        nextinputs = []
        for x in inputs:
            if (
                isinstance(x, ak._v2.highlevel.Array)
                and x.layout.is_IndexedType
                and not x.layout.is_OptionType
            ):
                nextinputs.append(
                    ak._v2.highlevel.Array(
                        x.layout.project(), behavior=ak._v2._util.behavior_of(x)
                    )
                )
            else:
                nextinputs.append(x)

        return getattr(ufunc, method)(*nextinputs, **kwargs)

    ak._v2.behavior[np.ufunc, "categorical"] = _apply_ufunc

    array = ak._v2.highlevel.Array(
        ak._v2.contents.IndexedArray(
            ak._v2.index.Index64(np.array([0, 1, 2, 1, 3, 1, 4])),
            ak._v2.contents.NumpyArray(np.array([321, 1.1, 123, 999, 2])),
            parameters={"__array__": "categorical"},
        )
    )
    assert ak.to_list(array * 10) == [3210, 11, 1230, 11, 9990, 11, 20]

    array = ak.Array(["HAL"])
    with pytest.raises(ValueError):
        array + 1
