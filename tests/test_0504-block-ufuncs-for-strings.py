# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test():
    def _apply_ufunc(ufunc, method, inputs, kwargs):
        nextinputs = []
        for x in inputs:
            if (
                isinstance(x, ak.highlevel.Array)
                and x.layout.is_indexed
                and not x.layout.is_option
            ):
                nextinputs.append(
                    ak.highlevel.Array(
                        x.layout.project(), behavior=ak._util.behavior_of(x)
                    )
                )
            else:
                nextinputs.append(x)

        return getattr(ufunc, method)(*nextinputs, **kwargs)

    behavior = {}
    behavior[np.ufunc, "categorical"] = _apply_ufunc

    array = ak.highlevel.Array(
        ak.contents.IndexedArray(
            ak.index.Index64(np.array([0, 1, 2, 1, 3, 1, 4])),
            ak.contents.NumpyArray(np.array([321, 1.1, 123, 999, 2])),
            parameters={"__array__": "categorical"},
        ),
        behavior=behavior,
    )
    assert to_list(array * 10) == [3210, 11, 1230, 11, 9990, 11, 20]

    array = ak.highlevel.Array(["HAL"])
    with pytest.raises(TypeError):
        array + 1
