from __future__ import annotations

import awkward as ak


def test():
    array = ak.typetracer.typetracer_from_form(
        ak.forms.from_type(
            ak.types.from_datashape("{x: var * {y:int64}}", highlevel=False)
        )
    )
    sliced = array[0, "x", 0]
    assert sliced.type.is_equal_to(
        ak.types.ScalarType(
            ak.types.RecordType([ak.types.NumpyType("int64")], ["y"]), None
        )
    )
