# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np


def test():
    record = ak._v2.contents.RecordArray(
        [ak._v2.contents.NumpyArray(np.arange(10))], ["x"]
    )
    array = ak._v2.Array(record)

    with pytest.raises(AttributeError):
        array.x = 10

    with pytest.raises(AttributeError):
        array.not_an_existing_attribute = 10

    array._not_an_existing_attribute = 10
    assert array._not_an_existing_attribute == 10


class BadBehavior(ak._v2.Array):
    FIELD_STRING = "I am not a list of fields!"

    @property
    def fields(self):
        return self.FIELD_STRING


behavior = {("*", "bad"): BadBehavior}


def test_bad_behavior():
    record = ak._v2.contents.RecordArray(
        [ak._v2.contents.NumpyArray(np.arange(10))], ["x"]
    )
    array = ak._v2.Array(record, with_name="bad", behavior=behavior)
    assert isinstance(array, BadBehavior)
    assert array.fields == BadBehavior.FIELD_STRING
