# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np


def test_array():
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


def test_record():
    record = ak._v2.contents.RecordArray(
        [ak._v2.contents.NumpyArray(np.arange(10))], ["x"]
    )
    array = ak._v2.Array(record)
    record = array[0]

    with pytest.raises(AttributeError):
        record.x = 10

    with pytest.raises(AttributeError):
        record.not_an_existing_attribute = 10

    record._not_an_existing_attribute = 10
    assert record._not_an_existing_attribute == 10


class BadBehaviorBase:
    FIELD_STRING = "I am not a list of fields!"

    @property
    def fields(self):
        return self.__class__.FIELD_STRING

    @fields.setter
    def fields(self, value):
        self.__class__.FIELD_STRING = value


class BadBehaviorArray(BadBehaviorBase, ak._v2.Array):
    pass


class BadBehaviorRecord(BadBehaviorBase, ak._v2.Record):
    pass


behavior = {("*", "bad"): BadBehaviorArray, "bad": BadBehaviorRecord}


def test_bad_behavior_array():
    record = ak._v2.contents.RecordArray(
        [ak._v2.contents.NumpyArray(np.arange(10))], ["x"]
    )
    array = ak._v2.Array(record, with_name="bad", behavior=behavior)
    assert isinstance(array, BadBehaviorArray)

    assert array.fields == BadBehaviorArray.FIELD_STRING
    array.fields = "yo ho ho and a bottle of rum"
    assert array.fields == "yo ho ho and a bottle of rum"


def test_bad_behavior_record():
    record = ak._v2.contents.RecordArray(
        [ak._v2.contents.NumpyArray(np.arange(10))], ["x"]
    )
    array = ak._v2.Array(record, with_name="bad", behavior=behavior)
    record = array[0]
    assert isinstance(record, BadBehaviorRecord)

    assert record.fields == BadBehaviorRecord.FIELD_STRING
    record.fields = "yo ho ho and a bottle of rum"
    assert record.fields == "yo ho ho and a bottle of rum"
