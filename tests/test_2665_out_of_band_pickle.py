# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pickle

import numpy as np

import awkward as ak


def test_protocol_5():
    # Check arrays
    array = ak.Array([[1, 2, 4, 5, 6], [7, 8, 9], [10]])
    buffers = []

    array_round_trip = pickle.loads(
        pickle.dumps(array, buffer_callback=buffers.append, protocol=5), buffers=buffers
    )
    assert np.shares_memory(
        array_round_trip.layout.content.data, array.layout.content.data
    )

    # Now check records
    record = ak.Record({"x": 1, "y": [2, 3]})
    buffers = []

    record_round_trip = pickle.loads(
        pickle.dumps(record, buffer_callback=buffers.append, protocol=5),
        buffers=buffers,
    )
    assert np.shares_memory(
        record.layout.array.contents[0].data,
        record_round_trip.layout.array.contents[0].data,
    )
    assert np.shares_memory(
        record.layout.array.contents[1].content.data,
        record_round_trip.layout.array.contents[1].content.data,
    )


def test_protocol_4():
    array = ak.Array([[1, 2, 4, 5, 6], [7, 8, 9], [10]])

    array_round_trip = pickle.loads(pickle.dumps(array, protocol=4))
    assert not np.shares_memory(
        array_round_trip.layout.content.data, array.layout.content.data
    )

    # Now check records
    record = ak.Record({"x": 1, "y": [2, 3]})

    record_round_trip = pickle.loads(pickle.dumps(record, protocol=4))
    assert not np.shares_memory(
        record.layout.array.contents[0].data,
        record_round_trip.layout.array.contents[0].data,
    )
    assert not np.shares_memory(
        record.layout.array.contents[1].content.data,
        record_round_trip.layout.array.contents[1].content.data,
    )
