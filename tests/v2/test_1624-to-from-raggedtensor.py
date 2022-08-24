# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

tf = pytest.importorskip("tensorflow")


def test_empty_array_to_ragged_tensor():
    array = ak._v2.contents.EmptyArray()

    with pytest.raises(TypeError):
        ak._v2.to_raggedtensor(array)


def test_string_array_to_ragged_tensor():
    array = ak._v2.from_iter([b"this", b"is", b"an", b"array", b"of", b"strings"])

    tensor = ak._v2.to_raggedtensor(array)
    result = ak._v2.from_raggedtensor(tensor)
    assert result.tolist() == array.tolist()


def test_list_of_bytestring_array_to_ragged_tensor():
    array = ak._v2.from_iter(
        [
            [b"this", b"is"],
            [b"an", b"array", b"of", b"strings"],
        ]
    )

    tensor = ak._v2.to_raggedtensor(array)
    assert tensor.to_list() == array.tolist()


def test_numpy_array_to_ragged_tensor():
    for dtype in (np.float32, np.float64):
        array = ak._v2.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3], dtype=dtype))

        tensor = ak._v2.to_raggedtensor(array)
        assert tensor.numpy().tolist() == array.tolist()
        assert tensor.dtype == dtype


def test_regular_array_numpy_array_to_ragged_tensor():
    array = ak._v2.contents.RegularArray(
        ak._v2.contents.NumpyArray(
            np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)
        ),
        3,
    )

    tensor = ak._v2.to_raggedtensor(array)
    assert tensor.numpy().tolist() == array.tolist()


def test_list_array_numpy_array_to_ragged_tensor():
    array = ak._v2.contents.ListArray(
        ak._v2.index.Index(np.array([0, 5, 3], np.int64)),
        ak._v2.index.Index(np.array([3, 8, 5], np.int64)),
        ak._v2.contents.NumpyArray(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])),
    )

    tensor = ak._v2.to_raggedtensor(array)
    assert tensor.to_list() == array.tolist()


def test_nested_list_offset_array_numpy_array_to_ragged_tensor():
    array = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 2, 5], dtype=np.int64)),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index(np.array([1, 1, 2, 4, 6, 7], np.int64)),
            ak._v2.contents.NumpyArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]),
        ),
    )

    tensor = ak._v2.to_raggedtensor(array)
    assert tensor.to_list() == array.tolist()


def test_record_array_to_ragged_tensor():
    array = ak._v2.contents.RecordArray(
        [
            ak._v2.contents.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4])),
        ],
        ["x", "y"],
    )

    with pytest.raises(TypeError):
        ak._v2.to_raggedtensor(array)


def test_record_to_ragged_tensor():
    array = ak._v2.contents.RecordArray(
        [
            ak._v2.contents.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4])),
        ],
        ["x", "y"],
    )
    record = array[0]
    assert isinstance(record, ak._v2.record.Record)

    with pytest.raises(TypeError):
        ak._v2.to_raggedtensor(record)


def test_indexed_array_numpy_array_to_ragged_tensor():
    array = ak._v2.contents.IndexedArray(
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak._v2.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    tensor = ak._v2.to_raggedtensor(array)
    assert tensor.numpy().tolist() == array.tolist()


def test_indexed_option_array_numpy_array_to_ragged_tensor():
    array = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    with pytest.raises(TypeError):
        ak._v2.to_raggedtensor(array)


def test_byte_masked_array_numpy_array_to_ragged_tensor():
    array = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )

    with pytest.raises(TypeError):
        ak._v2.to_raggedtensor(array)


def test_bit_masked_array_numpy_array_to_ragged_tensor():
    array = ak._v2.contents.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    with pytest.raises(TypeError):
        ak._v2.to_raggedtensor(array)


def test_nested_unmasked_array_numpy_array_to_ragged_tensor():
    array = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak._v2.contents.UnmaskedArray(
            ak._v2.contents.NumpyArray(np.array([999, 0.0, 1.1, 2.2, 3.3]))
        ),
    )

    with pytest.raises(TypeError):
        ak._v2.to_raggedtensor(array)


def test_union_array_numpy_array_to_ragged_tensor():
    array = ak._v2.contents.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak._v2.contents.NumpyArray(np.array([1, 2, 3], np.int64)),
            ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )

    with pytest.raises(TypeError):
        ak._v2.to_raggedtensor(array)


def test_ragged_tensor_strings_to_array():
    paragraphs = [
        [["I", "have", "a", "cat"], ["His", "name", "is", "Mat"]],
        [["Do", "you", "want", "to", "come", "visit"], ["I'm", "free", "tomorrow"]],
    ]
    tensor = tf.ragged.constant(paragraphs)
    array = ak._v2.from_raggedtensor(tensor)
    assert array.to_list() == paragraphs
