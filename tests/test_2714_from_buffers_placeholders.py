# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import unknown_length

numpy = Numpy.instance()


def test_numpyarray():
    layout = ak.from_buffers(
        {"class": "NumpyArray", "primitive": "int64", "form_key": "node0"},
        10,
        {"node0-data": PlaceholderArray(numpy, (10,), np.int64)},
        highlevel=False,
    )
    assert layout.length == 10

    # Content too small
    with pytest.raises(TypeError, match=r"is less than size of form"):
        ak.from_buffers(
            {"class": "NumpyArray", "primitive": "int64", "form_key": "node0"},
            10,
            {"node0-data": PlaceholderArray(numpy, (9,), np.int64)},
            highlevel=False,
        )

    # Unknown length content at top-level
    layout = ak.from_buffers(
        {"class": "NumpyArray", "primitive": "int64", "form_key": "node0"},
        10,
        {"node0-data": PlaceholderArray(numpy, (unknown_length,), np.int64)},
        highlevel=False,
    )
    assert layout.length == 10


def test_listoffsetarray_numpyarray():
    # Unknown data
    layout = ak.from_buffers(
        {
            "class": "ListOffsetArray",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "offsets": "i64",
            "form_key": "node0",
        },
        2,
        {
            "node0-offsets": np.array([0, 1, 2], dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (10,), dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 2
    assert layout.content.length == 2

    # Unknown offsets above known data
    layout = ak.from_buffers(
        {
            "class": "ListOffsetArray",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "offsets": "i64",
            "form_key": "node0",
        },
        2,
        {
            "node0-offsets": PlaceholderArray(numpy, (3,), dtype=np.int64),
            "node1-data": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 2
    assert layout.content.length is unknown_length

    # Unknown offsets and unknown data
    layout = ak.from_buffers(
        {
            "class": "ListOffsetArray",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "offsets": "i64",
            "form_key": "node0",
        },
        2,
        {
            "node0-offsets": PlaceholderArray(numpy, (3,), dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (10,), dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 2
    assert layout.content.length is unknown_length

    # Unknown offsets and unknown data with unknown offset length
    layout = ak.from_buffers(
        {
            "class": "ListOffsetArray",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "offsets": "i64",
            "form_key": "node0",
        },
        2,
        {
            "node0-offsets": PlaceholderArray(numpy, (unknown_length,), dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (10,), dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 2
    assert layout.content.length is unknown_length


def test_listarray_numpyarray():
    # Unknown data
    layout = ak.from_buffers(
        {
            "class": "ListArray",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "starts": "i64",
            "stops": "i64",
            "form_key": "node0",
        },
        2,
        {
            "node0-starts": np.array([0, 1], dtype=np.int64),
            "node0-stops": np.array([1, 2], dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (10,), dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 2
    assert layout.content.length == 2

    # Unknown offsets
    layout = ak.from_buffers(
        {
            "class": "ListArray",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "starts": "i64",
            "stops": "i64",
            "form_key": "node0",
        },
        2,
        {
            "node0-starts": PlaceholderArray(numpy, (2,), dtype=np.int64),
            "node0-stops": PlaceholderArray(numpy, (2,), dtype=np.int64),
            "node1-data": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 2
    assert layout.content.length is unknown_length

    # Unknown offsets and unknown data
    layout = ak.from_buffers(
        {
            "class": "ListArray",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "starts": "i64",
            "stops": "i64",
            "form_key": "node0",
        },
        2,
        {
            "node0-starts": PlaceholderArray(numpy, (2,), dtype=np.int64),
            "node0-stops": PlaceholderArray(numpy, (2,), dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (10,), dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 2
    assert layout.content.length is unknown_length

    # Unknown starts, stops, and data with unknown starts lengths
    layout = ak.from_buffers(
        {
            "class": "ListArray",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "starts": "i64",
            "stops": "i64",
            "form_key": "node0",
        },
        2,
        {
            "node0-starts": PlaceholderArray(numpy, (unknown_length,), dtype=np.int64),
            "node0-stops": PlaceholderArray(numpy, (2,), dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (10,), dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 2
    assert layout.content.length is unknown_length


def test_indexedoptionarray():
    # Unknown data
    layout = ak.from_buffers(
        {
            "class": "IndexedOptionArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "form_key": "node0",
        },
        3,
        {
            "node0-index": np.array([0, 1, 2], dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (3,), np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.content.length == 3

    # Unknown index
    layout = ak.from_buffers(
        {
            "class": "IndexedOptionArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "form_key": "node0",
        },
        3,
        {
            "node0-index": PlaceholderArray(numpy, (3,), np.int64),
            "node1-data": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.content.length is unknown_length

    # Unknown index and data
    layout = ak.from_buffers(
        {
            "class": "IndexedOptionArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "form_key": "node0",
        },
        3,
        {
            "node0-index": PlaceholderArray(numpy, (3,), np.int64),
            "node1-data": PlaceholderArray(numpy, (6,), np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.content.length is unknown_length

    # Unknown index and data with unknown index length
    layout = ak.from_buffers(
        {
            "class": "IndexedOptionArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "form_key": "node0",
        },
        3,
        {
            "node0-index": PlaceholderArray(numpy, (unknown_length,), np.int64),
            "node1-data": PlaceholderArray(numpy, (6,), np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.content.length is unknown_length


def test_indexedarray():
    # Unknown data
    layout = ak.from_buffers(
        {
            "class": "IndexedArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "form_key": "node0",
        },
        3,
        {
            "node0-index": np.array([0, 1, 2], dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (3,), np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.content.length == 3

    # Unknown index
    layout = ak.from_buffers(
        {
            "class": "IndexedArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "form_key": "node0",
        },
        3,
        {
            "node0-index": PlaceholderArray(numpy, (3,), np.int64),
            "node1-data": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.content.length is unknown_length

    # Unknown data
    layout = ak.from_buffers(
        {
            "class": "IndexedArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "form_key": "node0",
        },
        3,
        {
            "node0-index": PlaceholderArray(numpy, (3,), np.int64),
            "node1-data": PlaceholderArray(numpy, (6,), np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.content.length is unknown_length

    # Unknown index and data, with unknown index length
    layout = ak.from_buffers(
        {
            "class": "IndexedArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1",
            },
            "form_key": "node0",
        },
        3,
        {
            "node0-index": PlaceholderArray(numpy, (unknown_length,), np.int64),
            "node1-data": PlaceholderArray(numpy, (6,), np.int64),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.content.length is unknown_length


def test_unionarray():
    # Unknown data
    layout = ak.from_buffers(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "form_key": "node1",
                },
                {
                    "class": "NumpyArray",
                    "primitive": "datetime64[D]",
                    "form_key": "node2",
                },
            ],
            "form_key": "node0",
        },
        3,
        {
            "node0-tags": np.array([0, 0, 1], dtype=np.int8),
            "node0-index": np.array([0, 1, 0], dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (3,), np.int64),
            "node2-data": PlaceholderArray(numpy, (6,), np.dtype("datetime64[D]")),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.contents[0].length == 2
    assert layout.contents[1].length == 1

    # Unknown tags
    layout = ak.from_buffers(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "form_key": "node1",
                },
                {
                    "class": "NumpyArray",
                    "primitive": "datetime64[D]",
                    "form_key": "node2",
                },
            ],
            "form_key": "node0",
        },
        3,
        {
            "node0-tags": PlaceholderArray(numpy, (3,), np.int8),
            "node0-index": np.array([0, 1, 0], dtype=np.int64),
            "node1-data": np.array([0, 1, 2], np.int64),
            "node2-data": np.array([0, 1, 2, 3, 4, 5, 6], np.dtype("datetime64[D]")),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.contents[0].length is unknown_length
    assert layout.contents[1].length is unknown_length

    # Unknown index
    layout = ak.from_buffers(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "form_key": "node1",
                },
                {
                    "class": "NumpyArray",
                    "primitive": "datetime64[D]",
                    "form_key": "node2",
                },
            ],
            "form_key": "node0",
        },
        3,
        {
            "node0-tags": np.array([0, 0, 1], dtype=np.int8),
            "node0-index": PlaceholderArray(numpy, (3,), np.int64),
            "node1-data": np.array([0, 1, 2], np.int64),
            "node2-data": np.array([0, 1, 2, 3, 4, 5, 6], np.dtype("datetime64[D]")),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.contents[0].length is unknown_length
    assert layout.contents[1].length is unknown_length

    # Unknown content length
    layout = ak.from_buffers(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "form_key": "node1",
                },
                {
                    "class": "NumpyArray",
                    "primitive": "datetime64[D]",
                    "form_key": "node2",
                },
            ],
            "form_key": "node0",
        },
        3,
        {
            "node0-tags": np.array([0, 0, 1], dtype=np.int8),
            "node0-index": np.array([0, 1, 0], dtype=np.int64),
            "node1-data": PlaceholderArray(numpy, (unknown_length,), np.int64),
            "node2-data": PlaceholderArray(numpy, (6,), np.dtype("datetime64[D]")),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.contents[0].length == 2
    assert layout.contents[1].length == 1

    # Unknown tags, index, and data
    layout = ak.from_buffers(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "form_key": "node1",
                },
                {
                    "class": "NumpyArray",
                    "primitive": "datetime64[D]",
                    "form_key": "node2",
                },
            ],
            "form_key": "node0",
        },
        3,
        {
            "node0-tags": PlaceholderArray(numpy, (3,), np.int8),
            "node0-index": PlaceholderArray(numpy, (3,), np.int64),
            "node1-data": PlaceholderArray(numpy, (3,), np.int64),
            "node2-data": PlaceholderArray(numpy, (6,), np.dtype("datetime64[D]")),
        },
        highlevel=False,
    )
    assert layout.length == 3
    assert layout.contents[0].length is unknown_length
    assert layout.contents[1].length is unknown_length
