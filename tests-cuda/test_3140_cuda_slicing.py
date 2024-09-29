from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_0315_integerindex_null_more():
    f = ak.highlevel.Array([[0, None, 2], None, [3, 4], []], backend="cuda").layout
    g1 = ak.highlevel.Array([[1, 2, None], None, [], [None]], backend="cuda").layout
    g2 = ak.highlevel.Array([[], None, None, []], backend="cuda").layout
    g3 = ak.highlevel.Array([[], [], [], []], backend="cuda").layout

    assert to_list(f[g1]) == [[None, 2, None], None, [], [None]]
    assert to_list(f[g2]) == [[], None, None, []]
    assert to_list(f[g3]) == [[], None, [], []]
    assert f.to_typetracer()[g1].form == f[g1].form
    assert f.to_typetracer()[g2].form == f[g2].form
    assert f.to_typetracer()[g3].form == f[g3].form

    a = ak.highlevel.Array([[0, 1, 2, None], None], backend="cuda").layout
    b = ak.highlevel.Array([[2, 1, None, 3], None], backend="cuda").layout

    assert to_list(a[b]) == [[2, 1, None, None], None]
    assert a.to_typetracer()[b].form == a[b].form

    b = ak.highlevel.Array([[2, 1, None, 3], []], backend="cuda").layout

    assert to_list(a[b]) == [[2, 1, None, None], None]
    assert a.to_typetracer()[b].form == a[b].form

    b = ak.highlevel.Array([[2, 1, None, 3], [0, 1]], backend="cuda").layout
    assert to_list(a[b]) == [[2, 1, None, None], None]
    assert a.to_typetracer()[b].form == a[b].form


def test_0315_integerindex_null_more_2():
    a = ak.highlevel.Array(
        [[[0, 1, 2, None], None], [[3, 4], [5]], None, [[6]]], backend="cuda"
    ).layout
    b = ak.highlevel.Array(
        [[[2, 1, None, 3], [0, 1]], [[0], None], None, [None]], backend="cuda"
    ).layout
    c = ak.highlevel.Array(
        [
            [[False, True, None, False], [False, True]],
            [[True, False], None],
            None,
            [None],
        ],
        backend="cuda",
    ).layout

    assert to_list(a[b]) == [
        [[2, 1, None, None], None],
        [[3], None],
        None,
        [None],
    ]
    assert to_list(a[c]) == [[[1, None], None], [[4], None], None, [None]]
    assert a.to_typetracer()[c].form == a[c].form


def test_1405_slicing_untested_cases_list_option_list():
    """Check that non-offset list(option(list indexes correctly"""
    content = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([2, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2, 2, 2], dtype=np.int64)),
        ),
    )

    index = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, 1], dtype=np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ),
        ),
    )

    cuda_content = ak.to_backend(content, "cuda")
    cuda_index = ak.to_backend(index, "cuda")

    assert cuda_content[cuda_index].to_list() == [[[], [2]]]


def test_1405_slicing_untested_cases_list_option_list_offset():
    """Check that offset list(option(list indexes correctly"""
    content = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([1, 3], dtype=np.int64)),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2, 2, 2], dtype=np.int64)),
        ),
    )

    index = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, 1], dtype=np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ),
        ),
    )

    cuda_content = ak.to_backend(content, "cuda")
    cuda_index = ak.to_backend(index, "cuda")

    assert cuda_content[cuda_index].to_list() == [[[], [2]]]


def test_1502_getitem_jagged_issue1406():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([1, 3], dtype=np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 2, 2, 3], dtype=np.int64)),
                ak.contents.NumpyArray(np.array([0, 1, 2], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    index = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak.contents.IndexedOptionArray(
                ak.index.Index64(np.array([0, 1], dtype=np.int64)),
                ak.contents.ListOffsetArray(
                    ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                    ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
                ),
            ),
        ),
        check_valid=True,
    )

    cuda_array = ak.to_backend(array, "cuda")
    cuda_index = ak.to_backend(index, "cuda")

    assert to_list(cuda_array[cuda_index]) == [[[], [2]]]


def test_1502_getitem_jagged_issue1406_success_start_offset0():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([2, 2, 3], dtype=np.int64)),
                ak.contents.NumpyArray(np.array([0, 1, 2], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    index = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak.contents.IndexedOptionArray(
                ak.index.Index64(np.array([0, 1], dtype=np.int64)),
                ak.contents.ListOffsetArray(
                    ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                    ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
                ),
            ),
        ),
        check_valid=True,
    )

    cuda_array = ak.to_backend(array, "cuda")
    cuda_index = ak.to_backend(index, "cuda")

    assert to_list(cuda_array[cuda_index]) == [[[], [2]]]


def test_1502_getitem_jagged_issue1406_success_remove_option_type():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([1, 3], dtype=np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 2, 2, 3], dtype=np.int64)),
                ak.contents.NumpyArray(np.array([0, 1, 2], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    index = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    cuda_array = ak.to_backend(array, "cuda")
    cuda_index = ak.to_backend(index, "cuda")

    assert to_list(cuda_array[cuda_index]) == [[[], [2]]]


def test_1502_getitem_jagged_issue1406_success_nonempty_list():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([1, 3], dtype=np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 1, 2, 3], dtype=np.int64)),
                ak.contents.NumpyArray(np.array([0, 1, 2], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    index = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak.contents.IndexedOptionArray(
                ak.index.Index64(np.array([0, 1], dtype=np.int64)),
                ak.contents.ListOffsetArray(
                    ak.index.Index64(np.array([0, 1, 2], dtype=np.int64)),
                    ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
                ),
            ),
        ),
        check_valid=True,
    )

    cuda_array = ak.to_backend(array, "cuda")
    cuda_index = ak.to_backend(index, "cuda")

    assert to_list(cuda_array[cuda_index]) == [[[1], [2]]]


def test_1904_drop_none_ListArray_and_axis_None():
    index = ak.index.Index64(np.asarray([0, -1, 1, -1, 4, -1, 5]))
    content = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(
                np.array([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])
            )
        ],
        ["nest"],
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)
    a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        indexoptionarray,
    )

    cuda_a = ak.to_backend(a, "cuda")

    assert (
        to_list(ak.drop_none(cuda_a))
        == to_list(cuda_a[~ak.is_none(cuda_a, axis=1)])
        == [
            [{"nest": 4.4}, {"nest": 5.5}],
            [],
            [{"nest": 1.1}],
        ]
    )


def test_1904_drop_none_ListOffsetArray_IndexedOptionArray_NumpyArray_outoforder():
    index = ak.index.Index64(np.asarray([0, -1, 1, 5, 4, 2, 5]))
    content = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)
    offsets = ak.index.Index64(np.asarray([0, 4, 5, 6]))
    listoffset = ak.contents.ListOffsetArray(offsets, indexoptionarray)

    cuda_listoffset = ak.to_backend(listoffset, "cuda")

    assert to_list(cuda_listoffset) == [[0.0, None, 1.1, 5.5], [4.4], [2.2]]
    assert (
        to_list(ak.drop_none(cuda_listoffset, axis=1))
        == to_list(cuda_listoffset[~ak.is_none(cuda_listoffset, axis=1)])
        == [[0.0, 1.1, 5.5], [4.4], [2.2]]
    )
    assert to_list(ak.drop_none(cuda_listoffset)) == [[0.0, 1.1, 5.5], [4.4], [2.2]]


def test_1904_drop_none_from_iter():
    a = ak.Array([[1], [2, None]], backend="cuda")
    assert to_list(ak.drop_none(a)) == [[1], [2]]

    a = ak.Array([[2, None]], backend="cuda")
    assert to_list(ak.drop_none(a)) == [[2]]

    a = ak.Array([[[None]]], backend="cuda")
    assert to_list(ak.drop_none(a)) == [[[]]]

    a = ak.Array([1, 2, None], backend="cuda")
    assert to_list(ak.drop_none(a, axis=0))

    a = ak.Array([[[1, None]], [[3, 4]], [[5, 6]], [[7.8]]], backend="cuda")
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.8]]]
    )

    a = ak.Array([[[0]], [[None]], [[1], None], [[2, None]]], backend="cuda")
    assert (
        to_list(ak.drop_none(a, axis=1))
        == to_list(a[~ak.is_none(a, axis=1)])
        == [[[0]], [[None]], [[1]], [[2, None]]]
    )

    a = ak.Array(
        [[[0]], [None, 34], [[1], None, 31], [[2, [[None]]]], [[[None]]]],
        backend="cuda",
    )
    assert (
        to_list(ak.drop_none(a, axis=0))
        == to_list(a[~ak.is_none(a, axis=0)])
        == [[[0]], [None, 34], [[1], None, 31], [[2, [[None]]]], [[[None]]]]
    )

    a = ak.Array([[[1, None]], [[3, None]], [[5, 6]], [[7.8]]], backend="cuda")
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[3.0]], [[5.0, 6.0]], [[7.8]]]
    )

    a = ak.Array([[[1, None]], [[None, 4]], [[5, 6]], [[7.8]]], backend="cuda")
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[4.0]], [[5.0, 6.0]], [[7.8]]]
    )

    a = ak.Array([[[1, None]], [[None, None]], [[5, 6]], [[7.8]]], backend="cuda")
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[]], [[5.0, 6.0]], [[7.8]]]
    )

    a = ak.Array([[[1, None]], [[None, None]], [[None, 6]], [[7.8]]], backend="cuda")
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[]], [[6.0]], [[7.8]]]
    )

    a = ak.Array(
        [[{"x": [1], "y": [[2]]}], [{"x": [None], "y": [[None]]}], None], backend="cuda"
    )
    assert to_list(a) == [
        [{"x": [1], "y": [[2]]}],
        [{"x": [None], "y": [[None]]}],
        None,
    ]
    assert to_list(ak.drop_none(a)) == [
        [{"x": [1], "y": [[2]]}],
        [{"x": [], "y": [[]]}],
    ]
    assert (
        to_list(ak.drop_none(a, axis=0))
        == to_list(a[~ak.is_none(a, axis=0)])
        == [[{"x": [1], "y": [[2]]}], [{"x": [None], "y": [[None]]}]]
    )
    assert to_list(ak.drop_none(a, axis=1)) == [
        [{"x": [1], "y": [[2]]}],
        [{"x": [], "y": [[None]]}],
        None,
    ]


def test_1904_drop_none_List_ByteMaskedArray_NumpyArray():
    a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([1, 3], np.int64)),
        ak.index.Index(np.array([3, 4], np.int64)),
        ak.contents.bytemaskedarray.ByteMaskedArray(
            ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            valid_when=True,
        ),
    )

    cuda_a = ak.to_backend(a, "cuda")

    assert to_list(cuda_a) == [[None, 3.3], [None]]
    assert to_list(ak.drop_none(cuda_a)) == [[3.3], []]
    assert to_list(ak.drop_none(cuda_a, axis=1)) == to_list(
        cuda_a[~ak.is_none(cuda_a, axis=1)]
    )


def test_1904_drop_none_RegularArray_RecordArray_NumpyArray():
    index = ak.index.Index64(np.asarray([0, -1, 1, 2, 3, 4, -1, 6, 7, 8, -1, 10]))
    content = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.listoffsetarray.ListOffsetArray(
        offsets, indexoptionarray
    )
    regulararray = ak.contents.regulararray.RegularArray(listoffsetarray, 2)

    cuda_regulararray = ak.to_backend(regulararray, "cuda")

    assert to_list(cuda_regulararray) == [
        [[0.0, None, 1.1], []],
        [[2.2, 3.3], [4.4]],
        [[None, 6.6, 7.7, 8.8], []],
    ]
    assert to_list(ak.drop_none(cuda_regulararray, axis=2)) == to_list(
        cuda_regulararray[~ak.is_none(cuda_regulararray, axis=2)]
    )


def test_1904_drop_none_RecordArray():
    a = ak.Array(
        [
            [{"x": [1], "y": [[2]]}],
            None,
            [None],
            [{"x": None, "y": None}],
            [{"x": [None], "y": [None]}],
            [{"x": [11], "y": [[None]]}],
        ],
        backend="cuda",
    )
    assert to_list(ak.drop_none(a, axis=1)) == to_list(a[~ak.is_none(a, axis=1)])
    assert to_list(ak.drop_none(a, axis=2)) == [
        [{"x": [1], "y": [[2]]}],
        None,
        [None],
        [{"x": None, "y": None}],
        [{"x": [], "y": []}],
        [{"x": [11], "y": [[None]]}],
    ]


def test_simple_slice():
    arrg = ak.Array([[1, 2, 3], [0], [4, 5]], backend="cuda")
    out = arrg[:, 0]
    assert out.tolist() == [1, 0, 4]


def test_2246_slice_not_packed():
    index = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 3, 5]),
            ak.contents.NumpyArray(
                np.array([True, False, False, True, True, False, False], dtype=np.bool_)
            ),
        )
    )
    array = ak.Array([[0, 1, 2], [3, 4]])

    cuda_index = ak.to_backend(index, "cuda")
    cuda_array = ak.to_backend(array, "cuda")

    result = cuda_array[cuda_index]
    assert result.tolist() == [[0], [3, 4]]


def test_0127_tomask_operation_ByteMaskedArray_jaggedslice0():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.index.Index64(np.array([0, 1, 2, 3], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, array)

    cuda_indexedarray = ak.to_backend(indexedarray, "cuda")

    assert to_list(cuda_indexedarray) == [
        [0.0, 1.1, 2.2],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert to_list(
        cuda_indexedarray[
            ak.highlevel.Array([[0, -1], [0], [], [1, 1]], backend="cuda")
        ]
    ) == [
        [0.0, 2.2],
        [3.3],
        [],
        [7.7, 7.7],
    ]

    mask = ak.index.Index8(np.array([0, 0, 0, 0], dtype=np.int8))
    maskedarray = ak.contents.ByteMaskedArray(mask, array, valid_when=False)

    cuda_maskedarray = ak.to_backend(maskedarray, "cuda")

    assert to_list(cuda_maskedarray) == [
        [0.0, 1.1, 2.2],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert to_list(
        cuda_maskedarray[ak.highlevel.Array([[0, -1], [0], [], [1, 1]], backend="cuda")]
    ) == [
        [0.0, 2.2],
        [3.3],
        [],
        [7.7, 7.7],
    ]


def test_0127_tomask_operation_ByteMaskedArray_jaggedslice1():
    model = ak.highlevel.Array(
        [
            [0.0, 1.1, None, 2.2],
            [],
            [3.3, None, 4.4],
            [5.5],
            [6.6, 7.7, None, 8.8, 9.9],
        ],
        backend="cuda",
    )
    assert to_list(
        model[
            ak.highlevel.Array(
                [[3, 2, 1, 1, 0], [], [1], [0, 0], [1, 2]], backend="cuda"
            )
        ]
    ) == [
        [2.2, None, 1.1, 1.1, 0.0],
        [],
        [None],
        [5.5, 5.5],
        [7.7, None],
    ]

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 999, 2.2, 3.3, 123, 4.4, 5.5, 6.6, 7.7, 321, 8.8, 9.9])
    )
    mask = ak.index.Index8(
        np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    )
    maskedarray = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    offsets = ak.index.Index64(np.array([0, 4, 4, 7, 8, 13], dtype=np.int64))
    listarray = ak.highlevel.Array(
        ak.contents.ListOffsetArray(offsets, maskedarray), backend="cuda"
    )

    assert to_list(listarray) == to_list(model)
    assert to_list(
        listarray[
            ak.highlevel.Array(
                [[3, 2, 1, 1, 0], [], [1], [0, 0], [1, 2]], backend="cuda"
            )
        ]
    ) == [[2.2, None, 1.1, 1.1, 0.0], [], [None], [5.5, 5.5], [7.7, None]]


def test_0127_tomask_operation_ByteMaskedArray_jaggedslice2():
    model = ak.highlevel.Array(
        [
            [[0.0, 1.1, None, 2.2], [], [3.3, None, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, None, 8.8, 9.9]],
        ],
        backend="cuda",
    )
    assert to_list(
        model[
            ak.highlevel.Array(
                [[[3, 2, 1, 1, 0], [], [1]], [], [[0, 0]], [[1, 2]]], backend="cuda"
            )
        ]
    ) == [[[2.2, None, 1.1, 1.1, 0.0], [], [None]], [], [[5.5, 5.5]], [[7.7, None]]]

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 999, 2.2, 3.3, 123, 4.4, 5.5, 6.6, 7.7, 321, 8.8, 9.9])
    )
    mask = ak.index.Index8(
        np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    )
    maskedarray = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    offsets = ak.index.Index64(np.array([0, 4, 4, 7, 8, 13], dtype=np.int64))
    sublistarray = ak.contents.ListOffsetArray(offsets, maskedarray)
    offsets2 = ak.index.Index64(np.array([0, 3, 3, 4, 5], dtype=np.int64))
    listarray = ak.highlevel.Array(
        ak.contents.ListOffsetArray(offsets2, sublistarray), backend="cuda"
    )
    assert to_list(listarray) == to_list(model)
    assert to_list(
        listarray[
            ak.highlevel.Array(
                [[[3, 2, 1, 1, 0], [], [1]], [], [[0, 0]], [[1, 2]]], backend="cuda"
            )
        ]
    ) == [[[2.2, None, 1.1, 1.1, 0.0], [], [None]], [], [[5.5, 5.5]], [[7.7, None]]]


def test_0127_tomask_operation_ByteMaskedArray_jaggedslice3():
    model = ak.highlevel.Array(
        [
            [[[0.0, 1.1, None, 2.2], [], [3.3, None, 4.4]], []],
            [[[5.5]], [[6.6, 7.7, None, 8.8, 9.9]]],
        ],
        backend="cuda",
    )
    assert to_list(
        model[
            ak.highlevel.Array(
                [[[[3, 2, 1, 1, 0], [], [1]], []], [[[0, 0]], [[1, 2]]]], backend="cuda"
            )
        ]
    ) == [[[[2.2, None, 1.1, 1.1, 0.0], [], [None]], []], [[[5.5, 5.5]], [[7.7, None]]]]

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 999, 2.2, 3.3, 123, 4.4, 5.5, 6.6, 7.7, 321, 8.8, 9.9])
    )
    mask = ak.index.Index8(
        np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    )
    maskedarray = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    offsets = ak.index.Index64(np.array([0, 4, 4, 7, 8, 13], dtype=np.int64))
    subsublistarray = ak.contents.ListOffsetArray(offsets, maskedarray)
    offsets2 = ak.index.Index64(np.array([0, 3, 3, 4, 5], dtype=np.int64))
    sublistarray = ak.contents.ListOffsetArray(offsets2, subsublistarray)
    offsets3 = ak.index.Index64(np.array([0, 2, 4], dtype=np.int64))
    listarray = ak.highlevel.Array(
        ak.contents.ListOffsetArray(offsets3, sublistarray), backend="cuda"
    )
    assert to_list(listarray) == to_list(model)
    assert to_list(
        listarray[
            ak.highlevel.Array(
                [[[[3, 2, 1, 1, 0], [], [1]], []], [[[0, 0]], [[1, 2]]]], backend="cuda"
            )
        ]
    ) == [[[[2.2, None, 1.1, 1.1, 0.0], [], [None]], []], [[[5.5, 5.5]], [[7.7, None]]]]


def test_0127_tomask_operation():
    array = ak.highlevel.Array(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], backend="cuda"
    )
    mask1 = ak.highlevel.Array([True, True, False, False, True], backend="cuda")
    assert to_list(array[mask1]) == [[0.0, 1.1, 2.2], [], [6.6, 7.7, 8.8, 9.9]]
    assert to_list(ak.operations.mask(array, mask1)) == [
        [0.0, 1.1, 2.2],
        [],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]

    mask2 = ak.highlevel.Array(
        [[False, True, False], [], [True, True], [False], [True, False, False, True]],
        backend="cuda",
    )
    assert to_list(array[mask2]) == [[1.1], [], [3.3, 4.4], [], [6.6, 9.9]]
    assert to_list(ak.operations.mask(array, mask2)) == [
        [None, 1.1, None],
        [],
        [3.3, 4.4],
        [None],
        [6.6, None, None, 9.9],
    ]


def test_simple_slice_cpu():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]])
    out = arr[:, 0]
    expected = [1, 0, 4]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )


def test_simple_slice_gpu():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]], backend="cuda")
    out = arr[:, 0]
    expected = [1, 0, 4]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )


def test_simple_slice_cpu1():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]])
    out = arr[:, 1:]
    expected = [[2, 3], [], [5]]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )


def test_simple_slice_gpu1():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]], backend="cuda")
    out = arr[:, 1:]
    expected = [[2, 3], [], [5]]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )


def test_simple_slice_cpu2():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]])
    out = arr[:, :1]
    expected = [[1], [0], [4]]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )


def test_simple_slice_gpu2():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]], backend="cuda")
    out = arr[:, :1]
    expected = [[1], [0], [4]]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )


def test_simple_slice_cpu3():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]])
    out = arr[:, 1::2]
    expected = [[2], [], [5]]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )


def test_simple_slice_gpu3():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]], backend="cuda")
    out = arr[:, 1::2]
    expected = [[2], [], [5]]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )


def test_simple_slice_cpu4():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]])
    out = arr[:, ::-1]
    expected = [[3, 2, 1], [0], [5, 4]]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )


def test_simple_slice_gpu4():
    arr = ak.Array([[1, 2, 3], [0], [4, 5]], backend="cuda")
    out = arr[:, ::-1]
    expected = [[3, 2, 1], [0], [5, 4]]
    result = out.tolist()
    cp.testing.assert_array_list_equal(
        result,
        expected,
        err_msg=f"Slice of [[1, 2, 3], [0], [4, 5]] should be {expected}, but got {result}",
    )
