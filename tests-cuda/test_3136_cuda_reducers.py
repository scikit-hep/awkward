from __future__ import annotations

import cupy.testing as cpt
import numpy as np

import awkward as ak

to_list = ak.operations.to_list


def test_sumprod_types():
    def prod(xs):
        out = 1
        for x in xs:
            out *= x
        return out

    array = np.array([[True, False, False], [True, False, False]])
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    depth1 = ak.to_backend(depth1, "cuda")

    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int8)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    depth1 = ak.to_backend(depth1, "cuda")

    assert (
        np.sum(array, axis=-1).dtype
        == ak.to_numpy(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == ak.to_numpy(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    depth1 = ak.to_backend(depth1, "cuda")

    assert (
        np.sum(array, axis=-1).dtype
        == ak.to_numpy(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == ak.to_numpy(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int16)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    depth1 = ak.to_backend(depth1, "cuda")

    assert (
        np.sum(array, axis=-1).dtype
        == ak.to_numpy(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == ak.to_numpy(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint16)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    depth1 = ak.to_backend(depth1, "cuda")

    assert (
        np.sum(array, axis=-1).dtype
        == ak.to_numpy(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == ak.to_numpy(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    depth1 = ak.to_backend(depth1, "cuda")

    assert (
        np.sum(array, axis=-1).dtype
        == ak.to_numpy(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == ak.to_numpy(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    depth1 = ak.to_backend(depth1, "cuda")

    assert (
        np.sum(array, axis=-1).dtype
        == ak.to_numpy(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == ak.to_numpy(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    depth1 = ak.to_backend(depth1, "cuda")

    assert (
        np.sum(array, axis=-1).dtype
        == ak.to_numpy(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == ak.to_numpy(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint64)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    depth1 = ak.to_backend(depth1, "cuda")

    assert (
        np.sum(array, axis=-1).dtype
        == ak.to_numpy(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == ak.to_numpy(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )


def test_sumprod_types_FIXME():
    array = np.array([[True, False, False], [True, False, False]])
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda")

    assert (
        np.sum(array, axis=-1).dtype
        == ak.to_numpy(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == ak.to_numpy(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )


array = ak.Array(
    [[0, 2, 3.0], [4, 5, 6, 7, 8], [], [9, 8, None], [10, 1], []], backend="cuda"
)


def test_sum():
    cpt.assert_allclose(ak.sum(array, axis=None), 63.0)
    assert ak.almost_equal(
        ak.sum(array, axis=None, keepdims=True),
        ak.to_regular(ak.Array([[63.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.sum(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array([[63.0]], backend="cuda").mask[ak.Array([[True]], backend="cuda")]
        ),
    )
    assert ak.sum(array[2], axis=None, mask_identity=True) is None


def test_prod():
    cpt.assert_allclose(ak.prod(array[1:], axis=None), 4838400.0)
    assert ak.prod(array, axis=None) == 0
    assert ak.almost_equal(
        ak.prod(array, axis=None, keepdims=True),
        ak.to_regular(ak.Array([[0.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.prod(array[1:], axis=None, keepdims=True),
        ak.to_regular(ak.Array([[4838400.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.prod(array[1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array([[4838400.0]], backend="cuda").mask[
                ak.Array([[True]], backend="cuda")
            ]
        ),
    )
    assert ak.prod(array[2], axis=None, mask_identity=True) is None


def test_min():
    cpt.assert_allclose(ak.min(array, axis=None), 0.0)
    assert ak.almost_equal(
        ak.min(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular(ak.Array([[0.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.min(array, axis=None, keepdims=True, initial=-100.0, mask_identity=False),
        ak.to_regular(ak.Array([[-100.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.min(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array([[0.0]], backend="cuda").mask[ak.Array([[True]], backend="cuda")]
        ),
    )
    assert ak.almost_equal(
        ak.min(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array(ak.Array([[np.inf]], backend="cuda")).mask[
                ak.Array([[False]], backend="cuda")
            ]
        ),
    )
    assert ak.min(array[2], axis=None, mask_identity=True) is None


def test_max():
    cpt.assert_allclose(ak.max(array, axis=None), 10.0)
    assert ak.almost_equal(
        ak.max(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular(ak.Array([[10.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.max(array, axis=None, keepdims=True, initial=100.0, mask_identity=False),
        ak.to_regular(ak.Array([[100.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.max(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array([[10.0]], backend="cuda").mask[ak.Array([[True]], backend="cuda")]
        ),
    )
    assert ak.almost_equal(
        ak.max(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array(ak.Array([[np.inf]], backend="cuda")).mask[
                ak.Array([[False]], backend="cuda")
            ]
        ),
    )
    assert ak.max(array[2], axis=None, mask_identity=True) is None


array = ak.Array(
    [[0, 2, 3.0], [4, 5, 6, 7, 8], [], [9, 8, None], [10, 1], []], backend="cuda"
)


def test_count():
    assert ak.count(array, axis=None) == 12
    assert ak.almost_equal(
        ak.count(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular(ak.Array([[12]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.count(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array([[12]], backend="cuda").mask[ak.Array([[True]], backend="cuda")]
        ),
    )
    assert ak.almost_equal(
        ak.count(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array([[0]], backend="cuda").mask[ak.Array([[False]], backend="cuda")]
        ),
    )
    assert ak.count(array[2], axis=None, mask_identity=True) is None
    assert ak.count(array[2], axis=None, mask_identity=False) == 0


def test_count_nonzero():
    assert ak.count_nonzero(array, axis=None) == 11
    assert ak.almost_equal(
        ak.count_nonzero(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular(ak.Array([[11]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.count_nonzero(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array([[11]], backend="cuda").mask[ak.Array([[True]], backend="cuda")]
        ),
    )
    assert ak.almost_equal(
        ak.count_nonzero(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array([[0]], backend="cuda").mask[ak.Array([[False]], backend="cuda")]
        ),
    )
    assert ak.count_nonzero(array[2], axis=None, mask_identity=True) is None
    assert ak.count_nonzero(array[2], axis=None, mask_identity=False) == 0


def test_std_no_mask_axis_none():
    assert ak.almost_equal(
        ak.std(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(
            ak.Array([[0.0]], backend="cuda").mask[ak.Array([[False]], backend="cuda")]
        ),
    )
    assert ak.std(array[2], axis=None, mask_identity=True) is None
