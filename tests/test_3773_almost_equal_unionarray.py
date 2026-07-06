# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_original_issue():
    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [
            ak.forms.NumpyForm("int64", form_key="numpy1"),
            ak.forms.NumpyForm("int64", form_key="numpy2"),
        ],
        form_key="union",
    )
    buffers = {
        "union-tags": np.array([1, 1], dtype=np.int8),
        "union-index": np.array([0, 1], dtype=np.int64),
        "numpy1-data": np.array([1, 2], dtype=np.int64),
        "numpy2-data": np.array([3, 4], dtype=np.int64),
    }
    array = ak.from_buffers(form, 2, buffers)
    assert ak.almost_equal(array, array)
    assert ak.array_equal(array, array)

    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [
            ak.forms.IndexedOptionForm("i64", ak.forms.EmptyForm(), form_key="nones"),
            ak.forms.UnmaskedForm(ak.forms.NumpyForm("int64", form_key="numpy")),
        ],
        form_key="union",
    )
    virtual_buffers = {
        "union-tags": lambda: np.array([1, 1], dtype=np.int8),
        "union-index": lambda: np.array([0, 1], dtype=np.int64),
        "numpy-data": lambda: np.array([1, 2], dtype=np.int64),
        "nones-index": lambda: np.array([-1], dtype=np.int64),
    }
    eager_buffers = {k: v() for k, v in virtual_buffers.items()}
    eager_array = ak.from_buffers(form, 2, eager_buffers)
    virtual_array = ak.from_buffers(form, 2, virtual_buffers)
    assert ak.almost_equal(eager_array, virtual_array)
    assert ak.array_equal(eager_array, virtual_array)

    from awkward._nplikes.numpy import Numpy
    from awkward._nplikes.virtual import VirtualNDArray

    nplike = Numpy.instance()
    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [
            ak.forms.IndexedOptionForm("i64", ak.forms.EmptyForm(), form_key="nones"),
            ak.forms.UnmaskedForm(ak.forms.NumpyForm("int64", form_key="numpy")),
        ],
        form_key="union",
    )
    virtual_buffers = {
        "union-tags": VirtualNDArray(
            nplike,
            dtype=np.int8,
            shape=(2,),
            generator=lambda: np.array([1, 1], dtype=np.int8),
        ),
        "union-index": VirtualNDArray(
            nplike,
            dtype=np.int64,
            shape=(2,),
            generator=lambda: np.array([0, 1], dtype=np.int64),
        ),
        "numpy-data": VirtualNDArray(
            nplike,
            dtype=np.int64,
            shape=(2,),
            generator=lambda: np.array([1, 2], dtype=np.int64),
        ),
        "nones-index": VirtualNDArray(
            nplike,
            dtype=np.int64,
            shape=(1,),
            generator=lambda: np.array([-1], dtype=np.int64),
        ),
    }
    eager_buffers = {k: v._generator() for k, v in virtual_buffers.items()}
    eager_array = ak.from_buffers(form, 2, eager_buffers)
    virtual_array = ak.from_buffers(form, 2, virtual_buffers)
    assert ak.almost_equal(eager_array, virtual_array)
    assert ak.array_equal(eager_array, virtual_array)


def test_empty_union():
    left = ak.contents.UnionArray(
        ak.index.Index8([]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([]),
        ak.index.Index64([0, 0, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right)
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(right, left)
    assert not ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([2, 3, 0, 1, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right)
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(right, left)
    assert not ak.array_equal(right, left)


def test_size_one_tags():
    left = ak.contents.UnionArray(
        ak.index.Index8([0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right)
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(right, left)
    assert not ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1]),
        ak.index.Index64([0, 0, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right)
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(right, left)
    assert not ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)


def test_all_tags_used():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([2, 3, 0, 1, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([3, 2, 0, 1, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right)
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(right, left)
    assert not ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([2, 3, 0, 1, 2]),
        ak.index.Index64([0, 0, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right)
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(right, left)
    assert not ak.array_equal(right, left)


def test_unused_tags():
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 2, 1, 1]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([3, 0, 0, 3, 3]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 3, 1, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([3, 0, 1, 3, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 3, 1, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([3, 1, 0, 3, 1]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right)
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(right, left)
    assert not ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 0, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([2, 2, 2, 2, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1, 1, 1]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([3, 3, 3, 3, 3]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 0, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([3, 3, 3, 3, 3]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right)
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(right, left)
    assert not ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1, 1, 1]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 0, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right)
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(right, left)
    assert not ak.array_equal(right, left)


def test_castable_dtypes():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right, dtype_exact=True)
    assert ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert ak.almost_equal(right, left, dtype_exact=True)
    assert ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert not ak.almost_equal(left, right, dtype_exact=False)
    assert not ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert not ak.almost_equal(right, left, dtype_exact=False)
    assert not ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float32)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float32)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right, dtype_exact=True)
    assert ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert ak.almost_equal(right, left, dtype_exact=True)
    assert ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float32)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float32)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right, dtype_exact=True)
    assert ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert ak.almost_equal(right, left, dtype_exact=True)
    assert ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)


def test_one_has_more_types():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([True, False], dtype=np.bool_)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([True, False], dtype=np.bool_)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(right, left)
    assert ak.array_equal(right, left)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([True, False], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([True, False], dtype=np.bool_)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([True, False], dtype=np.bool_)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
        ],
    )

    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1, 2]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([True, False], dtype=np.bool_)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert not ak.almost_equal(left, right, dtype_exact=False)
    assert not ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert not ak.almost_equal(right, left, dtype_exact=False)
    assert not ak.array_equal(right, left, dtype_exact=False)


def test_different_integer_sizes():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int8)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int16)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int16)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int8)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right, dtype_exact=True)
    assert ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert ak.almost_equal(right, left, dtype_exact=True)
    assert ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int8)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int16)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int8)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int16)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)


def test_unsigned_integers():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint8)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint16)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint16)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint8)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right, dtype_exact=True)
    assert ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert ak.almost_equal(right, left, dtype_exact=True)
    assert ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint32)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint32)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right, dtype_exact=True)
    assert ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert ak.almost_equal(right, left, dtype_exact=True)
    assert ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint8)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint32)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint16)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.uint64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)


def test_complex_numbers():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1 + 2j, 3 + 4j], dtype=np.complex64)),
            ak.contents.NumpyArray(np.array([5 + 6j, 7 + 8j], dtype=np.complex128)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([5 + 6j, 7 + 8j], dtype=np.complex128)),
            ak.contents.NumpyArray(np.array([1 + 2j, 3 + 4j], dtype=np.complex64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert ak.almost_equal(left, right, dtype_exact=True)
    assert ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert ak.almost_equal(right, left, dtype_exact=True)
    assert ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1 + 2j, 3 + 4j], dtype=np.complex64)),
            ak.contents.NumpyArray(np.array([5 + 6j, 7 + 8j], dtype=np.complex128)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([5 + 6j, 7 + 8j], dtype=np.complex128)),
            ak.contents.NumpyArray(np.array([1 + 2j, 3 + 4j], dtype=np.complex64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert not ak.almost_equal(left, right, dtype_exact=False)
    assert not ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert not ak.almost_equal(right, left, dtype_exact=False)
    assert not ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float32)),
            ak.contents.NumpyArray(np.array([3.0, 4.0], dtype=np.float64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([3 + 0j, 4 + 0j], dtype=np.complex128)),
            ak.contents.NumpyArray(np.array([1 + 0j, 2 + 0j], dtype=np.complex64)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert not ak.almost_equal(left, right, dtype_exact=False)
    assert not ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert not ak.almost_equal(right, left, dtype_exact=False)
    assert not ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([3 + 0j, 4 + 0j], dtype=np.complex64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([3 + 0j, 4 + 0j], dtype=np.complex128)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float32)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert ak.almost_equal(left, right, dtype_exact=False)
    assert ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert ak.almost_equal(right, left, dtype_exact=False)
    assert ak.array_equal(right, left, dtype_exact=False)

    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([3 + 0j, 4 + 0j], dtype=np.complex64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1]),
        ak.index.Index64([0, 1, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([3 + 0j, 4 + 0j], dtype=np.complex128)),
            ak.contents.NumpyArray(np.array([1.0, 2.0], dtype=np.float32)),
        ],
    )
    assert ak.almost_equal(left, left)
    assert ak.array_equal(left, left)
    assert ak.almost_equal(right, right)
    assert ak.array_equal(right, right)
    assert not ak.almost_equal(left, right, dtype_exact=True)
    assert not ak.array_equal(left, right, dtype_exact=True)
    assert not ak.almost_equal(left, right, dtype_exact=False)
    assert not ak.array_equal(left, right, dtype_exact=False)
    assert not ak.almost_equal(right, left, dtype_exact=True)
    assert not ak.array_equal(right, left, dtype_exact=True)
    assert not ak.almost_equal(right, left, dtype_exact=False)
    assert not ak.array_equal(right, left, dtype_exact=False)
