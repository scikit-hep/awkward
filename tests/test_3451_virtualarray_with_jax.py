# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak
from awkward._nplikes.jax import Jax
from awkward._nplikes.virtual import VirtualArray

np = pytest.importorskip("jax.numpy")
ak.jax.register_and_check()


# Create fixtures for common test setup
@pytest.fixture
def numpy_like():
    return Jax.instance()


@pytest.fixture
def simple_array_generator():
    return lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64)


@pytest.fixture
def virtual_array(numpy_like, simple_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=simple_array_generator,
    )


@pytest.fixture
def two_dim_array_generator():
    return lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)


@pytest.fixture
def two_dim_virtual_array(numpy_like, two_dim_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.int64),
        generator=two_dim_array_generator,
    )


@pytest.fixture
def scalar_array_generator():
    return lambda: np.array(42, dtype=np.int64)


@pytest.fixture
def scalar_virtual_array(numpy_like, scalar_array_generator):
    return VirtualArray(
        numpy_like, shape=(), dtype=np.dtype(np.int64), generator=scalar_array_generator
    )


@pytest.fixture
def float_array_generator():
    return lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)


@pytest.fixture
def float_virtual_array(numpy_like, float_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=float_array_generator,
    )


@pytest.fixture
def numpyarray():
    return ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64))


@pytest.fixture
def virtual_numpyarray(float_virtual_array):
    return ak.contents.NumpyArray(float_virtual_array)


@pytest.fixture
def offset_array_generator():
    return lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64)


@pytest.fixture
def virtual_offset_array(numpy_like, offset_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=offset_array_generator,
    )


@pytest.fixture
def listoffsetarray():
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    return ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )


@pytest.fixture
def virtual_listoffsetarray(numpy_like, virtual_offset_array, virtual_content_array):
    return ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offset_array),
        ak.contents.NumpyArray(virtual_content_array),
    )


@pytest.fixture
def starts_array_generator():
    return lambda: np.array([0, 2, 4, 7], dtype=np.int64)


@pytest.fixture
def virtual_starts_array(numpy_like, starts_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=starts_array_generator,
    )


@pytest.fixture
def stops_array_generator():
    return lambda: np.array([2, 4, 7, 10], dtype=np.int64)


@pytest.fixture
def virtual_stops_array(numpy_like, stops_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=stops_array_generator,
    )


@pytest.fixture
def content_array_generator():
    return lambda: np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )


@pytest.fixture
def virtual_content_array(numpy_like, content_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=content_array_generator,
    )


@pytest.fixture
def listarray():
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    return ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )


@pytest.fixture
def virtual_listarray(
    numpy_like, virtual_starts_array, virtual_stops_array, virtual_content_array
):
    return ak.contents.ListArray(
        ak.index.Index(virtual_starts_array),
        ak.index.Index(virtual_stops_array),
        ak.contents.NumpyArray(virtual_content_array),
    )


@pytest.fixture
def offsets_array_generator():
    return lambda: np.array([0, 2, 4, 7, 10, 10], dtype=np.int64)


@pytest.fixture
def virtual_offsets_array(numpy_like, offsets_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=offsets_array_generator,
    )


@pytest.fixture
def x_content_array_generator():
    return lambda: np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )


@pytest.fixture
def virtual_x_content_array(numpy_like, x_content_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=x_content_array_generator,
    )


@pytest.fixture
def y_array_generator():
    return lambda: np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)


@pytest.fixture
def virtual_y_array(numpy_like, y_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=y_array_generator,
    )


@pytest.fixture
def recordarray():
    # Create a regular RecordArray with ListOffsetArray and NumpyArray fields
    offsets = np.array([0, 2, 4, 7, 10, 10], dtype=np.int64)
    x_content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    y_content = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )

    y_field = ak.contents.NumpyArray(y_content)

    return ak.contents.RecordArray([x_field, y_field], ["x", "y"])


@pytest.fixture
def virtual_recordarray(
    numpy_like, virtual_offsets_array, virtual_x_content_array, virtual_y_array
):
    # Create a RecordArray with virtual components
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets_array),
        ak.contents.NumpyArray(virtual_x_content_array),
    )

    y_field = ak.contents.NumpyArray(virtual_y_array)

    return ak.contents.RecordArray([x_field, y_field], ["x", "y"])


def test_numpyarray_to_list(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.to_list(virtual_numpyarray) == ak.to_list(numpyarray)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_to_json(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.to_json(virtual_numpyarray) == ak.to_json(numpyarray)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_to_numpy(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.all(
        ak.to_numpy(ak.materialize(virtual_numpyarray))
        == ak.to_numpy(ak.materialize(numpyarray))
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_to_buffers(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    out1 = ak.to_buffers(numpyarray)
    out2 = ak.to_buffers(virtual_numpyarray)
    # form
    assert out1[0] == out2[0]
    # length
    assert out1[1] == out2[1]
    # container
    assert out1[2].keys() == out2[2].keys()
    for key in out1[2]:
        assert isinstance(out1[2][key], np.ndarray)
        assert isinstance(out2[2][key], VirtualArray)
        assert ak.all(out1[2][key] == out2[2][key])


def test_numpyarray_is_valid(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.is_valid(virtual_numpyarray) == ak.is_valid(numpyarray)
    assert ak.validity_error(virtual_numpyarray) == ak.validity_error(numpyarray)


def test_numpyarray_zip(numpyarray, virtual_numpyarray):
    zip1 = ak.zip({"x": numpyarray, "y": numpyarray})
    zip2 = ak.zip({"x": virtual_numpyarray, "y": virtual_numpyarray})
    assert not zip2.layout.is_any_materialized
    assert zip1.fields == zip2.fields
    assert ak.array_equal(ak.materialize(zip2), zip1)


def test_numpyarray_unzip(numpyarray, virtual_numpyarray):
    zip1 = ak.zip({"x": numpyarray, "y": numpyarray})
    zip2 = ak.zip({"x": virtual_numpyarray, "y": virtual_numpyarray})
    assert not zip2.layout.is_any_materialized
    unzip1 = ak.unzip(zip1)
    unzip2 = ak.unzip(zip2)
    assert not unzip2[0].layout.is_any_materialized
    assert not unzip2[1].layout.is_any_materialized
    assert ak.array_equal(ak.materialize(unzip2[0]), unzip1[0])
    assert ak.array_equal(ak.materialize(unzip2[1]), unzip1[1])


def test_numpyarray_concatenate(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.concatenate([numpyarray, numpyarray]),
        ak.concatenate([virtual_numpyarray, virtual_numpyarray]),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_where(numpyarray, virtual_numpyarray):
    numpyarray = ak.Array(numpyarray)
    virtual_numpyarray = ak.Array(virtual_numpyarray)
    assert not virtual_numpyarray.layout.is_any_materialized
    assert ak.array_equal(
        ak.where(numpyarray > 2, numpyarray, numpyarray + 100),
        ak.where(virtual_numpyarray > 2, virtual_numpyarray, virtual_numpyarray + 100),
    )
    assert virtual_numpyarray.layout.is_any_materialized
    assert virtual_numpyarray.layout.is_all_materialized


def test_numpyarray_unflatten(numpyarray, virtual_numpyarray):
    numpyarray = ak.Array(numpyarray)
    virtual_numpyarray = ak.Array(virtual_numpyarray)
    assert not virtual_numpyarray.layout.is_any_materialized
    assert ak.array_equal(
        ak.unflatten(numpyarray, ak.to_backend([2, 3], "jax")),
        ak.unflatten(virtual_numpyarray, ak.to_backend([2, 3], "jax")),
    )
    assert virtual_numpyarray.layout.is_any_materialized
    assert virtual_numpyarray.layout.is_all_materialized


def test_numpyarray_num(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.num(virtual_numpyarray, axis=0) == ak.num(numpyarray, axis=0)
    assert not virtual_numpyarray.is_any_materialized


def test_numpyarray_count(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.count(virtual_numpyarray, axis=0) == ak.count(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numppyarray_count_nonzero(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.count_nonzero(virtual_numpyarray, axis=0) == ak.count_nonzero(
        numpyarray, axis=0
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_sum(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.sum(virtual_numpyarray, axis=0) == ak.sum(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nansum(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nansum(virtual_numpyarray, axis=0) == ak.nansum(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_prod(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.prod(virtual_numpyarray, axis=0) == ak.prod(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanprod(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanprod(virtual_numpyarray, axis=0) == ak.nanprod(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_any(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.any(virtual_numpyarray, axis=0) == ak.any(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_all(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.all(virtual_numpyarray, axis=0) == ak.all(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_min(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.min(virtual_numpyarray, axis=0) == ak.min(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanmin(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanmin(virtual_numpyarray, axis=0) == ak.nanmin(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_max(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.max(virtual_numpyarray, axis=0) == ak.max(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanmax(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanmax(virtual_numpyarray, axis=0) == ak.nanmax(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argmin(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.argmin(virtual_numpyarray, axis=0) == ak.argmin(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanargmin(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanargmin(virtual_numpyarray, axis=0) == ak.nanargmin(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argmax(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.argmax(virtual_numpyarray, axis=0) == ak.argmax(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanargmax(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanargmax(virtual_numpyarray, axis=0) == ak.nanargmax(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_sort(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.sort(virtual_numpyarray, axis=0),
        ak.sort(numpyarray, axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argsort(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.argsort(virtual_numpyarray, axis=0),
        ak.argsort(numpyarray, axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_is_none(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.all(ak.is_none(virtual_numpyarray) == ak.is_none(numpyarray))
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_drop_none(numpy_like):
    array = ak.to_backend(ak.Array([1, None, 2, 3, None, 4, 5]), "jax", highlevel=False)
    virtual_index = ak.index.Index(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, -1, 1, 2, -1, 3, 4], dtype=np.int64),
        )
    )
    virtual_content = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    )
    virtual_array = ak.contents.IndexedOptionArray(virtual_index, virtual_content)
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.drop_none(virtual_array), ak.drop_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpy_array_pad_none(numpy_like):
    array = ak.to_backend(ak.Array([1, None, 2, 3, None, 4, 5]), "jax", highlevel=False)
    virtual_index = ak.index.Index(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, -1, 1, 2, -1, 3, 4], dtype=np.int64),
        )
    )
    virtual_content = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    )
    virtual_array = ak.contents.IndexedOptionArray(virtual_index, virtual_content)
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(
        ak.pad_none(virtual_array, 10, axis=0), ak.pad_none(array, 10, axis=0)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_fill_none(numpy_like):
    array = ak.to_backend(ak.Array([1, None, 2, 3, None, 4, 5]), "jax", highlevel=False)
    virtual_index = ak.index.Index(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, -1, 1, 2, -1, 3, 4], dtype=np.int64),
        )
    )
    virtual_content = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    )
    virtual_array = ak.contents.IndexedOptionArray(virtual_index, virtual_content)
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.fill_none(virtual_array, 100), ak.fill_none(array, 100))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_firsts(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.firsts(virtual_numpyarray, axis=0),
        ak.firsts(numpyarray, axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_singletons(numpy_like):
    array = ak.to_backend(ak.Array([1, 2, 3, 4, 5]), "jax", highlevel=False)
    virtual_index = ak.index.Index(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, 1, 2, 3, 4], dtype=np.int64),
        )
    )
    virtual_content = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    )
    virtual_array = ak.contents.IndexedArray(virtual_index, virtual_content)
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.singletons(virtual_array), ak.singletons(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_to_regular(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.to_regular(virtual_numpyarray, axis=0), ak.to_regular(numpyarray, axis=0)
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_broadcast_arrays(virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    out = ak.broadcast_arrays(5, virtual_numpyarray)
    assert ak.to_list(out[0]) == [5, 5, 5, 5, 5]
    assert not virtual_numpyarray.is_any_materialized
    assert not out[1].layout.is_any_materialized
    assert ak.to_list(out[1]) == ak.to_list(virtual_numpyarray)


def test_numpyarray_cartesian(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.cartesian([numpyarray, numpyarray], axis=0),
        ak.cartesian([virtual_numpyarray, virtual_numpyarray], axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argcartesian(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.argcartesian([numpyarray, numpyarray], axis=0),
        ak.argcartesian([virtual_numpyarray, virtual_numpyarray], axis=0),
    )
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_combinations(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.combinations(numpyarray, 2, axis=0),
        ak.combinations(virtual_numpyarray, 2, axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argcombinations(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.argcombinations(numpyarray, 2, axis=0),
        ak.argcombinations(virtual_numpyarray, 2, axis=0),
    )
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_nan_to_none(numpy_like):
    array = ak.to_backend(
        ak.Array([1, np.nan, 2, 3, np.nan, 4, 5]), "jax", highlevel=False
    )
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array(
                [1, np.nan, 2, 3, np.nan, 4, 5], dtype=np.float64
            ),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_none(virtual_array), ak.nan_to_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_nan_to_num(numpy_like):
    array = ak.to_backend(
        ak.Array([1, np.nan, 2, 3, np.nan, 4, 5]), "jax", highlevel=False
    )
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array(
                [1, np.nan, 2, 3, np.nan, 4, 5], dtype=np.float64
            ),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_num(virtual_array), ak.nan_to_num(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_local_index(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.local_index(virtual_numpyarray, axis=0), ak.local_index(numpyarray, axis=0)
    )
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_run_lengths(numpy_like):
    array = ak.to_backend(ak.Array([1, 1, 2, 3, 3, 3, 4, 5]), "jax", highlevel=False)
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(8,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 1, 2, 3, 3, 3, 4, 5], dtype=np.int64),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.run_lengths(virtual_array), ak.run_lengths(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_round(numpy_like):
    array = ak.to_backend(
        ak.Array([1.234, 2.567, 3.499, 4.501]), "jax", highlevel=False
    )
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(4,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.round(virtual_array), ak.round(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_isclose(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.isclose(virtual_numpyarray, numpyarray, rtol=1e-5, atol=1e-8),
        ak.isclose(numpyarray, numpyarray, rtol=1e-5, atol=1e-8),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_almost_equal(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.almost_equal(virtual_numpyarray, numpyarray),
        ak.almost_equal(numpyarray, numpyarray),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_real(numpy_like):
    array = ak.to_backend(ak.Array([1 + 2j, 3 + 4j, 5 + 6j]), "jax", highlevel=False)
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(3,),
            dtype=np.dtype(np.complex128),
            generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.real(virtual_array), ak.real(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_imag(numpy_like):
    array = ak.to_backend(ak.Array([1 + 2j, 3 + 4j, 5 + 6j]), "jax", highlevel=False)
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(3,),
            dtype=np.dtype(np.complex128),
            generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.imag(virtual_array), ak.imag(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_angle(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.angle(virtual_numpyarray, deg=True),
        ak.angle(numpyarray, deg=True),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_zeros_like(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(ak.zeros_like(virtual_numpyarray), ak.zeros_like(numpyarray))
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_ones_like(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(ak.ones_like(virtual_numpyarray), ak.ones_like(numpyarray))
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_full_like(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.full_like(virtual_numpyarray, 100), ak.full_like(numpyarray, 100)
    )
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_listoffsetarray_to_list(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.to_list(virtual_listoffsetarray) == ak.to_list(listoffsetarray)
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_to_json(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.to_json(virtual_listoffsetarray) == ak.to_json(listoffsetarray)
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_to_numpy(listoffsetarray, virtual_listoffsetarray):
    # ListOffsetArray can't be directly converted to numpy for non-rectangular data
    # Test with flattened data instead
    assert not virtual_listoffsetarray.is_any_materialized
    flat_listoffsetarray = ak.flatten(listoffsetarray)
    flat_virtual_listoffsetarray = ak.flatten(virtual_listoffsetarray)
    assert ak.all(
        ak.to_numpy(ak.materialize(flat_virtual_listoffsetarray))
        == ak.to_numpy(ak.materialize(flat_listoffsetarray))
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_to_buffers(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    out1 = ak.to_buffers(listoffsetarray)
    out2 = ak.to_buffers(virtual_listoffsetarray)
    # form
    assert out1[0] == out2[0]
    # length
    assert out1[1] == out2[1]
    # container
    assert set(out1[2].keys()) == set(out2[2].keys())
    for key in out1[2]:
        if isinstance(out2[2][key], VirtualArray):
            assert not out2[2][key].is_materialized
            assert ak.all(out1[2][key] == out2[2][key])
            assert out2[2][key].is_materialized
        else:
            assert ak.all(out1[2][key] == out2[2][key])


def test_listoffsetarray_is_valid(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.is_valid(virtual_listoffsetarray) == ak.is_valid(listoffsetarray)
    assert ak.validity_error(virtual_listoffsetarray) == ak.validity_error(
        listoffsetarray
    )


def test_listoffsetarray_zip(listoffsetarray, virtual_listoffsetarray):
    zip1 = ak.zip({"x": listoffsetarray, "y": listoffsetarray})
    zip2 = ak.zip({"x": virtual_listoffsetarray, "y": virtual_listoffsetarray})
    assert zip2.layout.is_any_materialized
    assert not zip2.layout.is_all_materialized
    assert zip1.fields == zip2.fields
    assert ak.array_equal(ak.materialize(zip2), zip1)


def test_listoffsetarray_unzip(listoffsetarray, virtual_listoffsetarray):
    zip1 = ak.zip({"x": listoffsetarray, "y": listoffsetarray})
    zip2 = ak.zip({"x": virtual_listoffsetarray, "y": virtual_listoffsetarray})
    assert zip2.layout.is_any_materialized
    unzip1 = ak.unzip(zip1)
    unzip2 = ak.unzip(zip2)
    assert unzip2[0].layout.is_any_materialized
    assert unzip2[1].layout.is_any_materialized
    assert ak.array_equal(ak.materialize(unzip2[0]), unzip1[0])
    assert ak.array_equal(ak.materialize(unzip2[1]), unzip1[1])


def test_listoffsetarray_concatenate(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.concatenate([listoffsetarray, listoffsetarray]),
        ak.concatenate([virtual_listoffsetarray, virtual_listoffsetarray]),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_where(listoffsetarray, virtual_listoffsetarray):
    # We need to use ak.Array for the where function
    list_array = ak.Array(listoffsetarray)
    virtual_list_array = ak.Array(virtual_listoffsetarray)

    assert not virtual_list_array.layout.is_any_materialized

    # For nested arrays, we need a compatible mask
    # Create a mask that's True for some elements in each list
    mask = ak.to_backend(
        ak.Array(
            [[True, False], [False, True], [True, False, True], [False, True, True]]
        ),
        "jax",
    )

    # Test with a conditional mask
    result1 = ak.where(mask, list_array, 999)
    result2 = ak.where(mask, virtual_list_array, 999)

    assert ak.array_equal(result1, result2)
    assert virtual_list_array.layout.is_any_materialized
    assert virtual_list_array.layout.is_all_materialized


def test_listoffsetarray_flaten(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.flatten(virtual_listoffsetarray),
        ak.flatten(listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_unflatten(listoffsetarray, virtual_listoffsetarray):
    # First flatten the arrays
    flat_list = ak.flatten(listoffsetarray)
    flat_virtual = ak.flatten(virtual_listoffsetarray)

    # Define counts for unflattening
    counts = np.array([2, 2, 3, 3])

    assert virtual_listoffsetarray.is_any_materialized

    # Unflatten and compare
    result1 = ak.unflatten(flat_list, counts)
    result2 = ak.unflatten(flat_virtual, counts)

    assert ak.array_equal(result1, result2)
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_num(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(ak.num(virtual_listoffsetarray) == ak.num(listoffsetarray))
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_count(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.count(virtual_listoffsetarray, axis=1) == ak.count(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_count_nonzero(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.count_nonzero(virtual_listoffsetarray, axis=1)
        == ak.count_nonzero(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_sum(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.sum(virtual_listoffsetarray, axis=1) == ak.sum(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nansum(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.nansum(virtual_listoffsetarray, axis=1) == ak.nansum(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_prod(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.prod(virtual_listoffsetarray, axis=1) == ak.prod(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanprod(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.nanprod(virtual_listoffsetarray, axis=1)
        == ak.nanprod(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_any(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.any(virtual_listoffsetarray, axis=1) == ak.any(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_all(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.all(virtual_listoffsetarray, axis=1) == ak.all(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_min(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.min(virtual_listoffsetarray, axis=1) == ak.min(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanmin(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.nanmin(virtual_listoffsetarray, axis=1) == ak.nanmin(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_max(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.max(virtual_listoffsetarray, axis=1) == ak.max(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanmax(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.nanmax(virtual_listoffsetarray, axis=1) == ak.nanmax(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_argmin(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.argmin(virtual_listoffsetarray), ak.argmin(listoffsetarray)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanargmin(numpy_like):
    # Create arrays with NaN values to test nanargmin
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )

    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    # Axis=1 to test within each nested list
    result1 = ak.nanargmin(array, axis=1)
    result2 = ak.nanargmin(virtual_array, axis=1)
    assert ak.array_equal(result1, result2)
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_argmax(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.argmax(virtual_listoffsetarray), ak.argmax(listoffsetarray)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanargmax(numpy_like):
    # Create arrays with NaN values to test nanargmax
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )

    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    # Axis=1 to test within each nested list
    result1 = ak.nanargmax(array, axis=1)
    result2 = ak.nanargmax(virtual_array, axis=1)
    assert ak.array_equal(result1, result2)
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_sort(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.sort(virtual_listoffsetarray),
        ak.sort(listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_argsort(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.argsort(virtual_listoffsetarray),
        ak.argsort(listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_is_none(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(ak.is_none(virtual_listoffsetarray) == ak.is_none(listoffsetarray))
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_drop_none(numpy_like):
    # Create a ListOffsetArray with some None values
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)

    # Create an IndexedOptionArray for the content that has None values
    index_data = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    content_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    index = ak.index.Index(index_data)
    content = ak.contents.NumpyArray(content_data)
    indexed_content = ak.contents.IndexedOptionArray(index, content)

    array = ak.contents.ListOffsetArray(ak.index.Index(offsets), indexed_content)

    # Create virtual versions
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_indexed_content = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_index), ak.contents.NumpyArray(virtual_content)
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), virtual_indexed_content
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.drop_none(virtual_array), ak.drop_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_pad_none(numpy_like):
    # Create a regular ListOffsetArray
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    # Pad each list to length 5
    assert ak.array_equal(
        ak.pad_none(virtual_array, 5, axis=1), ak.pad_none(array, 5, axis=1)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_fill_none(numpy_like):
    # Create a ListOffsetArray with some None values
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)

    # Create an IndexedOptionArray for the content that has None values
    index_data = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    content_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    index = ak.index.Index(index_data)
    content = ak.contents.NumpyArray(content_data)
    indexed_content = ak.contents.IndexedOptionArray(index, content)

    array = ak.contents.ListOffsetArray(ak.index.Index(offsets), indexed_content)

    # Create virtual versions
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_indexed_content = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_index), ak.contents.NumpyArray(virtual_content)
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), virtual_indexed_content
    )

    assert not virtual_array.is_any_materialized
    # Fill None values with 999.0
    assert ak.array_equal(
        ak.fill_none(virtual_array, 999.0), ak.fill_none(array, 999.0)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_firsts(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.firsts(virtual_listoffsetarray),
        ak.firsts(listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_singletons(numpy_like):
    # Create a regular array to test
    offsets = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    content = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 1, 2, 3, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.singletons(virtual_array), ak.singletons(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_to_regular(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.to_regular(virtual_listoffsetarray, axis=0),
        ak.to_regular(listoffsetarray, axis=0),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_broadcast_arrays(virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    out = ak.broadcast_arrays(5, virtual_listoffsetarray)
    assert virtual_listoffsetarray.is_any_materialized
    assert out[1].layout.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized
    assert not out[1].layout.is_all_materialized
    assert ak.to_list(out[1]) == ak.to_list(virtual_listoffsetarray)


def test_listoffsetarray_cartesian(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.cartesian([listoffsetarray, listoffsetarray]),
        ak.cartesian([virtual_listoffsetarray, virtual_listoffsetarray]),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_argcartesian(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.argcartesian([listoffsetarray, listoffsetarray]),
        ak.argcartesian([virtual_listoffsetarray, virtual_listoffsetarray]),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_combinations(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    # Need to use axis=1 for nested structures
    assert ak.array_equal(
        ak.combinations(virtual_listoffsetarray, 2, axis=1),
        ak.combinations(listoffsetarray, 2, axis=1),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_argcombinations(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    # Need to use axis=1 for nested structures
    assert ak.array_equal(
        ak.argcombinations(virtual_listoffsetarray, 2, axis=1),
        ak.argcombinations(listoffsetarray, 2, axis=1),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nan_to_none(numpy_like):
    # Create a ListOffsetArray with NaN values
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_none(virtual_array), ak.nan_to_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_nan_to_num(numpy_like):
    # Create a ListOffsetArray with NaN values
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_num(virtual_array), ak.nan_to_num(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_local_index(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    # For ListOffsetArray, we should use axis=1 to get indices within each list
    assert ak.array_equal(
        ak.local_index(virtual_listoffsetarray, axis=1),
        ak.local_index(listoffsetarray, axis=1),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_run_lengths(numpy_like):
    # Create a ListOffsetArray with repeated values for run_lengths test
    offsets = np.array([0, 3, 6], dtype=np.int64)
    content = np.array([1, 1, 2, 3, 3, 3], dtype=np.int64)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 3, 6], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 1, 2, 3, 3, 3], dtype=np.int64),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    # Need to use axis=1 to check run lengths within each list
    assert ak.array_equal(ak.run_lengths(virtual_array), ak.run_lengths(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_round(numpy_like):
    # Create a ListOffsetArray with float values for rounding
    offsets = np.array([0, 2, 4], dtype=np.int64)
    content = np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.round(virtual_array), ak.round(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_isclose(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.isclose(virtual_listoffsetarray, listoffsetarray, rtol=1e-5, atol=1e-8),
        ak.isclose(listoffsetarray, listoffsetarray, rtol=1e-5, atol=1e-8),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_almost_equal(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.almost_equal(virtual_listoffsetarray, listoffsetarray),
        ak.almost_equal(listoffsetarray, listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_real(numpy_like):
    # Create a ListOffsetArray with complex values for real test
    offsets = np.array([0, 2, 3], dtype=np.int64)
    content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.real(virtual_array), ak.real(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_imag(numpy_like):
    # Create a ListOffsetArray with complex values for imag test
    offsets = np.array([0, 2, 3], dtype=np.int64)
    content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.imag(virtual_array), ak.imag(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_angle(numpy_like):
    # Create a ListOffsetArray with complex values for angle test
    offsets = np.array([0, 2, 4], dtype=np.int64)
    content = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array(
            [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(
        ak.angle(virtual_array, deg=True),
        ak.angle(array, deg=True),
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_zeros_like(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.zeros_like(virtual_listoffsetarray), ak.zeros_like(listoffsetarray)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_ones_like(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.ones_like(virtual_listoffsetarray), ak.ones_like(listoffsetarray)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_full_like(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.full_like(virtual_listoffsetarray, 100), ak.full_like(listoffsetarray, 100)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


# Additional tests for ListOffsetArray-specific operations


def test_listoffsetarray_flatten(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.flatten(virtual_listoffsetarray, axis=1), ak.flatten(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_slicing(listoffsetarray, virtual_listoffsetarray):
    # Convert to ak.Array for slicing operations
    list_array = ak.Array(listoffsetarray)
    virtual_list_array = ak.Array(virtual_listoffsetarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Test slicing the outer dimension
    assert ak.array_equal(virtual_list_array[1:3], list_array[1:3])

    # Test indexing and then slicing inner dimension
    assert ak.array_equal(virtual_list_array[0][0:2], list_array[0][0:2])

    assert virtual_list_array.layout.is_any_materialized


def test_listoffsetarray_mask_operations(listoffsetarray, virtual_listoffsetarray):
    list_array = ak.Array(listoffsetarray)
    virtual_list_array = ak.Array(virtual_listoffsetarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Create a boolean mask
    mask = ak.to_backend(ak.Array([True, False, True, False]), "jax")

    # Test masking
    assert ak.array_equal(virtual_list_array[mask], list_array[mask])

    assert virtual_list_array.layout.is_any_materialized


def test_listoffsetarray_arithmetics(listoffsetarray, virtual_listoffsetarray):
    list_array = ak.Array(listoffsetarray)
    virtual_list_array = ak.Array(virtual_listoffsetarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Test addition
    assert ak.array_equal(virtual_list_array + 10, list_array + 10)

    # Test multiplication
    assert ak.array_equal(virtual_list_array * 2, list_array * 2)

    # Test division
    assert ak.array_equal(virtual_list_array / 2, list_array / 2)

    assert virtual_list_array.layout.is_any_materialized
    assert virtual_list_array.layout.is_all_materialized


def test_listarray_to_list(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.to_list(virtual_listarray) == ak.to_list(listarray)
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_to_json(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.to_json(virtual_listarray) == ak.to_json(listarray)
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_to_numpy(listarray, virtual_listarray):
    # ListArray can't be directly converted to numpy for non-rectangular data
    # Test with flattened data instead
    assert not virtual_listarray.is_any_materialized
    flat_listarray = ak.flatten(listarray)
    flat_virtual_listarray = ak.flatten(virtual_listarray)
    assert ak.all(
        ak.to_numpy(ak.materialize(flat_virtual_listarray))
        == ak.to_numpy(ak.materialize(flat_listarray))
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_to_buffers(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    out1 = ak.to_buffers(listarray)
    out2 = ak.to_buffers(virtual_listarray)
    # form
    assert out1[0] == out2[0]
    # length
    assert out1[1] == out2[1]
    # container
    assert set(out1[2].keys()) == set(out2[2].keys())
    for key in out1[2]:
        if isinstance(out2[2][key], VirtualArray):
            assert not out2[2][key].is_materialized
            assert ak.all(out1[2][key] == out2[2][key])
            assert out2[2][key].is_materialized
        else:
            assert ak.all(out1[2][key] == out2[2][key])


def test_listarray_is_valid(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.is_valid(virtual_listarray) == ak.is_valid(listarray)
    assert ak.validity_error(virtual_listarray) == ak.validity_error(listarray)


def test_listarray_zip(listarray, virtual_listarray):
    zip1 = ak.zip({"x": listarray, "y": listarray})
    zip2 = ak.zip({"x": virtual_listarray, "y": virtual_listarray})
    assert zip2.layout.is_any_materialized
    assert not zip2.layout.is_all_materialized
    assert zip1.fields == zip2.fields
    assert ak.array_equal(ak.materialize(zip2), zip1)


def test_listarray_unzip(listarray, virtual_listarray):
    zip1 = ak.zip({"x": listarray, "y": listarray})
    zip2 = ak.zip({"x": virtual_listarray, "y": virtual_listarray})
    assert zip2.layout.is_any_materialized
    unzip1 = ak.unzip(zip1)
    unzip2 = ak.unzip(zip2)
    assert unzip2[0].layout.is_any_materialized
    assert unzip2[1].layout.is_any_materialized
    assert ak.array_equal(ak.materialize(unzip2[0]), unzip1[0])
    assert ak.array_equal(ak.materialize(unzip2[1]), unzip1[1])


def test_listarray_concatenate(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.concatenate([listarray, listarray]),
        ak.concatenate([virtual_listarray, virtual_listarray]),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_where(listarray, virtual_listarray):
    # We need to use ak.Array for the where function
    list_array = ak.Array(listarray)
    virtual_list_array = ak.Array(virtual_listarray)

    assert not virtual_list_array.layout.is_any_materialized

    # For nested arrays, we need a compatible mask
    # Create a mask that's True for some elements in each list
    mask = ak.to_backend(
        ak.Array(
            [[True, False], [False, True], [True, False, True], [False, True, True]]
        ),
        "jax",
    )

    # Test with a conditional mask
    result1 = ak.where(mask, list_array, 999)
    result2 = ak.where(mask, virtual_list_array, 999)

    assert ak.array_equal(result1, result2)
    assert virtual_list_array.layout.is_any_materialized
    assert virtual_list_array.layout.is_all_materialized


def test_listarray_flatten(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.flatten(virtual_listarray, axis=1), ak.flatten(listarray, axis=1)
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_unflatten(listarray, virtual_listarray):
    # First flatten the arrays
    flat_list = ak.flatten(listarray)
    flat_virtual = ak.flatten(virtual_listarray)

    # Define counts for unflattening
    counts = np.array([2, 2, 3, 3])

    assert virtual_listarray.is_any_materialized

    # Unflatten and compare
    result1 = ak.unflatten(flat_list, counts)
    result2 = ak.unflatten(flat_virtual, counts)

    assert ak.array_equal(result1, result2)
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_num(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.num(virtual_listarray) == ak.num(listarray))
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_count(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.count(virtual_listarray, axis=1) == ak.count(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_count_nonzero(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(
        ak.count_nonzero(virtual_listarray, axis=1)
        == ak.count_nonzero(listarray, axis=1)
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_sum(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.sum(virtual_listarray, axis=1) == ak.sum(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nansum(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.nansum(virtual_listarray, axis=1) == ak.nansum(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_prod(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.prod(virtual_listarray, axis=1) == ak.prod(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanprod(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(
        ak.nanprod(virtual_listarray, axis=1) == ak.nanprod(listarray, axis=1)
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_any(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.any(virtual_listarray, axis=1) == ak.any(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_all(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.all(virtual_listarray, axis=1) == ak.all(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_min(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.min(virtual_listarray, axis=1) == ak.min(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanmin(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.nanmin(virtual_listarray, axis=1) == ak.nanmin(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_max(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.max(virtual_listarray, axis=1) == ak.max(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanmax(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.nanmax(virtual_listarray, axis=1) == ak.nanmax(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_argmin(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(ak.argmin(virtual_listarray), ak.argmin(listarray))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanargmin(numpy_like):
    # Create arrays with NaN values to test nanargmin
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )

    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    # Axis=1 to test within each nested list
    result1 = ak.nanargmin(array, axis=1)
    result2 = ak.nanargmin(virtual_array, axis=1)
    assert ak.array_equal(result1, result2)
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_argmax(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(ak.argmax(virtual_listarray), ak.argmax(listarray))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanargmax(numpy_like):
    # Create arrays with NaN values to test nanargmax
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )

    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    # Axis=1 to test within each nested list
    result1 = ak.nanargmax(array, axis=1)
    result2 = ak.nanargmax(virtual_array, axis=1)
    assert ak.array_equal(result1, result2)
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_sort(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.sort(virtual_listarray),
        ak.sort(listarray),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_argsort(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.argsort(virtual_listarray),
        ak.argsort(listarray),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_is_none(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.is_none(virtual_listarray) == ak.is_none(listarray))
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_drop_none(numpy_like):
    # Create a ListArray with some None values
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)

    # Create an IndexedOptionArray for the content that has None values
    index_data = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    content_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    index = ak.index.Index(index_data)
    content = ak.contents.NumpyArray(content_data)
    indexed_content = ak.contents.IndexedOptionArray(index, content)

    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), indexed_content
    )

    # Create virtual versions
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_indexed_content = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_index), ak.contents.NumpyArray(virtual_content)
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        virtual_indexed_content,
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.drop_none(virtual_array), ak.drop_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_pad_none(numpy_like):
    # Create a regular ListArray
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    # Pad each list to length 5
    assert ak.array_equal(
        ak.pad_none(virtual_array, 5, axis=1), ak.pad_none(array, 5, axis=1)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_fill_none(numpy_like):
    # Create a ListArray with some None values
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)

    # Create an IndexedOptionArray for the content that has None values
    index_data = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    content_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    index = ak.index.Index(index_data)
    content = ak.contents.NumpyArray(content_data)
    indexed_content = ak.contents.IndexedOptionArray(index, content)

    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), indexed_content
    )

    # Create virtual versions
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_indexed_content = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_index), ak.contents.NumpyArray(virtual_content)
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        virtual_indexed_content,
    )

    assert not virtual_array.is_any_materialized
    # Fill None values with 999.0
    assert ak.array_equal(
        ak.fill_none(virtual_array, 999.0), ak.fill_none(array, 999.0)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_firsts(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.firsts(virtual_listarray),
        ak.firsts(listarray),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_singletons(numpy_like):
    # Create a regular array to test
    starts = np.array([0, 1, 2, 3], dtype=np.int64)
    stops = np.array([1, 2, 3, 4], dtype=np.int64)
    content = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 1, 2, 3], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 2, 3, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.singletons(virtual_array), ak.singletons(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_to_regular(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.to_regular(virtual_listarray, axis=0),
        ak.to_regular(listarray, axis=0),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_broadcast_arrays(virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    out = ak.broadcast_arrays(5, virtual_listarray)
    assert virtual_listarray.is_any_materialized
    assert out[1].layout.is_any_materialized
    assert not virtual_listarray.is_all_materialized
    assert not out[1].layout.is_all_materialized
    assert ak.to_list(out[1]) == ak.to_list(virtual_listarray)


def test_listarray_cartesian(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.cartesian([listarray, listarray]),
        ak.cartesian([virtual_listarray, virtual_listarray]),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_argcartesian(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.argcartesian([listarray, listarray]),
        ak.argcartesian([virtual_listarray, virtual_listarray]),
    )
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_combinations(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    # Need to use axis=1 for nested structures
    assert ak.array_equal(
        ak.combinations(virtual_listarray, 2, axis=1),
        ak.combinations(listarray, 2, axis=1),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_argcombinations(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    # Need to use axis=1 for nested structures
    assert ak.array_equal(
        ak.argcombinations(virtual_listarray, 2, axis=1),
        ak.argcombinations(listarray, 2, axis=1),
    )
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_nan_to_none(numpy_like):
    # Create a ListArray with NaN values
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_none(virtual_array), ak.nan_to_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_nan_to_num(numpy_like):
    # Create a ListArray with NaN values
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_num(virtual_array), ak.nan_to_num(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_local_index(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    # For ListArray, we should use axis=1 to get indices within each list
    assert ak.array_equal(
        ak.local_index(virtual_listarray, axis=1),
        ak.local_index(listarray, axis=1),
    )
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_run_lengths(numpy_like):
    # Create a ListArray with repeated values for run_lengths test
    starts = np.array([0, 3], dtype=np.int64)
    stops = np.array([3, 6], dtype=np.int64)
    content = np.array([1, 1, 2, 3, 3, 3], dtype=np.int64)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 3], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([3, 6], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 1, 2, 3, 3, 3], dtype=np.int64),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    # Need to use axis=1 to check run lengths within each list
    assert ak.array_equal(ak.run_lengths(virtual_array), ak.run_lengths(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_round(numpy_like):
    # Create a ListArray with float values for rounding
    starts = np.array([0, 2], dtype=np.int64)
    stops = np.array([2, 4], dtype=np.int64)
    content = np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.round(virtual_array), ak.round(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_isclose(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.isclose(virtual_listarray, listarray, rtol=1e-5, atol=1e-8),
        ak.isclose(listarray, listarray, rtol=1e-5, atol=1e-8),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_almost_equal(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.almost_equal(virtual_listarray, listarray),
        ak.almost_equal(listarray, listarray),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_real(numpy_like):
    # Create a ListArray with complex values for real test
    starts = np.array([0, 2], dtype=np.int64)
    stops = np.array([2, 3], dtype=np.int64)
    content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 3], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.real(virtual_array), ak.real(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_imag(numpy_like):
    # Create a ListArray with complex values for imag test
    starts = np.array([0, 2], dtype=np.int64)
    stops = np.array([2, 3], dtype=np.int64)
    content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 3], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.imag(virtual_array), ak.imag(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_angle(numpy_like):
    # Create a ListArray with complex values for angle test
    starts = np.array([0, 2], dtype=np.int64)
    stops = np.array([2, 4], dtype=np.int64)
    content = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array(
            [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(
        ak.angle(virtual_array, deg=True),
        ak.angle(array, deg=True),
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_zeros_like(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(ak.zeros_like(virtual_listarray), ak.zeros_like(listarray))
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_ones_like(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(ak.ones_like(virtual_listarray), ak.ones_like(listarray))
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_full_like(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.full_like(virtual_listarray, 100), ak.full_like(listarray, 100)
    )
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_slicing(listarray, virtual_listarray):
    # Convert to ak.Array for slicing operations
    list_array = ak.Array(listarray)
    virtual_list_array = ak.Array(virtual_listarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Test slicing the outer dimension
    assert ak.array_equal(virtual_list_array[1:3], list_array[1:3])

    # Test indexing and then slicing inner dimension
    assert ak.array_equal(virtual_list_array[0][0:2], list_array[0][0:2])

    assert virtual_list_array.layout.is_any_materialized


def test_listarray_mask_operations(listarray, virtual_listarray):
    list_array = ak.Array(listarray)
    virtual_list_array = ak.Array(virtual_listarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Create a boolean mask
    mask = ak.to_backend(ak.Array([True, False, True, False]), "jax")

    # Test masking
    assert ak.array_equal(virtual_list_array[mask], list_array[mask])

    assert virtual_list_array.layout.is_any_materialized


def test_listarray_arithmetics(listarray, virtual_listarray):
    list_array = ak.Array(listarray)
    virtual_list_array = ak.Array(virtual_listarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Test addition
    assert ak.array_equal(virtual_list_array + 10, list_array + 10)

    # Test multiplication
    assert ak.array_equal(virtual_list_array * 2, list_array * 2)

    # Test division
    assert ak.array_equal(virtual_list_array / 2, list_array / 2)

    assert virtual_list_array.layout.is_any_materialized
    assert virtual_list_array.layout.is_all_materialized


def test_recordarray_to_list(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    assert ak.to_list(virtual_recordarray) == ak.to_list(recordarray)
    assert virtual_recordarray.is_any_materialized
    assert virtual_recordarray.is_all_materialized


def test_recordarray_to_json(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    assert ak.to_json(virtual_recordarray) == ak.to_json(recordarray)
    assert virtual_recordarray.is_any_materialized
    assert virtual_recordarray.is_all_materialized


def test_recordarray_to_buffers(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    out1 = ak.to_buffers(recordarray)
    out2 = ak.to_buffers(virtual_recordarray)
    # form
    assert out1[0] == out2[0]
    # length
    assert out1[1] == out2[1]
    # container
    assert set(out1[2].keys()) == set(out2[2].keys())
    for key in out1[2]:
        if isinstance(out2[2][key], VirtualArray):
            assert not out2[2][key].is_materialized
            assert ak.all(out1[2][key] == out2[2][key])
            assert out2[2][key].is_materialized
        else:
            assert ak.all(out1[2][key] == out2[2][key])


def test_recordarray_is_valid(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    assert ak.is_valid(virtual_recordarray) == ak.is_valid(recordarray)
    assert ak.validity_error(virtual_recordarray) == ak.validity_error(recordarray)


def test_recordarray_field_access(recordarray, virtual_recordarray):
    # Test accessing fields directly
    assert not virtual_recordarray.is_any_materialized

    # Access x field (ListOffsetArray)
    x_field = virtual_recordarray["x"]
    assert isinstance(x_field, ak.contents.ListOffsetArray)
    assert not x_field.is_any_materialized

    # Access y field (NumpyArray)
    y_field = virtual_recordarray["y"]
    assert isinstance(y_field, ak.contents.NumpyArray)
    assert not y_field.data.is_materialized

    # Verify field values
    assert ak.to_list(x_field) == ak.to_list(recordarray["x"])
    assert ak.to_list(y_field) == ak.to_list(recordarray["y"])

    # Check materialization state after access
    assert x_field.is_any_materialized
    assert y_field.data.is_materialized


def test_recordarray_zip(recordarray, virtual_recordarray):
    # Test zipping the RecordArray
    zip1 = ak.zip({"record": recordarray})
    zip2 = ak.zip({"record": virtual_recordarray})

    assert not virtual_recordarray.is_any_materialized
    assert zip1.fields == zip2.fields
    assert ak.array_equal(ak.materialize(zip2), zip1)


def test_recordarray_concatenate(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    result = ak.concatenate([virtual_recordarray, virtual_recordarray])
    expected = ak.concatenate([recordarray, recordarray])
    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_where_x_field(recordarray, virtual_recordarray):
    # Test where operation on the x field (ListOffsetArray)
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Create a mask compatible with nested list structure in x field
    mask = ak.to_backend(
        ak.Array(
            [[True, False], [False, True], [True, False, True], [False, True, True], []]
        ),
        "jax",
    )

    # Apply where operation on the x field
    result1 = ak.where(mask, record_array.x, 999)
    result2 = ak.where(mask, virtual_record_array.x, 999)

    assert ak.array_equal(result1, result2)
    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_where_y_field(recordarray, virtual_recordarray):
    # Test where operation on the y field (NumpyArray)
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Create a simple mask for the y field
    mask = ak.to_backend(ak.Array([True, False, True, False, True]), "jax")

    # Apply where operation on the y field
    result1 = ak.where(mask, record_array.y, 999)
    result2 = ak.where(mask, virtual_record_array.y, 999)

    assert ak.array_equal(result1, result2)
    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_count_x_field(recordarray, virtual_recordarray):
    # Test count on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.count(recordarray["x"], axis=1)

    result = ak.count(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_count_y_field(recordarray, virtual_recordarray):
    # Test count on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.count(recordarray["y"])

    result = ak.count(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_count_nonzero_x_field(recordarray, virtual_recordarray):
    # Test count_nonzero on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.count_nonzero(recordarray["x"], axis=1)

    result = ak.count_nonzero(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_count_nonzero_y_field(recordarray, virtual_recordarray):
    # Test count_nonzero on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.count_nonzero(recordarray["y"])

    result = ak.count_nonzero(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_sum_x_field(recordarray, virtual_recordarray):
    # Test sum on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.sum(recordarray["x"], axis=1)

    result = ak.sum(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_sum_y_field(recordarray, virtual_recordarray):
    # Test sum on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.sum(recordarray["y"])

    result = ak.sum(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_any_x_field(recordarray, virtual_recordarray):
    # Test any on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.any(recordarray["x"], axis=1)

    result = ak.any(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_any_y_field(recordarray, virtual_recordarray):
    # Test any on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.any(recordarray["y"])

    result = ak.any(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_all_x_field(recordarray, virtual_recordarray):
    # Test all on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.all(recordarray["x"], axis=1)

    result = ak.all(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_all_y_field(recordarray, virtual_recordarray):
    # Test all on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.all(recordarray["y"])

    result = ak.all(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_min_x_field(recordarray, virtual_recordarray):
    # Test min on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.min(recordarray["x"], axis=1)

    result = ak.min(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_min_y_field(recordarray, virtual_recordarray):
    # Test min on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.min(recordarray["y"])

    result = ak.min(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_max_x_field(recordarray, virtual_recordarray):
    # Test max on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.max(recordarray["x"], axis=1)

    result = ak.max(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_max_y_field(recordarray, virtual_recordarray):
    # Test max on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.max(recordarray["y"])

    result = ak.max(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argmin_x_field(recordarray, virtual_recordarray):
    # Test argmin on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.argmin(recordarray["x"], axis=1)

    result = ak.argmin(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argmin_y_field(recordarray, virtual_recordarray):
    # Test argmin on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.argmin(recordarray["y"])

    result = ak.argmin(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argmax_x_field(recordarray, virtual_recordarray):
    # Test argmax on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.argmax(recordarray["x"], axis=1)

    result = ak.argmax(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argmax_y_field(recordarray, virtual_recordarray):
    # Test argmax on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.argmax(recordarray["y"])

    result = ak.argmax(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_sort_x_field(recordarray, virtual_recordarray):
    # Test sort on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.sort(recordarray["x"], axis=1)

    result = ak.sort(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_sort_y_field(recordarray, virtual_recordarray):
    # Test sort on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.sort(recordarray["y"])

    result = ak.sort(y_field)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argsort_x_field(recordarray, virtual_recordarray):
    # Test argsort on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.argsort(recordarray["x"], axis=1)

    result = ak.argsort(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argsort_y_field(recordarray, virtual_recordarray):
    # Test argsort on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.argsort(recordarray["y"])

    result = ak.argsort(y_field)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_is_none(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    assert ak.array_equal(ak.is_none(virtual_recordarray), ak.is_none(recordarray))
    assert virtual_recordarray.is_any_materialized


def test_recordarray_local_index_x_field(recordarray, virtual_recordarray):
    # Test local_index on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.local_index(recordarray["x"], axis=1)

    result = ak.local_index(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_local_index_y_field(recordarray, virtual_recordarray):
    # Test local_index on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.local_index(recordarray["y"])

    result = ak.local_index(y_field)
    assert ak.array_equal(result, expected)
    assert not virtual_recordarray.is_any_materialized


def test_recordarray_combinations_x_field(recordarray, virtual_recordarray):
    # Test combinations on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.combinations(recordarray["x"], 2, axis=1)

    result = ak.combinations(x_field, 2, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_combinations_y_field(recordarray, virtual_recordarray):
    # Test combinations on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.combinations(recordarray["y"], 2, axis=0)

    result = ak.combinations(y_field, 2, axis=0)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_nan_to_none_x_field(numpy_like):
    # Create a RecordArray with NaN values in the x field
    offsets = np.array([0, 2, 4, 7, 10, 10], dtype=np.int64)
    x_content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    y_content = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10, 10], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test nan_to_none on x field
    assert not virtual_array.is_any_materialized

    result_x = ak.nan_to_none(virtual_array["x"])
    expected_x = ak.nan_to_none(array["x"])

    assert ak.array_equal(result_x, expected_x)
    assert virtual_array.is_any_materialized


def test_recordarray_nan_to_none_y_field(numpy_like):
    # Create a RecordArray with NaN values in the y field
    offsets = np.array([0, 2, 4, 7, 10, 10], dtype=np.int64)
    x_content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    y_content = np.array([0.1, np.nan, 0.3, np.nan, 0.5], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10, 10], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
        ),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, np.nan, 0.3, np.nan, 0.5], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test nan_to_none on y field
    assert not virtual_array.is_any_materialized

    result_y = ak.nan_to_none(virtual_array["y"])
    expected_y = ak.nan_to_none(array["y"])

    assert ak.array_equal(result_y, expected_y)
    assert virtual_array.is_any_materialized


def test_recordarray_zeros_like(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.zeros_like(virtual_recordarray)
    expected = ak.zeros_like(recordarray)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_ones_like(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.ones_like(virtual_recordarray)
    expected = ak.ones_like(recordarray)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_full_like(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.full_like(virtual_recordarray, 100)
    expected = ak.full_like(recordarray, 100)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_slicing(recordarray, virtual_recordarray):
    # Convert to ak.Array for slicing operations
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Test slicing the record array
    assert ak.array_equal(virtual_record_array[1:3], record_array[1:3])

    # Test slicing a field
    assert ak.array_equal(virtual_record_array.x[1:3], record_array.x[1:3])
    assert ak.array_equal(virtual_record_array.y[1:3], record_array.y[1:3])

    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_mask_operations(recordarray, virtual_recordarray):
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Create a boolean mask
    mask = ak.to_backend(ak.Array([True, False, True, False, True]), "jax")

    # Test masking the record array
    assert ak.array_equal(virtual_record_array[mask], record_array[mask])

    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_arithmetics_x_field(recordarray, virtual_recordarray):
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Test addition on the x field
    assert ak.array_equal(virtual_record_array.x + 10, record_array.x + 10)

    # Test multiplication on the x field
    assert ak.array_equal(virtual_record_array.x * 2, record_array.x * 2)

    # Test division on the x field
    assert ak.array_equal(virtual_record_array.x / 2, record_array.x / 2)

    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_arithmetics_y_field(recordarray, virtual_recordarray):
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Test addition on the y field
    assert ak.array_equal(virtual_record_array.y + 10, record_array.y + 10)

    # Test multiplication on the y field
    assert ak.array_equal(virtual_record_array.y * 2, record_array.y * 2)

    # Test division on the y field
    assert ak.array_equal(virtual_record_array.y / 2, record_array.y / 2)

    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_firsts_x_field(recordarray, virtual_recordarray):
    # Test firsts on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.firsts(recordarray["x"])

    result = ak.firsts(x_field)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_firsts_y_field(recordarray, virtual_recordarray):
    # Test firsts on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.firsts(recordarray["y"], axis=0)

    result = ak.firsts(y_field, axis=0)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_flatten_x_field(recordarray, virtual_recordarray):
    # Test flatten on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.flatten(recordarray["x"])

    result = ak.flatten(x_field)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_to_regular_x_field(recordarray, virtual_recordarray):
    # Test to_regular on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.to_regular(recordarray["x"], axis=0)

    result = ak.to_regular(x_field, axis=0)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_to_regular_y_field(recordarray, virtual_recordarray):
    # Test to_regular on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.to_regular(recordarray["y"], axis=0)

    result = ak.to_regular(y_field, axis=0)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_run_lengths_x_field(numpy_like):
    # Create a RecordArray with repeated values in the x field for run_lengths test
    offsets = np.array([0, 3, 6, 6], dtype=np.int64)
    x_content = np.array([1, 1, 2, 3, 3, 3], dtype=np.int64)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 3, 6, 6], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 1, 2, 3, 3, 3], dtype=np.int64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test run_lengths on x field
    assert not virtual_array.is_any_materialized

    result = ak.run_lengths(virtual_array["x"])
    expected = ak.run_lengths(array["x"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_run_lengths_y_field(numpy_like):
    # Create a RecordArray with repeated values in the y field for run_lengths test
    offsets = np.array([0, 3, 6, 6], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)
    y_content = np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 3, 6, 6], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test run_lengths on y field
    assert not virtual_array.is_any_materialized

    result = ak.run_lengths(virtual_array["y"])
    expected = ak.run_lengths(array["y"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_round_x_field(numpy_like):
    # Create a RecordArray with float values in the x field for rounding
    offsets = np.array([0, 2, 4, 4], dtype=np.int64)
    x_content = np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 4], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test round on x field
    assert not virtual_array.is_any_materialized

    result = ak.round(virtual_array["x"])
    expected = ak.round(array["x"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_round_y_field(numpy_like):
    # Create a RecordArray with float values in the y field for rounding
    offsets = np.array([0, 2, 4, 4], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    y_content = np.array([1.234, 2.567, 3.499], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 4], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.234, 2.567, 3.499], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test round on y field
    assert not virtual_array.is_any_materialized

    result = ak.round(virtual_array["y"])
    expected = ak.round(array["y"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_isclose(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.isclose(virtual_recordarray, recordarray, rtol=1e-5, atol=1e-8)
    expected = ak.isclose(recordarray, recordarray, rtol=1e-5, atol=1e-8)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_almost_equal(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.almost_equal(virtual_recordarray, recordarray)
    expected = ak.almost_equal(recordarray, recordarray)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_real_x_field(numpy_like):
    # Create a RecordArray with complex values in the x field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test real on x field
    assert not virtual_array.is_any_materialized

    result = ak.real(virtual_array["x"])
    expected = ak.real(array["x"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_real_y_field(numpy_like):
    # Create a RecordArray with complex values in the y field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    y_content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3], dtype=np.float64),
    )
    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test real on y field
    assert not virtual_array.is_any_materialized

    result = ak.real(virtual_array["y"])
    expected = ak.real(array["y"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_imag_x_field(numpy_like):
    # Create a RecordArray with complex values in the x field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test imag on x field
    assert not virtual_array.is_any_materialized

    result = ak.imag(virtual_array["x"])
    expected = ak.imag(array["x"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_imag_y_field(numpy_like):
    # Create a RecordArray with complex values in the y field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    y_content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test imag on y field
    assert not virtual_array.is_any_materialized

    result = ak.imag(virtual_array["y"])
    expected = ak.imag(array["y"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_angle_x_field(numpy_like):
    # Create a RecordArray with complex values in the x field
    offsets = np.array([0, 2, 4, 4], dtype=np.int64)
    x_content = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 4], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array(
            [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128
        ),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test angle on x field
    assert not virtual_array.is_any_materialized

    result = ak.angle(virtual_array["x"], deg=True)
    expected = ak.angle(array["x"], deg=True)

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_angle_y_field(numpy_like):
    # Create a RecordArray with complex values in the y field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    y_content = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex128)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex128),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test angle on y field
    assert not virtual_array.is_any_materialized

    result = ak.angle(virtual_array["y"], deg=True)
    expected = ak.angle(array["y"], deg=True)

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_fields(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    # Test fields property
    assert virtual_recordarray.fields == recordarray.fields
    assert not virtual_recordarray.is_any_materialized


def test_recordarray_with_field(recordarray, virtual_recordarray, numpy_like):
    assert not virtual_recordarray.is_any_materialized

    # Create a new field to add
    new_field = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
    )

    # Test with_field method which adds a new field
    result = ak.with_field(virtual_recordarray, ak.contents.NumpyArray(new_field), "z")
    expected = ak.with_field(
        recordarray, ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])), "z"
    )

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_without_field(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    # Test without_field method which removes a field
    result = ak.without_field(virtual_recordarray, "y")
    expected = ak.without_field(recordarray, "y")

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_broadcast_arrays(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    # Test broadcast_arrays with a RecordArray
    out = ak.broadcast_arrays(5, virtual_recordarray)
    assert len(out) == 2
    # The virtual_recordarray should stay virtual until accessed
    assert not virtual_recordarray.is_any_materialized
    assert not out[1].layout.is_any_materialized

    # Verify content after materialization
    assert ak.array_equal(out[1], recordarray)


def test_recordarray_with_custom_generator(numpy_like):
    # Create a RecordArray with a more complex generator function

    # Define generator functions that compute values
    def compute_offsets():
        # Compute offsets dynamically
        return np.array([0, 2, 5, 9, 10], dtype=np.int64)

    def compute_content():
        # Compute content with a formula
        return np.array([i**2 for i in range(10)], dtype=np.float64)

    def compute_y_values():
        # Compute y values with a formula
        return np.array([np.sin(i) for i in range(5)], dtype=np.float64)

    # Create virtual arrays with these generators
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=compute_offsets,
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=compute_content,
    )

    virtual_y = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=compute_y_values,
    )

    # Create the RecordArray
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    y_field = ak.contents.NumpyArray(virtual_y)

    virtual_array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create the expected array for comparison
    offsets = np.array([0, 2, 5, 9, 10], dtype=np.int64)
    content = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81], dtype=np.float64)
    y_values = np.array(
        [
            0.0,
            0.8414709848078965,
            0.9092974268256817,
            0.1411200080598672,
            -0.7568024953079282,
        ],
        dtype=np.float64,
    )

    x_field_regular = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    y_field_regular = ak.contents.NumpyArray(y_values)

    regular_array = ak.contents.RecordArray(
        [x_field_regular, y_field_regular], ["x", "y"]
    )

    # Test operations
    assert not virtual_array.is_any_materialized

    # Check to_list
    assert ak.to_list(virtual_array) == ak.to_list(regular_array)

    # Check field access and operations
    assert ak.array_equal(
        ak.sum(virtual_array["x"], axis=1), ak.sum(regular_array["x"], axis=1)
    )
    assert ak.array_equal(ak.min(virtual_array["y"]), ak.min(regular_array["y"]))

    assert virtual_array.is_any_materialized


def test_recordarray_with_none_values(numpy_like):
    # Create a RecordArray with None values in both fields

    # Create x field with None values
    x_offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    x_index = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    # Create y field with None values
    y_index = np.array([0, -1, 1, -1, 2], dtype=np.int64)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(x_offsets),
        ak.contents.IndexedOptionArray(
            ak.index.Index(x_index), ak.contents.NumpyArray(x_content)
        ),
    )

    y_field = ak.contents.IndexedOptionArray(
        ak.index.Index(y_index), ak.contents.NumpyArray(y_content)
    )

    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual versions
    virtual_x_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_x_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_y_index = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2], dtype=np.int64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_x_offsets),
        ak.contents.IndexedOptionArray(
            ak.index.Index(virtual_x_index), ak.contents.NumpyArray(virtual_x_content)
        ),
    )

    virtual_y_field = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_y_index), ak.contents.NumpyArray(virtual_y_content)
    )

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test operations with None values
    assert not virtual_array.is_any_materialized

    # Check to_list
    assert ak.to_list(virtual_array) == ak.to_list(array)

    # Test drop_none
    assert ak.array_equal(ak.drop_none(virtual_array), ak.drop_none(array))

    # Test fill_none
    assert ak.array_equal(ak.fill_none(virtual_array, 999), ak.fill_none(array, 999))

    # Test is_none
    assert ak.array_equal(ak.is_none(virtual_array), ak.is_none(array))

    assert virtual_array.is_any_materialized


def test_recordarray_advanced_indexing(recordarray, virtual_recordarray):
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Test slicing with step
    slice_result = virtual_record_array[::2]
    expected_slice = record_array[::2]
    assert ak.array_equal(slice_result, expected_slice)

    # Test fancy indexing with array of indices
    indices = np.array([3, 1, 2])
    fancy_result = virtual_record_array[indices]
    expected_fancy = record_array[indices]
    assert ak.array_equal(fancy_result, expected_fancy)

    # Test boolean masking
    mask = np.array([True, False, True, False, True])
    mask_result = virtual_record_array[mask]
    expected_mask = record_array[mask]
    assert ak.array_equal(mask_result, expected_mask)

    assert virtual_record_array.layout.is_any_materialized
