# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

import awkward._connect.numba.layoutbuilder as lb

ak.numba.register_and_check()


def test_Numpy():
    builder = lb.Numpy(np.float64)

    builder.append(1.1)
    builder.append(2.2)
    builder.extend([3.3, 4.4, 5.5])

    error = ""
    assert builder.is_valid(error), error.value

    array = builder.snapshot()
    assert str(ak.type(array)) == "5 * float64"
    assert ak.to_list(array) == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert (
        builder.form()
        == '{"class": "NumpyArray", "primitive": "float64", "form_key": "node0"}'
    )

    array1 = builder.snapshot()
    assert str(ak.type(array1)) == "5 * float64"
    assert ak.to_list(array1) == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_python_append():
    # small 'initial' and 'resize' for testing
    builder = lb.Numpy(np.int32, parameters="", initial=10, resize=2.0)
    assert ak.to_list(builder.snapshot()) == []
    assert len(builder) == 0

    # within the first panel
    for x in range(0, 5):
        builder.append(x)
    assert ak.to_list(builder.snapshot()) == list(range(5))
    assert len(builder) == 5


def test_Empty():
    builder = lb.Empty()
    assert len(builder) == 0
    assert ak.to_list(builder.snapshot()) == []

    error = ""
    assert builder.is_valid(error), error.value

    with pytest.raises(AttributeError):
        builder.content.append(1.1)

    with pytest.raises(AttributeError):
        builder.content.extend([3.3, 4.4, 5.5])

    error = ""
    assert builder.is_valid(error), error.value

    array = builder.snapshot()
    assert str(ak.type(array)) == "0 * unknown"  # FIXME: float64 ???
    assert ak.to_list(array) == []

    assert builder.form() == '{"class": "EmptyArray"}'


def test_ListOffset():
    builder = lb.ListOffset(np.int32, lb.Numpy(np.float64))
    assert len(builder) == 0
    subbuilder = builder.begin_list()
    subbuilder.append(1.1)
    subbuilder.append(2.2)
    subbuilder.append(3.3)
    builder.end_list()

    builder.begin_list()
    builder.end_list()

    builder.begin_list()
    subbuilder.append(4.4)
    subbuilder.append(5.5)
    builder.end_list()
    assert ak.to_list(builder.snapshot()) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    error = ""
    assert builder.is_valid(error), error.value


def test_List():
    builder = lb.List(np.int32, lb.Numpy(np.float64))
    assert len(builder) == 0
    assert ak.to_list(builder.snapshot()) == []

    error = ""
    assert builder.is_valid(error), error.value


def test_Regular():
    builder = lb.Regular(lb.Numpy(np.float64), 3)
    assert len(builder) == 0
    assert ak.to_list(builder.snapshot()) == []

    error = ""
    assert builder.is_valid(error), error.value


def test_unbox():
    @numba.njit
    def f1(x):
        x  # noqa: B018 (we want to test the unboxing)
        return 3.14

    builder = lb.Numpy(np.int32, parameters="", initial=10, resize=2.0)
    f1(builder)

    builder = lb.Empty()
    f1(builder)


def test_unbox_for_loop():
    @numba.njit
    def f1(x):
        for i in range(0, 10):
            x.append(i)
        return

    builder = lb.Numpy(np.int32, parameters="", initial=10, resize=2.0)
    f1(builder)
    assert ak.to_list(builder.snapshot()) == list(range(10))

    builder = lb.Empty()
    # Unknown attribute 'append' of type ak.Empty()
    with pytest.raises(numba.core.errors.TypingError):
        f1(builder)


def test_box():
    @numba.njit
    def f2(x):
        return x

    builder = lb.Numpy(np.int32)

    out1 = f2(builder)
    assert ak.to_list(out1.snapshot()) == []

    for x in range(15):
        builder.append(x)

    out2 = f2(builder)

    assert ak.to_list(out2.snapshot()) == list(range(15))

    builder = lb.Empty()

    out3 = f2(builder)
    assert ak.to_list(out3.snapshot()) == []


def test_len():
    @numba.njit
    def f3(x):
        return len(x)

    builder = lb.Numpy(np.int32, parameters="", initial=10, resize=2.0)

    assert f3(builder) == 0

    builder.append(123)

    assert f3(builder) == 1

    builder = lb.Empty()
    assert f3(builder) == 0


#
# @pytest.mark.skip("No implementation of function")
# def test_from_buffer():
#     @numba.njit
#     def f4():
#         data = ak.numba._from_data(
#             numba.typed.List([np.array([3.12], np.float32)]),
#             np.array([1, 0], np.int64),
#             1.23,
#         )
#         return _from_buffer(data)
#
#     out = f4()
#     assert isinstance(out, Numpy)
#     assert out.dtype == np.dtype(np.float32)
#     assert len(out) == 1
#
#
# def test_ctor():
#     @numba.njit
#     def f5():
#         return GrowableBuffer("f4")
#
#     out = f5()
#     assert isinstance(out, GrowableBuffer)
#     assert out.dtype == np.dtype("f4")
#     assert len(out) == 0
#     assert len(out._panels) == 1
#     assert len(out._panels[0]) == 1024
#     assert out._pos == 0
#     assert out._resize == 8.0
#
#     @numba.njit
#     def f6():
#         return GrowableBuffer("f4", initial=10)
#
#     out = f6()
#     assert isinstance(out, GrowableBuffer)
#     assert out.dtype == np.dtype("f4")
#     assert len(out) == 0
#     assert len(out._panels) == 1
#     assert len(out._panels[0]) == 10
#     assert out._pos == 0
#     assert out._resize == 8.0
#
#     @numba.njit
#     def f7():
#         return GrowableBuffer("f4", resize=2.0)
#
#     out = f7()
#     assert isinstance(out, GrowableBuffer)
#     assert out.dtype == np.dtype("f4")
#     assert len(out) == 0
#     assert len(out._panels) == 1
#     assert len(out._panels[0]) == 1024
#     assert out._pos == 0
#     assert out._resize == 2.0
#
#     @numba.njit
#     def f8():
#         return GrowableBuffer("f4", resize=2.0, initial=10)
#
#     out = f8()
#     assert isinstance(out, GrowableBuffer)
#     assert out.dtype == np.dtype("f4")
#     assert len(out) == 0
#     assert len(out._panels) == 1
#     assert len(out._panels[0]) == 10
#     assert out._pos == 0
#     assert out._resize == 2.0
#
#     @numba.njit
#     def f9():
#         return GrowableBuffer(np.float32)
#
#     out = f9()
#     assert isinstance(out, GrowableBuffer)
#     assert out.dtype == np.dtype(np.float32)
#     assert len(out) == 0
#     assert len(out._panels) == 1
#     assert len(out._panels[0]) == 1024
#     assert out._pos == 0
#     assert out._resize == 8.0
#
#     @numba.njit
#     def f10():
#         return GrowableBuffer(np.dtype(np.float32))
#
#     out = f10()
#     assert isinstance(out, GrowableBuffer)
#     assert out.dtype == np.dtype(np.float32)
#     assert len(out) == 0
#     assert len(out._panels) == 1
#     assert len(out._panels[0]) == 1024
#     assert out._pos == 0
#     assert out._resize == 8.0
#
#
#


def test_append():
    @numba.njit
    def f15(builder):
        for i in range(8):
            builder.append(i)

    builder = lb.Numpy(np.float32)

    f15(builder)

    assert ak.to_list(builder.snapshot()) == list(range(8))

    f15(builder)

    assert ak.to_list(builder.snapshot()) == list(range(8)) + list(range(8))


def test_extend():
    @numba.njit
    def f16(builder):
        builder.extend(np.arange(8))

    builder = lb.Numpy(np.float32)

    f16(builder)

    assert ak.to_list(builder.snapshot()) == list(range(8))

    f16(builder)

    assert ak.to_list(builder.snapshot()) == list(range(8)) + list(range(8))


# def test_snapshot():
#     @numba.njit
#     def f17(builder):
#         return builder.snapshot()
#
#     builder = Numpy(np.float32)
#
#     assert ak.to_list(f17(builder)) == []
#
#     builder.extend(range(8))
#
#     assert ak.to_list(f17(builder)) == list(range(8))
#
#     builder.extend(range(8))
#
#     assert ak.to_list(f17(builder)) == list(range(8)) + list(range(8))
#


def test_numba_append():
    # FIXME:@numba.njit
    def create():
        return lb.Numpy(np.int32)

    @numba.njit
    def append_range(builder, start, stop):
        for x in range(start, stop):
            builder.append(x)

    @numba.njit
    def append_single(builder, x):
        builder.append(x)

    @numba.njit
    def snapshot(builder):
        return builder.snapshot()

    builder = create()
    assert ak.to_list(snapshot(builder)) == []
    assert len(builder) == 0

    append_range(builder, 0, 5)
    assert ak.to_list(snapshot(builder)) == list(range(5))
    assert len(builder) == 5

    append_range(builder, 5, 9)
    assert ak.to_list(snapshot(builder)) == list(range(9))
    assert len(builder) == 9

    append_single(builder, 9)
    assert ak.to_list(snapshot(builder)) == list(range(10))
    assert len(builder) == 10

    append_single(builder, 10)
    assert ak.to_list(snapshot(builder)) == list(range(11))
    assert len(builder) == 11

    append_single(builder, 11)
    assert ak.to_list(snapshot(builder)) == list(range(12))
    assert len(builder) == 12

    append_range(builder, 12, 30)
    assert ak.to_list(snapshot(builder)) == list(range(30))
    assert len(builder) == 30

    append_single(builder, 30)
    assert ak.to_list(snapshot(builder)) == list(range(31))
    assert len(builder) == 31
