# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from awkward._connect.numba.growablebuffer import (  # noqa: E402
    GrowableBuffer,
    _from_data,
)


def test_python_append():
    # small 'initial' and 'resize' for testing
    growablebuffer = GrowableBuffer(np.int32, initial=10, resize=2.0)
    assert growablebuffer.snapshot().tolist() == []
    assert len(growablebuffer._panels) == 1

    # within the first panel
    for x in range(5):
        growablebuffer.append(x)
    assert growablebuffer.snapshot().tolist() == list(range(5))
    assert len(growablebuffer._panels) == 1

    # reaching the end of the first panel (10)
    for x in range(5, 9):
        growablebuffer.append(x)
    assert growablebuffer.snapshot().tolist() == list(range(9))
    assert len(growablebuffer._panels) == 1

    # at the end
    growablebuffer.append(9)
    assert growablebuffer.snapshot().tolist() == list(range(10))
    assert len(growablebuffer._panels) == 1

    # beyond the end; onto the second panel
    growablebuffer.append(10)
    assert growablebuffer.snapshot().tolist() == list(range(11))
    assert len(growablebuffer._panels) == 2

    # continuing into the second panel
    growablebuffer.append(11)
    assert growablebuffer.snapshot().tolist() == list(range(12))
    assert len(growablebuffer._panels) == 2

    # to the end of the second panel (30)
    for x in range(12, 30):
        growablebuffer.append(x)
    assert growablebuffer.snapshot().tolist() == list(range(30))
    assert len(growablebuffer._panels) == 2

    # continuing into the third panel
    growablebuffer.append(30)
    assert growablebuffer.snapshot().tolist() == list(range(31))
    assert len(growablebuffer._panels) == 3


def test_python_extend():
    # small 'initial' and 'resize' for testing
    growablebuffer = GrowableBuffer(np.int32, initial=10, resize=2.0)
    assert growablebuffer.snapshot().tolist() == []
    assert len(growablebuffer._panels) == 1

    # within the first panel
    growablebuffer.extend(np.array(range(5)))
    assert growablebuffer.snapshot().tolist() == list(range(5))
    assert len(growablebuffer._panels) == 1

    # up to (and touching) the end of the first panel (10)
    growablebuffer.extend(np.array(range(5, 10)))
    assert growablebuffer.snapshot().tolist() == list(range(10))
    assert len(growablebuffer._panels) == 1

    # within the second panel (which ends at 30)
    growablebuffer.extend(np.array(range(10, 20)))
    assert growablebuffer.snapshot().tolist() == list(range(20))
    assert len(growablebuffer._panels) == 2

    # touching the end of the second panel
    growablebuffer.extend(np.array(range(20, 30)))
    assert growablebuffer.snapshot().tolist() == list(range(30))
    assert len(growablebuffer._panels) == 2

    # fill one whole panel exactly (start to end)
    growablebuffer.extend(np.array(range(30, 50)))
    assert growablebuffer.snapshot().tolist() == list(range(50))
    assert len(growablebuffer._panels) == 3

    # fill more than one panel, starting at a threshold
    growablebuffer.extend(np.array(range(50, 80)))
    assert growablebuffer.snapshot().tolist() == list(range(80))
    assert len(growablebuffer._panels) == 4

    # fill more than one panel, not starting at a threshold, but ending on one
    growablebuffer.extend(np.array(range(80, 110)))
    assert growablebuffer.snapshot().tolist() == list(range(110))
    assert len(growablebuffer._panels) == 5

    # fill lots of panels, starting at a threshold
    growablebuffer.extend(np.array(range(110, 160)))
    assert growablebuffer.snapshot().tolist() == list(range(160))
    assert len(growablebuffer._panels) == 6

    # fill lots of panels, not starting at a threshold or ending on one
    growablebuffer.extend(np.array(range(160, 200)))
    assert growablebuffer.snapshot().tolist() == list(range(200))
    assert len(growablebuffer._panels) == 7

    # fill lots of panels, not starting at a threshold, but ending on one
    growablebuffer.extend(np.array(range(200, 250)))
    assert growablebuffer.snapshot().tolist() == list(range(250))
    assert len(growablebuffer._panels) == 8

    # fill a whole lot of panels, just for fun
    growablebuffer.extend(np.array(range(250, 1000)))
    assert growablebuffer.snapshot().tolist() == list(range(1000))
    assert len(growablebuffer._panels) == 9


def test_unbox():
    @numba.njit
    def f1(x):
        x  # noqa: B018 (we want to test the unboxing)
        return 3.14

    growablebuffer = GrowableBuffer(np.int32)
    f1(growablebuffer)


def test_box():
    @numba.njit
    def f2(x):
        return x

    growablebuffer = GrowableBuffer(np.int32, initial=10)

    out1 = f2(growablebuffer)
    assert len(out1._panels) == len(growablebuffer._panels)
    assert out1._panels[0] is growablebuffer._panels[0]
    assert out1._length == growablebuffer._length
    assert out1._pos == growablebuffer._pos
    assert out1._resize == growablebuffer._resize

    for x in range(15):
        growablebuffer.append(x)

    out2 = f2(growablebuffer)
    assert len(out2._panels) == len(growablebuffer._panels)
    assert out2._panels[0] is growablebuffer._panels[0]
    assert out2._panels[1] is growablebuffer._panels[1]
    assert out2._length == growablebuffer._length
    assert out2._pos == growablebuffer._pos
    assert out2._resize == growablebuffer._resize

    assert len(out1._panels) == len(growablebuffer._panels)
    assert out1._panels[0] is growablebuffer._panels[0]
    assert out1._panels[1] is growablebuffer._panels[1]
    assert out1._length == growablebuffer._length
    assert out1._pos == growablebuffer._pos
    assert out1._resize == growablebuffer._resize


def test_len():
    @numba.njit
    def f3(x):
        return len(x)

    growablebuffer = GrowableBuffer(np.int32)

    assert f3(growablebuffer) == 0

    growablebuffer.append(123)

    assert f3(growablebuffer) == 1


def test_from_data():
    @numba.njit
    def f4():
        return _from_data(
            numba.typed.List([np.array([3.12], np.float32)]),
            np.array([1, 0], np.int64),
            1.23,
        )

    out = f4()
    assert isinstance(out, GrowableBuffer)
    assert out.dtype == np.dtype(np.float32)
    assert len(out) == 1
    assert out._pos == 0
    assert out._resize == 1.23


def test_ctor():
    @numba.njit
    def f5():
        return GrowableBuffer("f4")

    out = f5()
    assert isinstance(out, GrowableBuffer)
    assert out.dtype == np.dtype("f4")
    assert len(out) == 0
    assert len(out._panels) == 1
    assert len(out._panels[0]) == 1024
    assert out._pos == 0
    assert out._resize == 8.0

    @numba.njit
    def f6():
        return GrowableBuffer("f4", initial=10)

    out = f6()
    assert isinstance(out, GrowableBuffer)
    assert out.dtype == np.dtype("f4")
    assert len(out) == 0
    assert len(out._panels) == 1
    assert len(out._panels[0]) == 10
    assert out._pos == 0
    assert out._resize == 8.0

    @numba.njit
    def f7():
        return GrowableBuffer("f4", resize=2.0)

    out = f7()
    assert isinstance(out, GrowableBuffer)
    assert out.dtype == np.dtype("f4")
    assert len(out) == 0
    assert len(out._panels) == 1
    assert len(out._panels[0]) == 1024
    assert out._pos == 0
    assert out._resize == 2.0

    @numba.njit
    def f8():
        return GrowableBuffer("f4", resize=2.0, initial=10)

    out = f8()
    assert isinstance(out, GrowableBuffer)
    assert out.dtype == np.dtype("f4")
    assert len(out) == 0
    assert len(out._panels) == 1
    assert len(out._panels[0]) == 10
    assert out._pos == 0
    assert out._resize == 2.0

    @numba.njit
    def f9():
        return GrowableBuffer(np.float32)

    out = f9()
    assert isinstance(out, GrowableBuffer)
    assert out.dtype == np.dtype(np.float32)
    assert len(out) == 0
    assert len(out._panels) == 1
    assert len(out._panels[0]) == 1024
    assert out._pos == 0
    assert out._resize == 8.0

    @numba.njit
    def f10():
        return GrowableBuffer(np.dtype(np.float32))

    out = f10()
    assert isinstance(out, GrowableBuffer)
    assert out.dtype == np.dtype(np.float32)
    assert len(out) == 0
    assert len(out._panels) == 1
    assert len(out._panels[0]) == 1024
    assert out._pos == 0
    assert out._resize == 8.0


def test_length_and_pos():
    @numba.njit
    def f11(growablebuffer):
        return growablebuffer._length_get(), growablebuffer._pos_get()

    @numba.njit
    def f12(growablebuffer):
        growablebuffer._length_set(1)
        growablebuffer._pos_set(2)

    @numba.njit
    def f13(growablebuffer):
        growablebuffer._length_inc(10)
        growablebuffer._pos_inc(10)

    growablebuffer = GrowableBuffer(np.float32)
    growablebuffer._length = 123
    growablebuffer._pos = 99

    assert f11(growablebuffer) == (123, 99)

    f12(growablebuffer)

    assert growablebuffer._length == 1
    assert growablebuffer._pos == 2

    f13(growablebuffer)

    assert growablebuffer._length == 11
    assert growablebuffer._pos == 12


def test_add_panel():
    @numba.njit
    def f14(growablebuffer):
        growablebuffer._add_panel()

    growablebuffer = GrowableBuffer(np.float32, initial=10, resize=2.0)
    growablebuffer._pos = 5

    assert len(growablebuffer._panels) == 1
    assert len(growablebuffer._panels[0]) == 10
    assert growablebuffer._pos == 5

    f14(growablebuffer)

    assert len(growablebuffer._panels) == 2
    assert len(growablebuffer._panels[0]) == 10
    assert len(growablebuffer._panels[1]) == 20
    assert growablebuffer._pos == 0


def test_append():
    @numba.njit
    def f15(growablebuffer):
        for i in range(8):
            growablebuffer.append(i)

    growablebuffer = GrowableBuffer(np.float32, initial=10, resize=2.0)

    f15(growablebuffer)

    assert growablebuffer.snapshot().tolist() == list(range(8))

    f15(growablebuffer)

    assert growablebuffer.snapshot().tolist() == list(range(8)) + list(range(8))


def test_extend():
    @numba.njit
    def f16(growablebuffer):
        growablebuffer.extend(np.arange(8))

    growablebuffer = GrowableBuffer(np.float32, initial=10, resize=2.0)

    f16(growablebuffer)

    assert growablebuffer.snapshot().tolist() == list(range(8))

    f16(growablebuffer)

    assert growablebuffer.snapshot().tolist() == list(range(8)) + list(range(8))


def test_snapshot():
    @numba.njit
    def f17(growablebuffer):
        return growablebuffer.snapshot()

    growablebuffer = GrowableBuffer(np.float32, initial=10, resize=2.0)

    assert f17(growablebuffer).tolist() == []

    growablebuffer.extend(range(8))

    assert f17(growablebuffer).tolist() == list(range(8))

    growablebuffer.extend(range(8))

    assert f17(growablebuffer).tolist() == list(range(8)) + list(range(8))


def test_numba_append():
    @numba.njit
    def create():
        # small 'initial' and 'resize' for testing
        return GrowableBuffer(np.int32, initial=10, resize=2.0)

    @numba.njit
    def append_range(growablebuffer, start, stop):
        for x in range(start, stop):
            growablebuffer.append(x)

    @numba.njit
    def append_single(growablebuffer, x):
        growablebuffer.append(x)

    @numba.njit
    def snapshot(growablebuffer):
        return growablebuffer.snapshot()

    growablebuffer = create()
    assert snapshot(growablebuffer).tolist() == []
    assert len(growablebuffer._panels) == 1

    # within the first panel
    append_range(growablebuffer, 0, 5)
    assert snapshot(growablebuffer).tolist() == list(range(5))
    assert len(growablebuffer._panels) == 1

    # reaching the end of the first panel (10)
    append_range(growablebuffer, 5, 9)
    assert snapshot(growablebuffer).tolist() == list(range(9))
    assert len(growablebuffer._panels) == 1

    # at the end
    append_single(growablebuffer, 9)
    assert snapshot(growablebuffer).tolist() == list(range(10))
    assert len(growablebuffer._panels) == 1

    # beyond the end; onto the second panel
    append_single(growablebuffer, 10)
    assert snapshot(growablebuffer).tolist() == list(range(11))
    assert len(growablebuffer._panels) == 2

    # continuing into the second panel
    append_single(growablebuffer, 11)
    assert snapshot(growablebuffer).tolist() == list(range(12))
    assert len(growablebuffer._panels) == 2

    # to the end of the second panel (30)
    append_range(growablebuffer, 12, 30)
    assert snapshot(growablebuffer).tolist() == list(range(30))
    assert len(growablebuffer._panels) == 2

    # continuing into the third panel
    append_single(growablebuffer, 30)
    assert snapshot(growablebuffer).tolist() == list(range(31))
    assert len(growablebuffer._panels) == 3


def test_numba_extend():
    @numba.njit
    def create():
        # small 'initial' and 'resize' for testing
        return GrowableBuffer(np.int32, initial=10, resize=2.0)

    @numba.njit
    def extend_range(growablebuffer, start, stop):
        growablebuffer.extend(np.arange(start, stop))

    @numba.njit
    def snapshot(growablebuffer):
        return growablebuffer.snapshot()

    growablebuffer = create()
    assert snapshot(growablebuffer).tolist() == []
    assert len(growablebuffer._panels) == 1

    # within the first panel
    extend_range(growablebuffer, 0, 5)
    assert snapshot(growablebuffer).tolist() == list(range(5))
    assert len(growablebuffer._panels) == 1

    # up to (and touching) the end of the first panel (10)
    extend_range(growablebuffer, 5, 10)
    assert snapshot(growablebuffer).tolist() == list(range(10))
    assert len(growablebuffer._panels) == 1

    # within the second panel (which ends at 30)
    extend_range(growablebuffer, 10, 20)
    assert snapshot(growablebuffer).tolist() == list(range(20))
    assert len(growablebuffer._panels) == 2

    # touching the end of the second panel
    extend_range(growablebuffer, 20, 30)
    assert snapshot(growablebuffer).tolist() == list(range(30))
    assert len(growablebuffer._panels) == 2

    # fill one whole panel exactly (start to end)
    extend_range(growablebuffer, 30, 50)
    assert snapshot(growablebuffer).tolist() == list(range(50))
    assert len(growablebuffer._panels) == 3

    # fill more than one panel, starting at a threshold
    extend_range(growablebuffer, 50, 80)
    assert snapshot(growablebuffer).tolist() == list(range(80))
    assert len(growablebuffer._panels) == 4

    # fill more than one panel, not starting at a threshold, but ending on one
    extend_range(growablebuffer, 80, 110)
    assert snapshot(growablebuffer).tolist() == list(range(110))
    assert len(growablebuffer._panels) == 5

    # fill lots of panels, starting at a threshold
    extend_range(growablebuffer, 110, 160)
    assert snapshot(growablebuffer).tolist() == list(range(160))
    assert len(growablebuffer._panels) == 6

    # fill lots of panels, not starting at a threshold or ending on one
    extend_range(growablebuffer, 160, 200)
    assert snapshot(growablebuffer).tolist() == list(range(200))
    assert len(growablebuffer._panels) == 7

    # fill lots of panels, not starting at a threshold, but ending on one
    extend_range(growablebuffer, 200, 250)
    assert snapshot(growablebuffer).tolist() == list(range(250))
    assert len(growablebuffer._panels) == 8

    # fill a whole lot of panels, just for fun
    extend_range(growablebuffer, 250, 1000)
    assert snapshot(growablebuffer).tolist() == list(range(1000))
    assert len(growablebuffer._panels) == 9
