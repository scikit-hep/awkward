# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import sys

import numpy as np
import pytest

from awkward._connect.numba.growablebuffer import GrowableBuffer


def test_python_append():
    # small 'initial' and 'resize' for testing
    growablebuffer = GrowableBuffer(np.int32, initial=10, resize=2.0)
    assert growablebuffer.snapshot().tolist() == []
    assert len(growablebuffer._panels) == 1

    # within the first panel
    for x in range(0, 5):
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
    growablebuffer.extend(np.array(range(0, 5)))
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
    assert len(growablebuffer._panels) == 5

    # fill more than one panel, not starting at a threshold, but ending on one
    growablebuffer.extend(np.array(range(80, 110)))
    assert growablebuffer.snapshot().tolist() == list(range(110))
    assert len(growablebuffer._panels) == 6

    # fill lots of panels, starting at a threshold
    growablebuffer.extend(np.array(range(110, 160)))
    assert growablebuffer.snapshot().tolist() == list(range(160))
    assert len(growablebuffer._panels) == 9

    # fill lots of panels, not starting at a threshold or ending on one
    growablebuffer.extend(np.array(range(160, 200)))
    assert growablebuffer.snapshot().tolist() == list(range(200))
    assert len(growablebuffer._panels) == 11

    # fill lots of panels, not starting at a threshold, but ending on one
    growablebuffer.extend(np.array(range(200, 250)))
    assert growablebuffer.snapshot().tolist() == list(range(250))
    assert len(growablebuffer._panels) == 13

    # fill a whole lot of panels, just for fun
    growablebuffer.extend(np.array(range(250, 1000)))
    assert growablebuffer.snapshot().tolist() == list(range(1000))
    assert len(growablebuffer._panels) == 51


numba = pytest.importorskip("numba")


def test_unbox():
    @numba.njit
    def f1(x):
        x
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

    assert len(out1._panels) == 1 != len(growablebuffer._panels)
    assert out1._panels[0] is growablebuffer._panels[0]
    assert out1._length == 0 != growablebuffer._length
    assert out1._pos == 0 != growablebuffer._pos
    assert out1._resize == growablebuffer._resize
