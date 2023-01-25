# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np

import awkward as ak
from awkward._nplikes.typetracer import TypeTracer

typetracer = TypeTracer.instance()


def test_num():
    ak.num(ak.Array(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout.to_typetracer()))


def test_repr():
    # feel free to change these if the string format ever changes

    array = ak.Array(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout.to_typetracer())
    assert repr(array) == "<Array-typetracer [...] type='3 * var * float64'>"
    assert str(array) == "[...]"

    array2 = ak.Array(
        ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout.to_typetracer(
            forget_length=True
        )
    )
    assert repr(array2) == "<Array-typetracer [...] type='## * var * float64'>"
    assert str(array2) == "[...]"

    record = ak.Array(ak.Array([{"x": 1.1, "y": [1, 2, 3]}]).layout.to_typetracer())[0]
    assert (
        repr(record)
        == "<Record-typetracer {x: ##, y: [...]} type='{x: float64, y: var * int64}'>"
    )
    assert str(record) == "{x: ##, y: [...]}"


def test_issue_1864():
    a = ak.from_iter([[None, 1], None, [1, 2]])
    tt = ak.Array(a.layout.to_typetracer())
    assert str(ak.is_none(tt, axis=0).layout.form.type) == "bool"
    assert str(ak.is_none(tt, axis=1).layout.form.type) == "option[var * bool]"


def test_numpy_touch_data():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], backend="typetracer")
    assert str((array - [100, 200, 300]).layout.form.type) == "var * float64"


def test_typetracer_binary_operator():
    a = typetracer.asarray(np.array([[1, 2], [3, 4], [5, 6]]))
    b = typetracer.asarray(np.array([[1.1], [2.2], [3.3]]))
    c = a + b
    assert c.dtype == np.dtype(np.float64)
    assert c.shape[1] == 2


def test_typetracer_formal_ufunc():
    a = typetracer.asarray(np.array([[1, 2], [3, 4], [5, 6]]))
    b = typetracer.asarray(np.array([[1.1], [2.2], [3.3]]))
    c = TypeTracer.instance().add(a, b)
    assert c.dtype == np.dtype(np.float64)
    assert c.shape[1] == 2
