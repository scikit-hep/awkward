# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def test_num():
    ak.num(ak.Array(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout.to_typetracer()))


def test_repr():
    # feel free to change these if the string format ever changes

    array = ak.Array(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout.to_typetracer())
    assert repr(array) == "<Array-typetracer [...] type='?? * var * float64'>"
    assert str(array) == "[...]"

    record = ak.Array(ak.Array([{"x": 1.1, "y": [1, 2, 3]}]).layout.to_typetracer())[0]
    assert (
        repr(record)
        == "<Record-typetracer {x: unknown-float64, y: [...]} type='{x: float64, y: var...'>"
    )
    assert str(record) == "{x: unknown-float64, y: [...]}"


def test_issue_1864():
    a = ak.from_iter([[None, 1], None, [1, 2]])
    tt = ak.Array(a.layout.to_typetracer())
    assert str(ak.is_none(tt, axis=0).layout.form.type) == "bool"
    assert str(ak.is_none(tt, axis=1).layout.form.type) == "option[var * bool]"


def test_numpy_touch_data():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], backend="typetracer")
    assert str((array - [100, 200, 300]).layout.form.type) == "var * float64"
