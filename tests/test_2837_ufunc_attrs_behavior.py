from __future__ import annotations

import awkward as ak


def test():
    x = ak.Array([1, 2, 3], behavior={"foo": "BAR"}, attrs={"hello": "world"})
    y = ak.Array([8, 0, 9], behavior={"do": "re"}, attrs={"mi": "fa"})
    z = x + y
    assert z.attrs == {"hello": "world", "mi": "fa"}
    assert z.behavior == {"foo": "BAR", "do": "re"}


def test_unary():
    x = ak.Array([1, 2, 3], behavior={"foo": "BAR"}, attrs={"hello": "world"})
    y = -x
    assert y.attrs is x.attrs
    assert x.behavior is y.behavior


def test_two_return():
    x = ak.Array([1, 2, 3], behavior={"foo": "BAR"}, attrs={"hello": "world"})
    y, y_ret = divmod(x, 2)
    assert y.attrs is y_ret.attrs
    assert y.attrs is x.attrs

    assert y.behavior is y_ret.behavior
    assert y.behavior is x.behavior
