# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations


def test_hist_broadcast_and_flatten_returns_not_implemented_on_type_error():
    from awkward._connect.hist import broadcast_and_flatten

    result = broadcast_and_flatten([object()])
    assert result is NotImplemented
