# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak

ROOT = pytest.importorskip("ROOT")


def test_buffer_names_are_python_strings(monkeypatch):
    # The names used to be cppyy proxies into a C++ map that to_char_buffers
    # clears before from_buffers runs (#4237). Recycle the freed memory here so
    # stale proxies do not pass by luck.
    real_from_buffers = ak.from_buffers
    seen = []

    def from_buffers(form, length, container, *args, **kwargs):
        seen.append(container)
        assert all(type(name) is str for name in container)
        _recycled = ROOT.std.vector["std::string"](3000, "B" * 70)
        return real_from_buffers(form, length, container, *args, **kwargs)

    monkeypatch.setattr(ak, "from_buffers", from_buffers)

    array = ak.Array([[1], [2, 3, 4], [5]])
    result = ak.from_rdataframe(ak.to_rdataframe({"x": array}), columns=("x",))

    assert len(seen) == 1
    assert result["x"].to_list() == array.to_list()
