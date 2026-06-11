from __future__ import annotations

import awkward as ak


def test_record_setitem_does_not_leak_into_sibling():
    r1 = ak.Record({"x": 1})
    r2 = ak.Record(r1)
    r2["y"] = 2
    assert r2.fields == ["x", "y"]
    # the sibling sharing the layout must be untouched
    assert r1.fields == ["x"]
    assert r2["y"] == 2


def test_record_from_record_inherits_attrs_and_behavior():
    behavior = {"foo": "bar"}
    rec = ak.Record({"x": 1}, attrs={"a": 1}, behavior=behavior)
    out = ak.Record(rec)
    assert out.attrs == {"a": 1}
    assert out.behavior == behavior


def test_arraybuilder_record_repr_no_name():
    builder = ak.ArrayBuilder()
    ctx = builder.record()
    # must not raise TypeError on the class-level _name display label
    assert repr(ctx).startswith("<ArrayBuilder.record")


def test_arraybuilder_record_context_manager_with_name():
    builder = ak.ArrayBuilder()
    with builder.record("points"):
        builder.field("x").real(1.0)
    arr = builder.snapshot()
    assert ak.parameters(arr)["__record__"] == "points"


def test_cpp_type_caches_reset_on_layout_set():
    arr = ak.Array([{"x": 1}, {"x": 2}])
    arr._cpp_type = "SENTINEL"
    arr._generator = "SENTINEL"
    arr._lookup = "SENTINEL"
    arr.layout = arr.layout
    assert arr._cpp_type is None
    assert arr._generator is None
    assert arr._lookup is None
