# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._connect.rdataframe._from_rdataframe  # noqa: E402


ROOT = pytest.importorskip("ROOT")

compiler = ROOT.gInterpreter.Declare


def test_array_builder():
    import ctypes

    builder = ak.ArrayBuilder()

    ROOT.gInterpreter.Declare(
        f"""
    #include <functional>

    typedef uint8_t (*FuncPtr)(void*);
    typedef uint8_t (*FuncIntPtr)(void*, int64_t);

    uint8_t
    test_beginlist() {{
        return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginlist, ctypes.c_voidp).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    test_endlist() {{
        return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_endlist, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    test_integer(int64_t x) {{
        return std::invoke(reinterpret_cast<FuncIntPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_integer, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}), x);
    }}
    """
    )
    ROOT.test_beginlist()
    ROOT.test_integer(1)
    ROOT.test_integer(2)
    ROOT.test_integer(3)
    ROOT.test_endlist()

    assert ak.to_list(builder.snapshot()) == [[1, 2, 3]]

    ROOT.test_beginlist()
    ROOT.test_integer(1)
    ROOT.test_integer(2)
    ROOT.test_integer(3)
    ROOT.test_endlist()

    assert ak.to_list(builder.snapshot()) == [[1, 2, 3], [1, 2, 3]]


def test_array_builder_root():
    builder = ak.ArrayBuilder()
    func = ak._v2._connect.rdataframe._from_rdataframe.connect_ArrayBuilder(
        compiler, builder
    )

    getattr(ROOT, func["beginlist"])()
    getattr(ROOT, func["integer"])(1)
    getattr(ROOT, func["integer"])(2)
    getattr(ROOT, func["integer"])(3)
    getattr(ROOT, func["endlist"])()
    getattr(ROOT, func["real"])(3.3)

    assert ak.to_list(builder.snapshot()) == [[1, 2, 3], 3.3]


@pytest.mark.skip(reason="Failed to instantiate 'Foreach(function)'")
def test_as_ak_array():
    builder = ak.ArrayBuilder()
    func = ak._v2._connect.rdataframe._from_rdataframe.connect_ArrayBuilder(
        compiler, builder
    )

    rdf = ROOT.RDataFrame(100)
    rdf_x = rdf.Define("x", "gRandom->Rndm()")
    rdf_x.Foreach(lambda x: getattr(ROOT, func["real"])(x))
