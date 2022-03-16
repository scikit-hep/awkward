# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._connect.rdataframe.from_rdataframe  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare

# FIXME: this is a Singleton and should be initialized when the module is loaded
builder_name = ak._v2._connect.rdataframe.from_rdataframe.generate_ArrayBuilder(
    compiler
)


def test_array_builder_shim():

    # A test function accesses a ROOT.awkward.ArrayBuilderShim
    # instance - an API from Python is the same as the C++ one
    def my_awkward_array(builder_shim):
        builder_shim.beginlist()
        builder_shim.integer(1)
        builder_shim.integer(2)
        builder_shim.integer(3)
        builder_shim.endlist()

    # 1. Create an ArrayBuilder. It's not hidden from a user so that it can be
    # re-used for filling it in and taking its snapshot.
    builder = ak.ArrayBuilder()

    # 2. Create a a ROOT.awkward.ArrayBuilderShim instance
    b = ak._v2._connect.rdataframe.from_rdataframe.array_builder(builder)

    # 3. Apply the function
    my_awkward_array(b)

    # 4. Check the result by taking a snapshot
    assert ak.to_list(builder.snapshot()) == [[1, 2, 3]]

    # 5. Test it on another instance
    builder2 = ak.ArrayBuilder()
    b2 = ak._v2._connect.rdataframe.from_rdataframe.array_builder(builder2)

    my_awkward_array(b2)
    assert ak.to_list(builder2.snapshot()) == [[1, 2, 3]]

    # 6. Test it with a C++ user-defined function
    builder3 = ak.ArrayBuilder()
    b3 = ak._v2._connect.rdataframe.from_rdataframe.array_builder(builder3)

    # 7. Register a user-defined function: function name must be unique!
    compiler(
        """
    void
    my_awkward_array2(awkward::ArrayBuilderShim& builder) {{
        builder.beginlist();
        builder.integer(1);
        builder.integer(2);
        builder.integer(3);
        builder.endlist();
    }}
    """
    )

    # 8. Invoke the user-defined function
    ROOT.my_awkward_array2(b3)

    # Check the result
    assert ak.to_list(builder3.snapshot()) == [[1, 2, 3]]


def test_array_builder():
    import ctypes

    builder = ak.ArrayBuilder()

    compiler(
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


@pytest.mark.skip(reason="FIXME: TypeError: Template method resolution failed")
def test_as_ak_array():
    builder = ak.ArrayBuilder()
    b = ak._v2._connect.rdataframe.from_rdataframe.array_builder(builder)

    compiler(
        """
    void
    my_function(awkward::ArrayBuilderShim& builder, double x) {{
        builder.real(x);
    }}
    """
    )
    ROOT.my_function(b, 1.1)

    # // store a call to a member function
    # std::function<uint8_t(const awkward::ArrayBuilderShim&, double)> f_real = &awkward::ArrayBuilderShim::real;

    rdf = ROOT.RDataFrame(10).Define("x", "gRandom->Rndm()")
    rdf.Foreach[
        "std::function<uint8_t(const ROOT::awkward::ArrayBuilderShim&, double)>"
    ](ROOT.my_function, ["x"])
    array = builder.snapshot()
    assert (
        str(array.layout.form)
        == """{
    "class": "NumpyArray",
    "itemsize": 8,
    "format": "d",
    "primitive": "float64"
}"""
    )
    assert len(array.layout) == 10


def test_as_ak_array2():
    rdf = ROOT.RDataFrame(10).Define("x", "gRandom->Rndm()")

    builder = ak._v2.highlevel.ArrayBuilder()
    func = ak._v2._connect.rdataframe.from_rdataframe.connect_ArrayBuilder(
        compiler, builder
    )
    compiler(
        f"""
    uint8_t
    my_x_record(double x) {{
        {func["beginrecord"]}();
        {func["field_fast"]}("one");
        {func["real"]}(x);
        return {func["endrecord"]}();
    }}
    """
    )

    rdf.Foreach["std::function<uint8_t(double)>"](ROOT.my_x_record, ["x"])

    array = builder.snapshot()
    assert (
        str(array.layout.form)
        == """{
    "class": "RecordArray",
    "contents": {
        "one": "float64"
    }
}"""
    )
    assert len(array.layout) == 10


@pytest.mark.skip(reason="FIXME: NotImplementedError")
def test_nested_rvec_snapshot():
    compiler(
        """
    using namespace ROOT::VecOps;

    struct TwoInts {
        int a, b;
    };

    auto rv_of_rv = [] {
        return RVec<RVec<int>>{{1, 2}, {3, 4}};
    });

    auto vv() {{
        return RVec<RVec<int>>{{ {{1, 2}}, {{3, 4}} }};
    }}
    auto vvv() {{
        return RVec<RVec<RVec<int>>>{{ {{ {{1, 2}}, {{3, 4}} }},
                                       {{ {{5, 6}}, {{7, 8}} }} }};
    }}
    auto vvti() {{
        return RVec<RVec<TwoInts>>{{ {{1, 2}}, {{3, 4}} }};
    }}
    """
    )
    f = ROOT.vv()
    rdf = ROOT.RDataFrame(10).Define("x", f, ["array"])
    rdf.Count()


def test_as_awkward():
    rdf = (
        ROOT.RDataFrame(10)
        .Define("x", "gRandom->Rndm()")
        .Define("xx", "gRandom->Rndm()")
    )
    array = ak._v2._connect.rdataframe.from_rdataframe.to_awkward_array(
        rdf, compiler, columns_as_records=True
    )
    assert (
        str(array.layout.form)
        == """{
    "class": "UnionArray8_64",
    "tags": "i8",
    "index": "i64",
    "contents": [
        {
            "class": "RecordArray",
            "contents": {
                "x": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": "float64"
                }
            },
            "parameters": {
                "__record__": "x"
            }
        },
        {
            "class": "RecordArray",
            "contents": {
                "xx": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": "float64"
                }
            },
            "parameters": {
                "__record__": "xx"
            }
        }
    ]
}"""
    )


def test_rvec_snapshot():
    treeName = "t"
    fileName = "tests/samples/snapshot_rvecs.root"

    rdf = ROOT.RDataFrame(treeName, fileName)

    array = ak._v2._connect.rdataframe.from_rdataframe.to_awkward_array(
        rdf, compiler, columns_as_records=True
    )
    assert (
        str(array.layout.form)
        == """{
    "class": "UnionArray8_64",
    "tags": "i8",
    "index": "i64",
    "contents": [
        {
            "class": "RecordArray",
            "contents": {
                "NumpyArray_array": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": "float64"
                }
            },
            "parameters": {
                "__record__": "NumpyArray_array"
            }
        },
        {
            "class": "RecordArray",
            "contents": {
                "px": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": "float64"
                }
            },
            "parameters": {
                "__record__": "px"
            }
        },
        {
            "class": "RecordArray",
            "contents": {
                "py": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": "float64"
                }
            },
            "parameters": {
                "__record__": "py"
            }
        },
        {
            "class": "RecordArray",
            "contents": {
                "E": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": "float64"
                }
            },
            "parameters": {
                "__record__": "E"
            }
        },
        {
            "class": "RecordArray",
            "contents": {
                "pt": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": "float64"
                }
            },
            "parameters": {
                "__record__": "pt"
            }
        }
    ]
}"""
    )
    print(array.layout)
    print(array.to_list())


@pytest.mark.skip(reason="FIXME: a test root file is not in a github yet")
def test_rvec_snapshot_nestedrvecs():
    treeName = "t"
    fileName = "tests/samples/snapshot_nestedrvecs.root"

    ROOT.gInterpreter.ProcessLine(
        """
#include <ROOT/RVec.hxx>

struct TwoInts {
   int a, b;
};

#pragma link C++ class TwoInts+;
#pragma link C++ class ROOT::VecOps::RVec<TwoInts>+;
#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<TwoInts>>+;

    """
    )
    rdf = ROOT.RDataFrame(treeName, fileName)

    array = ak._v2._connect.rdataframe.from_rdataframe.to_awkward_array(
        rdf, compiler, columns={"vvti", "vv", "vvv"}, columns_as_records=True
    )
    # FIXME: SystemError: <cppyy.CPPOverload object at 0x139f80f40> returned a result with an exception set
    # for column `vv.fData`
    # array = rdf.AsAwkward(compiler, columns_as_records=True)
    assert (
        str(array.layout.form)
        == """{
    "class": "UnionArray8_64",
    "tags": "i8",
    "index": "i64",
    "contents": [
        {
            "class": "RecordArray",
            "contents": {
                "vvti": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": {
                        "class": "ListOffsetArray64",
                        "offsets": "i64",
                        "content": {
                            "class": "EmptyArray"
                        }
                    }
                }
            },
            "parameters": {
                "__record__": "vvti"
            }
        },
        {
            "class": "RecordArray",
            "contents": {
                "vvv": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": {
                        "class": "ListOffsetArray64",
                        "offsets": "i64",
                        "content": {
                            "class": "ListOffsetArray64",
                            "offsets": "i64",
                            "content": "int64"
                        }
                    }
                }
            },
            "parameters": {
                "__record__": "vvv"
            }
        },
        {
            "class": "RecordArray",
            "contents": {
                "vv": {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": {
                        "class": "ListOffsetArray64",
                        "offsets": "i64",
                        "content": "int64"
                    }
                }
            },
            "parameters": {
                "__record__": "vv"
            }
        }
    ]
}"""
    )
    print(array.layout)

    print(array.to_list())
