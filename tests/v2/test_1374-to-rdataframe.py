# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare

compiler(
    """
#include <cxxabi.h>
char* real_name(const char* name) {{
    int status;
    return abi::__cxa_demangle(name, 0, 0, &status);
}}
"""
)


def test_array_wrapper():
    compiler(
        """
template <typename T>
class ArrayWrapper {{ }};

class NumpyArray_float64_O1I50DFDJTY;
class ListArray_BgI9cDJVCAw;

template <typename ...ColumnTypes>
class AwkwardArrayDataSource {{ }};

template <typename ...ColumnTypes>
AwkwardArrayDataSource<ColumnTypes...> MakeDS(Wrapper<ColumnTypes>... wrappers) {{
    return AwkwardArrayDataSource<ColumnTypes...>{{}};
}}
"""
    )

    arr1 = ROOT.ArrayWrapper[ROOT.NumpyArray_float64_O1I50DFDJTY]()
    arr2 = ROOT.ArrayWrapper[ROOT.ListArray_BgI9cDJVCAw]()

    f = ROOT.MakeDS(arr1, arr2)
    print("HERE:", type(f))
