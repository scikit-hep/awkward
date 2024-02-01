# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import packaging.version
import pytest

import awkward as ak
import awkward._connect.cling

cppyy = pytest.importorskip("cppyy")


def test_array_as_generated_dataset():
    array = ak.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )

    generator = ak._connect.cling.togenerator(array.layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(array.layout)

    source_code = f"""
    double go_fast(ssize_t length, ssize_t* ptrs) {{
        auto awkward_array = {generator.dataset()};
        double out = 0.0;

        for (auto list : awkward_array) {{
            for (auto record : list) {{
                for (auto item : record.y()) {{
                    out += item;
                }}
            }}
        }}

        return out;
    }}
    """

    generator.generate(cppyy.cppdef)
    cppyy.cppdef(source_code)
    out = cppyy.gbl.go_fast(len(array), lookup.arrayptrs)
    assert out == ak.sum(array["y"])


@pytest.mark.skipif(
    packaging.version.Version(cppyy.__version__) < packaging.version.Version("3.0.1"),
    reason="Awkward Array can only work with cppyy 3.0.1 or later.",
)
def test_array_as_type():
    array = ak.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )

    source_code_cpp = f"""
    double go_fast_cpp({array.cpp_type} awkward_array) {{
        double out = 0.0;

        for (auto list : awkward_array) {{
            for (auto record : list) {{
                for (auto item : record.y()) {{
                    out += item;
                }}
            }}
        }}

        return out;
    }}
    """

    cppyy.cppdef(source_code_cpp)

    out = cppyy.gbl.go_fast_cpp(array)
    assert out == ak.sum(array["y"])


@pytest.mark.skipif(
    packaging.version.Version(cppyy.__version__) < packaging.version.Version("3.0.1"),
    reason="Awkward Array can only work with cppyy 3.0.1 or later.",
)
def test_array_as_templated_type():
    array = ak.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )

    source_code_cpp = """
    template<typename T>
    double go_fast_cpp_2(T& awkward_array) {
        double out = 0.0;

        for (auto list : awkward_array) {
            for (auto record : list) {
                for (auto item : record.y()) {
                    out += item;
                }
            }
        }

        return out;
    }
    """

    cppyy.cppdef(source_code_cpp)

    out = cppyy.gbl.go_fast_cpp_2[array.cpp_type](array)
    assert out == ak.sum(array["y"])
