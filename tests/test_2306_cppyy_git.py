# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pytest

import awkward as ak
import awkward._connect.cling

cppyy = pytest.importorskip("cppyy")
cppyy.set_debug()

# ak.cppyy.register_and_check()


def test_array_generated_dataset_git():
    array = ak.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )

    generator = ak._connect.cling.togenerator(array.layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(array.layout)
    print("test line 26:", generator.dataset())
    print("test line 27:", generator.dataset())

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
    print("test line 49:", out)


@pytest.mark.skip("cannot handle same two arrays")
def test_array_type_git():
    array = ak.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )

    # generator = ak._connect.cling.togenerator(array.layout.form, flatlist_as_rvec=False)
    # lookup = ak._lookup.Lookup(array.layout)
    # print(generator.dataset())
    # print(generator.dataset())
    # array._cpp_type = generator.class_type()
    #
    # ak.cppyy._register(cpp_type=array._cpp_type)

    # FIXME: register 'array._cpp_type' C++ type:
    #    (awkward::ListArray_Qxm6KAjfuk derived or awkward::ArrayView base class)
    # as this 'awkward_array' Python 'ak.Array'
    source_code_cpp = f"""
    double go_fast_cpp(awkward::{array.cpptype} awkward_array) {{    // this is the only change in C++
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

    cppyy.set_debug()
    cppyy.cppdef(source_code_cpp)  # , extension = awkward)
    out = cppyy.gbl.go_fast_cpp(array)
    print("test line 92:", out)


# void* castpy(PyObject* pyobject) { return (void*)pyobject; }
# T* cast(PyObject*)
# __castcpp__
# PyObject* __castcpp__(PyObject*)


@pytest.mark.skip("invalid range expression of type 'awkward::ArrayView'")
def test_array_base_git():
    array = ak.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )

    generator = ak._connect.cling.togenerator(array.layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(array.layout)  # noqa: F841

    # subsequent calls to the generator do not change the generated type
    print("test line 114:", generator.dataset())
    print("test line 115:", generator.dataset())

    # Note, if 'awkward_array' passed in by-value,
    # it will be slides to the base class
    # (if T is indeed a base class).
    # Pass it by pointer or reference if
    # the derived class has extra state.
    source_code_cpp = """
    template<typename T>
    double go_fast_cpp_1(T awkward_array, ssize_t length) {
        double out = 0.0;

        // FIXME: ArrayView base class does not provide begin() and end()
        // nor it provides a subscript operator.
        //
        // for (auto list : awkward_array) {
        //   for (auto record : list) {
        //
        // for (int64_t i = 0; i < length; i++) {
        //   for (auto record : awkward_array[i]) {
        //

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

    cppyy.set_debug()
    cppyy.cppdef(source_code_cpp)
    out = cppyy.gbl.go_fast_cpp_1["awkward::ArrayView"](array, len(array))
    print("test line 152:", out)


# On the Python side:
#     * make a new buffer (full of pointers)
#
# On the C++ side:
#     * new type named ArrayView_aslkdiuhflskeudfhn(buffer, size)


def test_array_derived_git():
    array = ak.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )

    # Note, if 'awkward_array' passed in by-value,
    # it will be slides to the base class
    # (if T is indeed a base class).
    # Pass it by pointer or reference if
    # the derived class has extra state.
    source_code_cpp = """
    template<typename T>
    double go_fast_cpp_2(T awkward_array) {
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

    cppyy.set_debug()
    cppyy.cppdef(source_code_cpp)

    # FIXME: a tuple of parameters: move it to __cpptype__??
    def convert(array):
        return (0, len(array), 0, array._lookup.arrayptrs, 0)

    out = cppyy.gbl.go_fast_cpp_2[f"awkward::{array.cpptype}"](convert(array))
    assert out == ak.sum(array["y"])
