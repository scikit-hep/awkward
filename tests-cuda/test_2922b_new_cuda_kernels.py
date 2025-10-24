# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_2651_parameter_union():
    layout = ak.contents.IndexedArray(
        ak.index.Index64([0, 1, 2]),
        ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.uint32)),
        parameters={"foo": {"bar": "baz"}},
    )

    cuda_layout = ak.to_backend(layout, "cuda", highlevel=False)

    result = cuda_layout.project()
    assert result.is_equal_to(
        ak.contents.NumpyArray(
            cp.array([1, 2, 3], dtype=cp.uint32), parameters={"foo": {"bar": "baz"}}
        )
    )


def test_1928_replace_simplify_method_with_classmethod_constructor_indexed_of_union():
    unionarray = ak.from_iter(
        [0.0, 1.1, "zero", 2.2, "one", "two", "three", 3.3, 4.4, 5.5, "four"],
        highlevel=False,
    )

    cuda_unionarray = ak.to_backend(unionarray, "cuda", highlevel=False)

    indexedarray = ak.contents.IndexedArray.simplified(
        ak.index.Index64(cp.array([4, 3, 3, 8, 7, 6], cp.int64)),
        cuda_unionarray,
    )
    assert indexedarray.to_list() == ["one", 2.2, 2.2, 4.4, 3.3, "three"]


def test_1928_replace_simplify_method_with_classmethod_constructor_indexedoption_of_union():
    unionarray = ak.from_iter(
        [0.0, 1.1, "zero", 2.2, "one", "two", "three", 3.3, 4.4, 5.5, "four"],
        highlevel=False,
    )

    cuda_unionarray = ak.to_backend(unionarray, "cuda", highlevel=False)

    indexedoptionarray = ak.contents.IndexedOptionArray.simplified(
        ak.index.Index64(cp.array([-1, 4, 3, -1, 3, 8, 7, 6, -1], cp.int64)),
        cuda_unionarray,
    )
    assert indexedoptionarray.to_list() == [
        None,
        "one",
        2.2,
        None,
        2.2,
        4.4,
        3.3,
        "three",
        None,
    ]


def test_1928_replace_simplify_method_with_classmethod_constructor_indexedoption_of_union_of_option_1():
    with pytest.raises(
        TypeError, match=r" must either be comprised of entirely optional contents"
    ):
        ak.contents.UnionArray(
            ak.index.Index8(cp.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=cp.int8)),
            ak.index.Index64(
                cp.array([0, 1, 0, 2, 1, 2, 3, 3, 4, 5, 4], dtype=cp.int64)
            ),
            [
                ak.from_iter([0.0, 1.1, 2.2, 3.3, None, 5.5], highlevel=False),
                ak.from_iter(["zero", "one", "two", "three", "four"], highlevel=False),
            ],
        )

    unionarray = ak.contents.UnionArray.simplified(
        ak.index.Index8(np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 0, 2, 1, 2, 3, 3, 4, 5, 4], dtype=np.int64)),
        [
            ak.from_iter([0.0, 1.1, 2.2, 3.3, None, 5.5], highlevel=False),
            ak.from_iter(["zero", "one", "two", "three", "four"], highlevel=False),
        ],
    )

    cuda_unionarray = ak.to_backend(unionarray, "cuda", highlevel=False)

    indexedoptionarray = ak.contents.IndexedOptionArray.simplified(
        ak.index.Index64(cp.array([-1, 4, 3, -1, 3, 8, 7, 6, -1], cp.int64)),
        cuda_unionarray,
    )
    assert indexedoptionarray.to_list() == [
        None,
        "one",
        2.2,
        None,
        2.2,
        None,
        3.3,
        "three",
        None,
    ]


def test_1928_replace_simplify_method_with_classmethod_constructor_indexedoption_of_union_of_option_2():
    with pytest.raises(
        TypeError, match=r"must either be comprised of entirely optional contents"
    ):
        ak.contents.UnionArray(
            ak.index.Index8(cp.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=cp.int8)),
            ak.index.Index64(
                cp.array([0, 1, 0, 2, 1, 2, 3, 3, 4, 5, 4], dtype=cp.int64)
            ),
            [
                ak.from_iter([0.0, 1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
                ak.from_iter(["zero", None, "two", "three", "four"], highlevel=False),
            ],
        )

    unionarray = ak.contents.UnionArray.simplified(
        ak.index.Index8(np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 0, 2, 1, 2, 3, 3, 4, 5, 4], dtype=np.int64)),
        [
            ak.from_iter([0.0, 1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
            ak.from_iter(["zero", None, "two", "three", "four"], highlevel=False),
        ],
    )

    cuda_unionarray = ak.to_backend(unionarray, "cuda", highlevel=False)

    indexedoptionarray = ak.contents.IndexedOptionArray.simplified(
        ak.index.Index64(cp.array([-1, 4, 3, -1, 3, 8, 7, 6, -1], cp.int64)),
        cuda_unionarray,
    )
    assert indexedoptionarray.to_list() == [
        None,
        None,
        2.2,
        None,
        2.2,
        4.4,
        3.3,
        "three",
        None,
    ]


def test_1928_replace_simplify_method_with_classmethod_constructor_indexedoption_of_union_of_option_1_2():
    unionarray = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 0, 2, 1, 2, 3, 3, 4, 5, 4], dtype=np.int64)),
        [
            ak.from_iter([0.0, 1.1, 2.2, 3.3, None, 5.5], highlevel=False),
            ak.from_iter(["zero", None, "two", "three", "four"], highlevel=False),
        ],
    )

    cuda_unionarray = ak.to_backend(unionarray, "cuda", highlevel=False)

    indexedoptionarray = ak.contents.IndexedOptionArray.simplified(
        ak.index.Index64(cp.array([-1, 4, 3, -1, 3, 8, 7, 6, -1], cp.int64)),
        cuda_unionarray,
    )
    assert indexedoptionarray.to_list() == [
        None,
        None,
        2.2,
        None,
        2.2,
        None,
        3.3,
        "three",
        None,
    ]


def test_0028_add_dressed_types_highlevel():
    a = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    )
    cuda_a = ak.to_backend(a, "cuda")

    assert (
        repr(cuda_a)
        == "<Array [[1.1, 2.2, 3.3], [], ..., [7.7, 8.8, 9.9]] type='5 * var * float64'>"
    )
    assert str(cuda_a) == "[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]"

    b = ak.highlevel.Array(np.arange(100, dtype=np.int32), check_valid=True)
    cuda_b = ak.to_backend(b, "cuda")

    assert (
        repr(cuda_b)
        == "<Array [0, 1, 2, 3, 4, 5, 6, ..., 94, 95, 96, 97, 98, 99] type='100 * int32'>"
    )
    assert (
        str(cuda_b)
        == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ..., 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]"
    )

    c = ak.highlevel.Array(
        '[{"one": 3.14, "two": [1.1, 2.2]}, {"one": 99.9, "two": [-3.1415926]}]',
        check_valid=True,
    )
    cuda_c = ak.to_backend(c, "cuda")

    assert (
        repr(cuda_c)
        == "<Array [{one: 3.14, two: [...]}, {...}] type='2 * {one: float64, two: var *...'>"
    )
    assert (
        str(cuda_c) == "[{one: 3.14, two: [1.1, 2.2]}, {one: 99.9, two: [-3.1415926]}]"
    )


def test_0049_distinguish_record_and_recordarray_behaviors():
    class Pointy(ak.highlevel.Record):
        def __str__(self):
            return "<{} {}>".format(self["x"], self["y"])

    behavior = {}
    behavior["__typestr__", "Point"] = "P"
    behavior["Point"] = Pointy
    array = ak.highlevel.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ],
        with_name="Point",
        behavior=behavior,
        check_valid=True,
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert str(cuda_array[0][0]) == "<1 [1.1]>"
    assert repr(cuda_array[0]) == "<Array [<1 [1.1]>, <2 [2.0, 0.2]>] type='2 * P'>"
    assert (
        repr(cuda_array)
        == "<Array [[<1 [1.1]>, <2 [2.0, 0.2]>], ..., [{...}]] type='3 * var * P'>"
    )


def test_0074_argsort_and_sort_UnionArray_FIXME():
    content0 = ak.operations.from_iter(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False
    )

    content1 = ak.operations.from_iter(
        ["one", "two", "three", "four", "five"], highlevel=False
    )

    tags = ak.index.Index8([])
    index = ak.index.Index32([])
    array = ak.contents.UnionArray(tags, index, [content0, content1])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array) == []

    assert to_list(ak.operations.sort(cuda_array)) == []
    assert (
        to_list(
            ak.operations.argsort(
                cuda_array,
            )
        )
        == []
    )


def test_0078_argcross_and_cross_axis0():
    one = ak.Array([1.1, 2.2, 3.3])
    two = ak.Array([100, 200, 300, 400, 500])
    three = ak.Array(["a", "b"])
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")
    cuda_three = ak.to_backend(three, "cuda")

    assert to_list(ak.operations.cartesian([cuda_one], axis=0)) == [
        (1.1,),
        (2.2,),
        (3.3,),
    ]
    assert to_list(ak.operations.cartesian({"x": cuda_one}, axis=0)) == [
        {"x": 1.1},
        {"x": 2.2},
        {"x": 3.3},
    ]

    assert to_list(ak.operations.cartesian([cuda_one, cuda_two], axis=0)) == [
        (1.1, 100),
        (1.1, 200),
        (1.1, 300),
        (1.1, 400),
        (1.1, 500),
        (2.2, 100),
        (2.2, 200),
        (2.2, 300),
        (2.2, 400),
        (2.2, 500),
        (3.3, 100),
        (3.3, 200),
        (3.3, 300),
        (3.3, 400),
        (3.3, 500),
    ]
    assert to_list(ak.operations.cartesian({"x": cuda_one, "y": cuda_two}, axis=0)) == [
        {"x": 1.1, "y": 100},
        {"x": 1.1, "y": 200},
        {"x": 1.1, "y": 300},
        {"x": 1.1, "y": 400},
        {"x": 1.1, "y": 500},
        {"x": 2.2, "y": 100},
        {"x": 2.2, "y": 200},
        {"x": 2.2, "y": 300},
        {"x": 2.2, "y": 400},
        {"x": 2.2, "y": 500},
        {"x": 3.3, "y": 100},
        {"x": 3.3, "y": 200},
        {"x": 3.3, "y": 300},
        {"x": 3.3, "y": 400},
        {"x": 3.3, "y": 500},
    ]
    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0)
    ) == [
        (1.1, 100, "a"),
        (1.1, 100, "b"),
        (1.1, 200, "a"),
        (1.1, 200, "b"),
        (1.1, 300, "a"),
        (1.1, 300, "b"),
        (1.1, 400, "a"),
        (1.1, 400, "b"),
        (1.1, 500, "a"),
        (1.1, 500, "b"),
        (2.2, 100, "a"),
        (2.2, 100, "b"),
        (2.2, 200, "a"),
        (2.2, 200, "b"),
        (2.2, 300, "a"),
        (2.2, 300, "b"),
        (2.2, 400, "a"),
        (2.2, 400, "b"),
        (2.2, 500, "a"),
        (2.2, 500, "b"),
        (3.3, 100, "a"),
        (3.3, 100, "b"),
        (3.3, 200, "a"),
        (3.3, 200, "b"),
        (3.3, 300, "a"),
        (3.3, 300, "b"),
        (3.3, 400, "a"),
        (3.3, 400, "b"),
        (3.3, 500, "a"),
        (3.3, 500, "b"),
    ]

    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0, nested=[0])
    ) == [
        [
            (1.1, 100, "a"),
            (1.1, 100, "b"),
            (1.1, 200, "a"),
            (1.1, 200, "b"),
            (1.1, 300, "a"),
        ],
        [
            (1.1, 300, "b"),
            (1.1, 400, "a"),
            (1.1, 400, "b"),
            (1.1, 500, "a"),
            (1.1, 500, "b"),
        ],
        [
            (2.2, 100, "a"),
            (2.2, 100, "b"),
            (2.2, 200, "a"),
            (2.2, 200, "b"),
            (2.2, 300, "a"),
        ],
        [
            (2.2, 300, "b"),
            (2.2, 400, "a"),
            (2.2, 400, "b"),
            (2.2, 500, "a"),
            (2.2, 500, "b"),
        ],
        [
            (3.3, 100, "a"),
            (3.3, 100, "b"),
            (3.3, 200, "a"),
            (3.3, 200, "b"),
            (3.3, 300, "a"),
        ],
        [
            (3.3, 300, "b"),
            (3.3, 400, "a"),
            (3.3, 400, "b"),
            (3.3, 500, "a"),
            (3.3, 500, "b"),
        ],
    ]
    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0, nested=[1])
    ) == [
        [(1.1, 100, "a"), (1.1, 100, "b")],
        [(1.1, 200, "a"), (1.1, 200, "b")],
        [(1.1, 300, "a"), (1.1, 300, "b")],
        [(1.1, 400, "a"), (1.1, 400, "b")],
        [(1.1, 500, "a"), (1.1, 500, "b")],
        [(2.2, 100, "a"), (2.2, 100, "b")],
        [(2.2, 200, "a"), (2.2, 200, "b")],
        [(2.2, 300, "a"), (2.2, 300, "b")],
        [(2.2, 400, "a"), (2.2, 400, "b")],
        [(2.2, 500, "a"), (2.2, 500, "b")],
        [(3.3, 100, "a"), (3.3, 100, "b")],
        [(3.3, 200, "a"), (3.3, 200, "b")],
        [(3.3, 300, "a"), (3.3, 300, "b")],
        [(3.3, 400, "a"), (3.3, 400, "b")],
        [(3.3, 500, "a"), (3.3, 500, "b")],
    ]
    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0, nested=[0, 1])
    ) == [
        [
            [(1.1, 100, "a"), (1.1, 100, "b")],
            [(1.1, 200, "a"), (1.1, 200, "b")],
            [(1.1, 300, "a"), (1.1, 300, "b")],
            [(1.1, 400, "a"), (1.1, 400, "b")],
            [(1.1, 500, "a"), (1.1, 500, "b")],
        ],
        [
            [(2.2, 100, "a"), (2.2, 100, "b")],
            [(2.2, 200, "a"), (2.2, 200, "b")],
            [(2.2, 300, "a"), (2.2, 300, "b")],
            [(2.2, 400, "a"), (2.2, 400, "b")],
            [(2.2, 500, "a"), (2.2, 500, "b")],
        ],
        [
            [(3.3, 100, "a"), (3.3, 100, "b")],
            [(3.3, 200, "a"), (3.3, 200, "b")],
            [(3.3, 300, "a"), (3.3, 300, "b")],
            [(3.3, 400, "a"), (3.3, 400, "b")],
            [(3.3, 500, "a"), (3.3, 500, "b")],
        ],
    ]

    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0, nested=[])
    ) == to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0, nested=False)
    )
    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0, nested=[])
    ) == to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0, nested=None)
    )
    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0, nested=[0, 1])
    ) == to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], axis=0, nested=True)
    )


def test_0078_argcross_and_cross_axis1():
    one = ak.Array([[0, 1, 2], [], [3, 4]])
    two = ak.Array([[100, 200], [300], [400, 500]])
    three = ak.Array([["a", "b"], ["c", "d"], ["e"]])
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")
    cuda_three = ak.to_backend(three, "cuda")

    assert to_list(ak.operations.cartesian([cuda_one])) == [
        [(0,), (1,), (2,)],
        [],
        [(3,), (4,)],
    ]
    assert to_list(ak.operations.cartesian({"x": one})) == [
        [{"x": 0}, {"x": 1}, {"x": 2}],
        [],
        [{"x": 3}, {"x": 4}],
    ]

    assert to_list(ak.operations.cartesian([cuda_one, cuda_two])) == [
        [(0, 100), (0, 200), (1, 100), (1, 200), (2, 100), (2, 200)],
        [],
        [(3, 400), (3, 500), (4, 400), (4, 500)],
    ]
    assert to_list(ak.operations.cartesian({"x": one, "y": two})) == [
        [
            {"x": 0, "y": 100},
            {"x": 0, "y": 200},
            {"x": 1, "y": 100},
            {"x": 1, "y": 200},
            {"x": 2, "y": 100},
            {"x": 2, "y": 200},
        ],
        [],
        [
            {"x": 3, "y": 400},
            {"x": 3, "y": 500},
            {"x": 4, "y": 400},
            {"x": 4, "y": 500},
        ],
    ]

    assert to_list(ak.operations.cartesian([cuda_one, cuda_two, cuda_three])) == [
        [
            (0, 100, "a"),
            (0, 100, "b"),
            (0, 200, "a"),
            (0, 200, "b"),
            (1, 100, "a"),
            (1, 100, "b"),
            (1, 200, "a"),
            (1, 200, "b"),
            (2, 100, "a"),
            (2, 100, "b"),
            (2, 200, "a"),
            (2, 200, "b"),
        ],
        [],
        [(3, 400, "e"), (3, 500, "e"), (4, 400, "e"), (4, 500, "e")],
    ]

    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], nested=[0])
    ) == [
        [
            [(0, 100, "a"), (0, 100, "b"), (0, 200, "a"), (0, 200, "b")],
            [(1, 100, "a"), (1, 100, "b"), (1, 200, "a"), (1, 200, "b")],
            [(2, 100, "a"), (2, 100, "b"), (2, 200, "a"), (2, 200, "b")],
        ],
        [],
        [[(3, 400, "e"), (3, 500, "e")], [(4, 400, "e"), (4, 500, "e")]],
    ]
    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], nested=[1])
    ) == [
        [
            [(0, 100, "a"), (0, 100, "b")],
            [(0, 200, "a"), (0, 200, "b")],
            [(1, 100, "a"), (1, 100, "b")],
            [(1, 200, "a"), (1, 200, "b")],
            [(2, 100, "a"), (2, 100, "b")],
            [(2, 200, "a"), (2, 200, "b")],
        ],
        [],
        [[(3, 400, "e")], [(3, 500, "e")], [(4, 400, "e")], [(4, 500, "e")]],
    ]
    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], nested=[0, 1])
    ) == [
        [
            [[(0, 100, "a"), (0, 100, "b")], [(0, 200, "a"), (0, 200, "b")]],
            [[(1, 100, "a"), (1, 100, "b")], [(1, 200, "a"), (1, 200, "b")]],
            [[(2, 100, "a"), (2, 100, "b")], [(2, 200, "a"), (2, 200, "b")]],
        ],
        [],
        [[[(3, 400, "e")], [(3, 500, "e")]], [[(4, 400, "e")], [(4, 500, "e")]]],
    ]

    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], nested=[])
    ) == to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], nested=False)
    )
    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], nested=[])
    ) == to_list(ak.operations.cartesian([cuda_one, cuda_two, cuda_three], nested=None))
    assert to_list(
        ak.operations.cartesian([cuda_one, cuda_two, cuda_three], nested=[0, 1])
    ) == to_list(ak.operations.cartesian([cuda_one, cuda_two, cuda_three], nested=True))


def test_0078_argcross_and_cross_axis2():
    one = ak.Array([[[0, 1, 2], [], [3, 4]], [[0, 1, 2], [], [3, 4]]])
    two = ak.Array([[[100, 200], [300], [400, 500]], [[100, 200], [300], [400, 500]]])
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")

    assert to_list(ak.operations.cartesian([cuda_one, cuda_two], axis=2)) == [
        [
            [(0, 100), (0, 200), (1, 100), (1, 200), (2, 100), (2, 200)],
            [],
            [(3, 400), (3, 500), (4, 400), (4, 500)],
        ],
        [
            [(0, 100), (0, 200), (1, 100), (1, 200), (2, 100), (2, 200)],
            [],
            [(3, 400), (3, 500), (4, 400), (4, 500)],
        ],
    ]


def test_0078_argcross_and_cross_localindex():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(ak._do.local_index(cuda_array, 0)) == [0, 1, 2, 3, 4]
    assert to_list(ak._do.local_index(cuda_array, 1)) == [
        [0, 1, 2],
        [],
        [0, 1],
        [0],
        [0, 1, 2, 3],
    ]

    array = ak.operations.from_iter(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[6.6, 7.7, 8.8, 9.9]]],
        highlevel=False,
    )
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(ak._do.local_index(cuda_array, 0)) == [0, 1, 2, 3]
    assert to_list(ak._do.local_index(cuda_array, 1)) == [[0, 1, 2], [], [0], [0]]
    assert to_list(ak._do.local_index(cuda_array, 2)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        [[0]],
        [[0, 1, 2, 3]],
    ]

    array = ak.operations.from_numpy(
        np.arange(2 * 3 * 5).reshape(2, 3, 5), regulararray=True, highlevel=False
    )
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(ak._do.local_index(cuda_array, 0)) == [0, 1]
    assert to_list(ak._do.local_index(cuda_array, 1)) == [[0, 1, 2], [0, 1, 2]]
    assert to_list(ak._do.local_index(cuda_array, 2)) == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
    ]


def test_0078_argcross_and_cross_argcartesian():
    one = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4]])
    two = ak.Array([[100, 200], [300], [400, 500]])
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")

    assert to_list(ak.operations.argcartesian([cuda_one, cuda_two])) == [
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        [],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ]
    assert to_list(ak.operations.argcartesian({"x": one, "y": two})) == [
        [
            {"x": 0, "y": 0},
            {"x": 0, "y": 1},
            {"x": 1, "y": 0},
            {"x": 1, "y": 1},
            {"x": 2, "y": 0},
            {"x": 2, "y": 1},
        ],
        [],
        [{"x": 0, "y": 0}, {"x": 0, "y": 1}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
    ]


def test_0078_argcross_and_cross_argcartesian_negative_axis():
    one = ak.Array([[["a", "b"], []], [], [["c"]]])
    two = ak.Array([[[1.1], []], [], [[2.2, 3.3]]])
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")

    assert ak.cartesian([cuda_one, cuda_two], axis=-1).to_list() == [
        [[("a", 1.1), ("b", 1.1)], []],
        [],
        [[("c", 2.2), ("c", 3.3)]],
    ]

    assert ak.argcartesian([cuda_one, cuda_two], axis=-1).to_list() == [
        [[(0, 0), (1, 0)], []],
        [],
        [[(0, 0), (0, 1)]],
    ]


def test_0093_simplify_uniontypes_and_optiontypes_concatenate():
    one = ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True).layout
    two = ak.highlevel.Array([[], [1], [2, 2], [3, 3, 3]], check_valid=True).layout
    three = ak.highlevel.Array(
        [True, False, False, True, True], check_valid=True
    ).layout
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")
    cuda_three = ak.to_backend(three, "cuda")

    assert to_list(ak.operations.concatenate([cuda_one, cuda_two, cuda_three])) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        [],
        [1],
        [2, 2],
        [3, 3, 3],
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    assert isinstance(
        ak.operations.concatenate([cuda_one, cuda_two, cuda_three], highlevel=False),
        ak.contents.unionarray.UnionArray,
    )
    assert (
        len(
            ak.operations.concatenate(
                [cuda_one, cuda_two, cuda_three], highlevel=False
            ).contents
        )
        == 2
    )


def test_0093_simplify_uniontypes_and_optiontypes_indexedarray_merge():
    content1 = ak.operations.from_iter(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False
    )
    content2 = ak.operations.from_iter([[1, 2], [], [3, 4]], highlevel=False)
    index1 = ak.index.Index64(np.array([2, 0, -1, 0, 1, 2], dtype=np.int64))

    cuda_content2 = ak.to_backend(content2, "cuda", highlevel=False)
    indexedarray1 = ak.contents.IndexedOptionArray(index1, content1)
    cuda_indexedarray1 = ak.to_backend(indexedarray1, "cuda", highlevel=False)

    assert to_list(cuda_indexedarray1) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]

    assert to_list(cuda_indexedarray1._mergemany([cuda_content2])) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [1.0, 2.0],
        [],
        [3.0, 4.0],
    ]
    assert to_list(cuda_content2._mergemany([cuda_indexedarray1])) == [
        [1.0, 2.0],
        [],
        [3.0, 4.0],
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert to_list(cuda_indexedarray1._mergemany([cuda_indexedarray1])) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]

    assert (
        cuda_indexedarray1.to_typetracer()
        ._mergemany([cuda_content2.to_typetracer()])
        .form
        == cuda_indexedarray1._mergemany([cuda_content2]).form
    )
    assert (
        cuda_content2.to_typetracer()
        ._mergemany([cuda_indexedarray1.to_typetracer()])
        .form
        == cuda_content2._mergemany([cuda_indexedarray1]).form
    )
    assert (
        cuda_indexedarray1.to_typetracer()
        ._mergemany([cuda_indexedarray1.to_typetracer()])
        .form
        == cuda_indexedarray1._mergemany([cuda_indexedarray1]).form
    )


def test_0093_simplify_uniontypes_and_optiontypes_indexedarray_simplify():
    array = ak.operations.from_iter(
        ["one", "two", None, "three", None, None, "four", "five"], highlevel=False
    )
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    index2 = ak.index.Index64(np.array([2, 2, 1, 6, 5], dtype=np.int64))

    array2 = ak.contents.IndexedArray.simplified(index2, array)
    cuda_array2 = ak.to_backend(array2, "cuda", highlevel=False)

    assert cp.asarray(cuda_array.index.data).tolist() == [0, 1, -1, 2, -1, -1, 3, 4]
    assert (
        to_list(cuda_array2)
        == to_list(cuda_array2)
        == [None, None, "two", "four", None]
    )

    assert cuda_array2.to_typetracer().form == cuda_array2.form


def test_0093_simplify_uniontypes_and_optiontypes_where():
    condition = ak.highlevel.Array(
        [True, False, True, False, True],
        check_valid=True,
    )
    one = ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)
    two = ak.highlevel.Array([False, False, False, True, True], check_valid=True)
    three = ak.highlevel.Array(
        [[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4]], check_valid=True
    )

    cuda_condition = ak.to_backend(condition, "cuda")
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")
    cuda_three = ak.to_backend(three, "cuda")

    assert to_list(ak.operations.where(cuda_condition, cuda_one, cuda_two)) == [
        1.1,
        0.0,
        3.3,
        1.0,
        5.5,
    ]
    assert to_list(ak.operations.where(cuda_condition, cuda_one, cuda_three)) == [
        1.1,
        [1],
        3.3,
        [3, 3, 3],
        5.5,
    ]


def test_0093_simplify_uniontypes_and_optiontypes_unionarray_simplify_one():
    one = ak.operations.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = ak.operations.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = ak.operations.from_iter([1.1, 2.2, 3.3], highlevel=False)
    tags = ak.index.Index8(
        np.array([0, 0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 0], dtype=np.int8)
    )
    index = ak.index.Index64(
        np.array([0, 1, 0, 0, 1, 2, 1, 2, 3, 3, 2, 4], dtype=np.int64)
    )
    array = ak.contents.UnionArray.simplified(tags, index, [one, two, three])

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(cuda_array) == [
        5,
        4,
        [],
        1.1,
        [1],
        3,
        2.2,
        [2, 2],
        [3, 3, 3],
        2,
        3.3,
        1,
    ]
    assert len(cuda_array.contents) == 2
    assert cuda_array.to_typetracer().form == cuda_array.form


def test_0093_simplify_uniontypes_and_optiontypes_unionarray_simplify():
    one = ak.operations.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = ak.operations.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = ak.operations.from_iter([1.1, 2.2, 3.3], highlevel=False)

    tags2 = ak.index.Index8(np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int8))
    index2 = ak.index.Index32(np.array([0, 0, 1, 1, 2, 3, 2], dtype=np.int32))
    inner = ak.contents.UnionArray(tags2, index2, [two, three])
    tags1 = ak.index.Index8(
        np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.int8)
    )
    index1 = ak.index.Index64(
        np.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=np.int64)
    )
    outer = ak.contents.UnionArray.simplified(tags1, index1, [one, inner])
    cuda_outer = ak.to_backend(outer, "cuda", highlevel=False)

    assert to_list(cuda_outer) == [
        5,
        4,
        [],
        1.1,
        [1],
        3,
        2.2,
        [2, 2],
        [3, 3, 3],
        2,
        3.3,
        1,
    ]

    assert isinstance(cuda_outer.content(0), ak.contents.numpyarray.NumpyArray)
    assert isinstance(
        cuda_outer.content(1), ak.contents.listoffsetarray.ListOffsetArray
    )
    assert len(cuda_outer.contents) == 2
    assert cuda_outer.to_typetracer().form == cuda_outer.form

    tags2 = ak.index.Index8(np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int8))
    index2 = ak.index.Index64(np.array([0, 0, 1, 1, 2, 3, 2], dtype=np.int64))
    inner = ak.contents.UnionArray(tags2, index2, [two, three])
    tags1 = ak.index.Index8(
        np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=np.int8)
    )
    index1 = ak.index.Index32(
        np.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=np.int32)
    )
    outer = ak.contents.UnionArray.simplified(tags1, index1, [inner, one])
    cuda_outer = ak.to_backend(outer, "cuda", highlevel=False)

    assert to_list(cuda_outer) == [
        5,
        4,
        [],
        1.1,
        [1],
        3,
        2.2,
        [2, 2],
        [3, 3, 3],
        2,
        3.3,
        1,
    ]


def test_0093_simplify_uniontypes_and_optiontypes_merge_parameters():
    one = ak.operations.from_iter(
        [[121, 117, 99, 107, 121], [115, 116, 117, 102, 102]], highlevel=False
    )
    two = ak.operations.from_iter(["good", "stuff"], highlevel=False)
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")

    assert to_list(ak.operations.concatenate([cuda_one, cuda_two])) == [
        [121, 117, 99, 107, 121],
        [115, 116, 117, 102, 102],
        "good",
        "stuff",
    ]
    assert to_list(ak.operations.concatenate([cuda_two, cuda_one])) == [
        "good",
        "stuff",
        [121, 117, 99, 107, 121],
        [115, 116, 117, 102, 102],
    ]

    assert (
        ak.operations.concatenate([cuda_one, cuda_two], highlevel=False)
        .to_typetracer()
        .form
        == ak.operations.concatenate([cuda_one, cuda_two], highlevel=False).form
    )
    assert (
        ak.operations.concatenate([cuda_two, cuda_one], highlevel=False)
        .to_typetracer()
        .form
        == ak.operations.concatenate([cuda_two, cuda_one], highlevel=False).form
    )


def test_0093_simplify_uniontypes_and_optiontypes_unionarray_merge():
    emptyarray = ak.contents.EmptyArray()

    one = ak.operations.from_iter([0.0, 1.1, 2.2, [], [1], [2, 2]], highlevel=False)
    two = ak.operations.from_iter(
        [{"x": 1, "y": 1.1}, 999, 123, {"x": 2, "y": 2.2}], highlevel=False
    )
    three = ak.operations.from_iter(["one", "two", "three"], highlevel=False)

    cuda_emptyarray = ak.to_backend(emptyarray, "cuda", highlevel=False)
    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two])) == [
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
    ]
    assert to_list(cuda_two._mergemany([cuda_one])) == [
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
    ]

    assert to_list(cuda_one._mergemany([cuda_emptyarray])) == [
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
    ]
    assert to_list(cuda_emptyarray._mergemany([cuda_one])) == [
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
    ]

    assert to_list(cuda_one._mergemany([cuda_three])) == [
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
        "one",
        "two",
        "three",
    ]
    assert to_list(cuda_two._mergemany([cuda_three])) == [
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
        "one",
        "two",
        "three",
    ]
    assert to_list(cuda_three._mergemany([cuda_one])) == [
        "one",
        "two",
        "three",
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
    ]
    assert to_list(cuda_three._mergemany([cuda_two])) == [
        "one",
        "two",
        "three",
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
    ]

    assert (
        cuda_one.to_typetracer()._mergemany([cuda_two.to_typetracer()]).form
        == cuda_one._mergemany([cuda_two]).form
    )
    assert (
        cuda_two.to_typetracer()._mergemany([cuda_one.to_typetracer()]).form
        == cuda_two._mergemany([cuda_one]).form
    )
    assert (
        cuda_one.to_typetracer()._mergemany([cuda_emptyarray.to_typetracer()]).form
        == cuda_one._mergemany([cuda_emptyarray]).form
    )
    assert (
        cuda_emptyarray.to_typetracer()._mergemany([cuda_one.to_typetracer()]).form
        == cuda_emptyarray._mergemany([cuda_one]).form
    )
    assert (
        cuda_one.to_typetracer()._mergemany([cuda_three.to_typetracer()]).form
        == cuda_one._mergemany([cuda_three]).form
    )
    assert (
        cuda_two.to_typetracer()._mergemany([cuda_three.to_typetracer()]).form
        == cuda_two._mergemany([cuda_three]).form
    )
    assert (
        cuda_three.to_typetracer()._mergemany([cuda_one.to_typetracer()]).form
        == cuda_three._mergemany([cuda_one]).form
    )
    assert (
        cuda_three.to_typetracer()._mergemany([cuda_two.to_typetracer()]).form
        == cuda_three._mergemany([cuda_two]).form
    )


def test_0093_simplify_uniontypes_and_optiontypes_numpyarray_merge():
    emptyarray = ak.contents.EmptyArray()

    np1 = np.arange(2 * 7 * 5).reshape(2, 7, 5)
    np2 = np.arange(3 * 7 * 5).reshape(3, 7, 5)

    cuda_emptyarray = ak.to_backend(emptyarray, "cuda", highlevel=False)
    cuda_np1 = ak.to_backend(np1, "cuda")
    cuda_np2 = ak.to_backend(np2, "cuda")
    cuda_ak1 = ak.contents.NumpyArray(cuda_np1)
    cuda_ak2 = ak.contents.NumpyArray(cuda_np2)

    assert to_list(cuda_ak1._mergemany([cuda_ak2])) == to_list(
        np.concatenate([cuda_np1, cuda_np2])
    )
    assert to_list(
        cuda_ak1[1:, :-1, ::-1]._mergemany([cuda_ak2[1:, :-1, ::-1]])
    ) == to_list(np.concatenate([cuda_np1[1:, :-1, ::-1], cuda_np2[1:, :-1, ::-1]]))
    assert (
        cuda_ak1.to_typetracer()._mergemany([cuda_ak2.to_typetracer()]).form
        == cuda_ak1._mergemany([cuda_ak2]).form
    )
    assert (
        cuda_ak1[1:, :-1, ::-1]
        .to_typetracer()
        ._mergemany([cuda_ak2[1:, :-1, ::-1].to_typetracer()])
        .form
        == cuda_ak1[1:, :-1, ::-1]._mergemany([cuda_ak2[1:, :-1, ::-1]]).form
    )

    for x in [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ]:
        for y in [
            np.bool_,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
        ]:
            z = np.concatenate(
                [np.array([1, 2, 3], dtype=x), np.array([4, 5], dtype=y)]
            ).dtype.type
            one = ak.contents.NumpyArray(np.array([1, 2, 3], dtype=x))
            two = ak.contents.NumpyArray(np.array([4, 5], dtype=y))
            cuda_one = ak.to_backend(one, "cuda", highlevel=False)
            cuda_two = ak.to_backend(two, "cuda", highlevel=False)

            cuda_three = cuda_one._mergemany([cuda_two])
            assert ak.to_numpy(cuda_three).dtype == np.dtype(z), (
                f"{x} {y} {z} {ak.to_numpy(cuda_three).dtype.type}"
            )
            assert to_list(cuda_three) == to_list(
                np.concatenate([ak.to_numpy(cuda_one), ak.to_numpy(two)])
            )
            assert to_list(cuda_one._mergemany([cuda_emptyarray])) == to_list(cuda_one)
            assert to_list(cuda_emptyarray._mergemany([cuda_one])) == to_list(cuda_one)

            assert (
                cuda_one.to_typetracer()._mergemany([cuda_two.to_typetracer()]).form
                == cuda_one._mergemany([cuda_two]).form
            )
            assert (
                cuda_one.to_typetracer()
                ._mergemany([cuda_emptyarray.to_typetracer()])
                .form
                == cuda_one._mergemany([cuda_emptyarray]).form
            )
            assert (
                cuda_emptyarray.to_typetracer()
                ._mergemany([cuda_one.to_typetracer()])
                .form
                == cuda_emptyarray._mergemany([cuda_one]).form
            )


def test_0093_simplify_uniontypes_and_optiontypes_regulararray_merge():
    emptyarray = ak.contents.EmptyArray()

    np1 = np.arange(2 * 7 * 5).reshape(2, 7, 5)
    np2 = np.arange(3 * 7 * 5).reshape(3, 7, 5)

    cuda_emptyarray = ak.to_backend(emptyarray, "cuda", highlevel=False)
    cuda_np1 = ak.to_backend(np1, "cuda")
    cuda_np2 = ak.to_backend(np2, "cuda")
    cuda_ak1 = ak.operations.from_iter(cuda_np1, highlevel=False)
    cuda_ak2 = ak.operations.from_iter(cuda_np2, highlevel=False)

    assert to_list(cuda_ak1._mergemany([cuda_ak2])) == to_list(
        np.concatenate([cuda_np1, cuda_np2])
    )
    assert to_list(cuda_ak1._mergemany([cuda_emptyarray])) == to_list(cuda_ak1)
    assert to_list(cuda_emptyarray._mergemany([cuda_ak1])) == to_list(cuda_ak1)

    assert (
        cuda_ak1.to_typetracer()._mergemany([cuda_ak2.to_typetracer()]).form
        == cuda_ak1._mergemany([cuda_ak2]).form
    )
    assert (
        cuda_ak1.to_typetracer()._mergemany([cuda_emptyarray.to_typetracer()]).form
        == cuda_ak1._mergemany([cuda_emptyarray]).form
    )
    assert (
        cuda_emptyarray.to_typetracer()._mergemany([cuda_ak1.to_typetracer()]).form
        == cuda_emptyarray._mergemany([cuda_ak1]).form
    )


def test_0093_simplify_uniontypes_and_optiontypes_mask_as_bool():
    array = ak.operations.from_iter(
        ["one", "two", None, "three", None, None, "four"], highlevel=False
    )
    index2 = ak.index.Index64(np.array([2, 2, 1, 5, 0], dtype=np.int64))
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_index2 = ak.to_backend(index2, "cuda", highlevel=False)

    cuda_array2 = ak.contents.IndexedArray.simplified(cuda_index2, cuda_array)

    assert cp.asarray(
        cuda_array.mask_as_bool(valid_when=False).view(cp.int8)
    ).tolist() == [
        0,
        0,
        1,
        0,
        1,
        1,
        0,
    ]
    assert cp.asarray(
        cuda_array2.mask_as_bool(valid_when=False).view(cp.int8)
    ).tolist() == [
        1,
        1,
        0,
        1,
        0,
    ]


def test_0093_simplify_uniontypes_and_optiontypes_listarray_merge():
    emptyarray = ak.contents.EmptyArray()
    cuda_emptyarray = ak.to_backend(emptyarray, "cuda", highlevel=False)

    content1 = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    content2 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7]))

    for (dtype1, Index1, ListArray1), (dtype2, Index2, ListArray2) in [
        (
            (np.int32, ak.index.Index32, ak.contents.ListArray),
            (np.int32, ak.index.Index32, ak.contents.ListArray),
        ),
        (
            (np.int32, ak.index.Index32, ak.contents.ListArray),
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
        ),
        (
            (np.int32, ak.index.Index32, ak.contents.ListArray),
            (np.int64, ak.index.Index64, ak.contents.ListArray),
        ),
        (
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
            (np.int32, ak.index.Index32, ak.contents.ListArray),
        ),
        (
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
        ),
        (
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
            (np.int64, ak.index.Index64, ak.contents.ListArray),
        ),
        (
            (np.int64, ak.index.Index64, ak.contents.ListArray),
            (np.int32, ak.index.Index32, ak.contents.ListArray),
        ),
        (
            (np.int64, ak.index.Index64, ak.contents.ListArray),
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
        ),
        (
            (np.int64, ak.index.Index64, ak.contents.ListArray),
            (np.int64, ak.index.Index64, ak.contents.ListArray),
        ),
    ]:
        starts1 = Index1(np.array([0, 3, 3], dtype=dtype1))
        stops1 = Index1(np.array([3, 3, 5], dtype=dtype1))
        starts2 = Index2(np.array([2, 99, 0], dtype=dtype2))
        stops2 = Index2(np.array([6, 99, 3], dtype=dtype2))
        array1 = ListArray1(starts1, stops1, content1)
        array2 = ListArray2(starts2, stops2, content2)

        cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)
        cuda_array2 = ak.to_backend(array2, "cuda", highlevel=False)

        assert to_list(cuda_array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert to_list(cuda_array2) == [[3, 4, 5, 6], [], [1, 2, 3]]

        assert to_list(cuda_array1._mergemany([cuda_array2])) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [3, 4, 5, 6],
            [],
            [1, 2, 3],
        ]
        assert to_list(cuda_array2._mergemany([cuda_array1])) == [
            [3, 4, 5, 6],
            [],
            [1, 2, 3],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]
        assert to_list(cuda_array1._mergemany([cuda_emptyarray])) == to_list(
            cuda_array1
        )
        assert to_list(cuda_emptyarray._mergemany([cuda_array1])) == to_list(
            cuda_array1
        )

        assert (
            cuda_array1.to_typetracer()._mergemany([cuda_array2.to_typetracer()]).form
            == cuda_array1._mergemany([cuda_array2]).form
        )
        assert (
            cuda_array2.to_typetracer()._mergemany([cuda_array1.to_typetracer()]).form
            == cuda_array2._mergemany([cuda_array1]).form
        )
        assert (
            cuda_array1.to_typetracer()
            ._mergemany([cuda_emptyarray.to_typetracer()])
            .form
            == cuda_array1._mergemany([cuda_emptyarray]).form
        )
        assert (
            cuda_emptyarray.to_typetracer()
            ._mergemany([cuda_array1.to_typetracer()])
            .form
            == cuda_emptyarray._mergemany([cuda_array1]).form
        )

    regulararray = ak.contents.RegularArray(content2, 2, zeros_length=0)
    cuda_regulararray = ak.to_backend(regulararray, "cuda", highlevel=False)

    assert to_list(cuda_regulararray) == [[1, 2], [3, 4], [5, 6]]
    assert to_list(cuda_regulararray._mergemany([cuda_emptyarray])) == to_list(
        cuda_regulararray
    )
    assert to_list(cuda_emptyarray._mergemany([cuda_regulararray])) == to_list(
        cuda_regulararray
    )

    for dtype1, Index1, ListArray1 in [
        (np.int32, ak.index.Index32, ak.contents.ListArray),
        (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
        (np.int64, ak.index.Index64, ak.contents.ListArray),
    ]:
        starts1 = Index1(np.array([0, 3, 3], dtype=dtype1))
        stops1 = Index1(np.array([3, 3, 5], dtype=dtype1))
        array1 = ListArray1(starts1, stops1, content1)
        cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)

        assert to_list(cuda_array1._mergemany([cuda_regulararray])) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        assert to_list(cuda_regulararray._mergemany([cuda_array1])) == [
            [1, 2],
            [3, 4],
            [5, 6],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]


def test_0193_is_none_axis_parameter():
    one = ak.operations.from_iter([1, None, 3], highlevel=False)
    two = ak.operations.from_iter([[], [1], None, [3, 3, 3]], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 1, 0, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int64))
    array = ak.Array(ak.contents.UnionArray(tags, index, [one, two]), check_valid=True)
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.to_list(cuda_array) == [1, [], [1], None, 3, None, [3, 3, 3]]

    assert ak.to_list(ak.operations.is_none(cuda_array)) == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
    ]


def test_0023_regular_array_getitem():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    regulararray = ak.contents.RegularArray(listoffsetarray, 2, zeros_length=0)

    cuda_regulararray = ak.to_backend(regulararray, "cuda", highlevel=False)

    assert to_list(cuda_regulararray[(0)]) == [[0.0, 1.1, 2.2], []]
    assert to_list(cuda_regulararray[(1)]) == [[3.3, 4.4], [5.5]]
    assert to_list(cuda_regulararray[(2)]) == [[6.6, 7.7, 8.8, 9.9], []]
    assert cuda_regulararray.to_typetracer()[(2)].form == cuda_regulararray[(2)].form
    assert to_list(cuda_regulararray[(slice(1, None, None))]) == [
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        cuda_regulararray.to_typetracer()[(slice(1, None, None))].form
        == cuda_regulararray[(slice(1, None, None))].form
    )
    assert to_list(cuda_regulararray[(slice(None, -1, None))]) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
    ]
    assert (
        cuda_regulararray.to_typetracer()[(slice(None, -1, None))].form
        == cuda_regulararray[(slice(None, -1, None))].form
    )
