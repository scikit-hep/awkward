# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_record_nochange():
    class Point(ak._v2.Record):
        def __getitem__(self, where):
            return super().__getitem__(where)

    array = ak._v2.Array(
        [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]],
        with_name="point",
        behavior={"point": Point},
    )

    assert array.tolist() == [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]]
    assert array[0].tolist() == [{"rho": 1, "phi": 1.0}]
    assert array[0, 0].tolist() == {"rho": 1, "phi": 1.0}


def test_array_nochange():
    class PointArray(ak._v2.Array):
        def __getitem__(self, where):
            return super().__getitem__(where)

    array = ak._v2.Array(
        [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]],
        with_name="point",
        behavior={("*", "point"): PointArray},
    )

    assert array.tolist() == [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]]
    assert array[0].tolist() == [{"rho": 1, "phi": 1.0}]
    assert array[0, 0].tolist() == {"rho": 1, "phi": 1.0}


def test_record_changed():
    class Point(ak._v2.Record):
        def __getitem__(self, where):
            return str(super().__getitem__(where))

    array = ak._v2.Array(
        [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]],
        with_name="point",
        behavior={"point": Point},
    )

    assert array.tolist() == [
        [{"rho": "1", "phi": "1.0"}],
        [],
        [{"rho": "2", "phi": "2.0"}],
    ]
    assert array[0].tolist() == [{"rho": "1", "phi": "1.0"}]
    assert array[0, 0].tolist() == {"rho": "1", "phi": "1.0"}


def test_array_changed():
    class PointArray(ak._v2.Array):
        def __getitem__(self, where):
            return str(super().__getitem__(where))

    array = ak._v2.Array(
        [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]],
        with_name="point",
        behavior={("*", "point"): PointArray},
    )

    assert array.tolist() == ["[{rho: 1, phi: 1}]", "[]", "[{rho: 2, phi: 2}]"]
    assert array[0] == "[{rho: 1, phi: 1}]"
    assert array[0, 0] == "{rho: 1, phi: 1}"


def test_record_to_Array():
    class Point(ak._v2.Record):
        def __getitem__(self, where):
            return ak._v2.Array([1, 2, 3])

    array = ak._v2.Array(
        [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]],
        with_name="point",
        behavior={"point": Point},
    )

    assert array.tolist() == [
        [{"rho": [1, 2, 3], "phi": [1, 2, 3]}],
        [],
        [{"rho": [1, 2, 3], "phi": [1, 2, 3]}],
    ]
    assert array[0].tolist() == [{"rho": [1, 2, 3], "phi": [1, 2, 3]}]
    assert array[0, 0].tolist() == {"rho": [1, 2, 3], "phi": [1, 2, 3]}


def test_array_to_Array():
    class PointArray(ak._v2.Array):
        def __getitem__(self, where):
            return ak._v2.Array([1, 2, 3])

    array = ak._v2.Array(
        [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]],
        with_name="point",
        behavior={("*", "point"): PointArray},
    )

    assert array.tolist() == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert array[0].tolist() == [1, 2, 3]
    assert array[0, 0].tolist() == [1, 2, 3]


def test_record_to_ndarray():
    class Point(ak._v2.Record):
        def __getitem__(self, where):
            return np.array([1, 2, 3])

    array = ak._v2.Array(
        [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]],
        with_name="point",
        behavior={"point": Point},
    )

    assert array.tolist() == [
        [{"rho": [1, 2, 3], "phi": [1, 2, 3]}],
        [],
        [{"rho": [1, 2, 3], "phi": [1, 2, 3]}],
    ]
    assert array[0].tolist() == [{"rho": [1, 2, 3], "phi": [1, 2, 3]}]
    assert array[0, 0].tolist() == {"rho": [1, 2, 3], "phi": [1, 2, 3]}


def test_array_to_ndarray():
    class PointArray(ak._v2.Array):
        def __getitem__(self, where):
            return np.array([1, 2, 3])

    array = ak._v2.Array(
        [[{"rho": 1, "phi": 1.0}], [], [{"rho": 2, "phi": 2.0}]],
        with_name="point",
        behavior={("*", "point"): PointArray},
    )

    assert array.tolist() == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert array[0].tolist() == [1, 2, 3]
    assert array[0, 0].tolist() == [1, 2, 3]
