# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_UnknownType():
    assert str(ak._v2.types.unknowntype.UnknownType()) == "unknown"
    assert (
        str(ak._v2.types.unknowntype.UnknownType({"x": 123}))
        == 'unknown[parameters={"x": 123}]'
    )
    assert str(ak._v2.types.unknowntype.UnknownType(None, "override")) == "override"
    assert (
        str(ak._v2.types.unknowntype.UnknownType({"x": 123}, "override")) == "override"
    )
    assert (
        str(ak._v2.types.unknowntype.UnknownType({"__categorical__": True}))
        == "categorical[type=unknown]"
    )
    assert (
        str(ak._v2.types.unknowntype.UnknownType({"__categorical__": True, "x": 123}))
        == 'categorical[type=unknown[parameters={"x": 123}]]'
    )
    assert (
        str(ak._v2.types.unknowntype.UnknownType({"__categorical__": True}, "override"))
        == "categorical[type=override]"
    )

    assert repr(ak._v2.types.unknowntype.UnknownType()) == "UnknownType()"
    assert (
        repr(
            ak._v2.types.unknowntype.UnknownType(
                parameters={"__categorical__": True}, typestr="override"
            )
        )
        == "UnknownType(parameters={'__categorical__': True}, typestr='override')"
    )


@pytest.mark.skipif(
    ak._util.win,
    reason="NumPy does not have float16, float128, and complex256 -- on Windows",
)
def test_NumpyType():
    assert str(ak._v2.types.numpytype.NumpyType("bool")) == "bool"
    assert str(ak._v2.types.numpytype.NumpyType("int8")) == "int8"
    assert str(ak._v2.types.numpytype.NumpyType("uint8")) == "uint8"
    assert str(ak._v2.types.numpytype.NumpyType("int16")) == "int16"
    assert str(ak._v2.types.numpytype.NumpyType("uint16")) == "uint16"
    assert str(ak._v2.types.numpytype.NumpyType("int32")) == "int32"
    assert str(ak._v2.types.numpytype.NumpyType("uint32")) == "uint32"
    assert str(ak._v2.types.numpytype.NumpyType("int64")) == "int64"
    assert str(ak._v2.types.numpytype.NumpyType("uint64")) == "uint64"
    assert str(ak._v2.types.numpytype.NumpyType("float16")) == "float16"
    assert str(ak._v2.types.numpytype.NumpyType("float32")) == "float32"
    assert str(ak._v2.types.numpytype.NumpyType("float64")) == "float64"
    assert str(ak._v2.types.numpytype.NumpyType("float128")) == "float128"
    assert str(ak._v2.types.numpytype.NumpyType("complex64")) == "complex64"
    assert str(ak._v2.types.numpytype.NumpyType("complex128")) == "complex128"
    assert str(ak._v2.types.numpytype.NumpyType("complex256")) == "complex256"
    assert (
        str(ak._v2.types.numpytype.NumpyType("bool", {"x": 123}))
        == 'bool[parameters={"x": 123}]'
    )
    assert str(ak._v2.types.numpytype.NumpyType("bool", None, "override")) == "override"
    assert (
        str(ak._v2.types.numpytype.NumpyType("bool", {"x": 123}, "override"))
        == "override"
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("bool", {"__categorical__": True}))
        == "categorical[type=bool]"
    )
    assert (
        str(
            ak._v2.types.numpytype.NumpyType(
                "bool", {"__categorical__": True, "x": 123}
            )
        )
        == 'categorical[type=bool[parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.numpytype.NumpyType(
                "bool", {"__categorical__": True}, "override"
            )
        )
        == "categorical[type=override]"
    )
    assert str(ak._v2.types.numpytype.NumpyType("datetime64")) == "datetime64"
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "Y"}))
        == 'datetime64[unit="Y"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "M"}))
        == 'datetime64[unit="M"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "W"}))
        == 'datetime64[unit="W"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "D"}))
        == 'datetime64[unit="D"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "h"}))
        == 'datetime64[unit="h"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "m"}))
        == 'datetime64[unit="m"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "s"}))
        == 'datetime64[unit="s"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "ms"}))
        == 'datetime64[unit="ms"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "us"}))
        == 'datetime64[unit="us"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "ns"}))
        == 'datetime64[unit="ns"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "ps"}))
        == 'datetime64[unit="ps"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "fs"}))
        == 'datetime64[unit="fs"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "as"}))
        == 'datetime64[unit="as"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "10s"}))
        == 'datetime64[unit="10s"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "1s"}))
        == 'datetime64[unit="s"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"__unit__": "s", "x": 123}))
        == 'datetime64[unit="s", parameters={"x": 123}]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("datetime64", {"x": 123}))
        == 'datetime64[parameters={"x": 123}]'
    )
    assert str(ak._v2.types.numpytype.NumpyType("timedelta64")) == "timedelta64"
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "Y"}))
        == 'timedelta64[unit="Y"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "M"}))
        == 'timedelta64[unit="M"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "W"}))
        == 'timedelta64[unit="W"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "D"}))
        == 'timedelta64[unit="D"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "h"}))
        == 'timedelta64[unit="h"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "m"}))
        == 'timedelta64[unit="m"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "s"}))
        == 'timedelta64[unit="s"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "ms"}))
        == 'timedelta64[unit="ms"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "us"}))
        == 'timedelta64[unit="us"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "ns"}))
        == 'timedelta64[unit="ns"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "ps"}))
        == 'timedelta64[unit="ps"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "fs"}))
        == 'timedelta64[unit="fs"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "as"}))
        == 'timedelta64[unit="as"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "10s"}))
        == 'timedelta64[unit="10s"]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "1s"}))
        == 'timedelta64[unit="s"]'
    )
    assert (
        str(
            ak._v2.types.numpytype.NumpyType("timedelta64", {"__unit__": "s", "x": 123})
        )
        == 'timedelta64[unit="s", parameters={"x": 123}]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("timedelta64", {"x": 123}))
        == 'timedelta64[parameters={"x": 123}]'
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("uint8", {"__array__": "char"})) == "char"
    )
    assert (
        str(ak._v2.types.numpytype.NumpyType("uint8", {"__array__": "byte"})) == "byte"
    )
    assert (
        repr(ak._v2.types.numpytype.NumpyType(primitive="bool")) == "NumpyType('bool')"
    )
    assert (
        repr(
            ak._v2.types.numpytype.NumpyType(
                primitive="bool",
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "NumpyType('bool', parameters={'__categorical__': True}, typestr='override')"
    )
    assert (
        repr(
            ak._v2.types.numpytype.NumpyType(
                primitive="datetime64", parameters={"__unit__": "s"}
            )
        )
        == "NumpyType('datetime64', parameters={'__unit__': 's'})"
    )
    assert (
        repr(
            ak._v2.types.numpytype.NumpyType(
                primitive="uint8", parameters={"__array__": "char"}
            )
        )
        == "NumpyType('uint8', parameters={'__array__': 'char'})"
    )
    assert (
        repr(
            ak._v2.types.numpytype.NumpyType(
                primitive="uint8", parameters={"__array__": "byte"}
            )
        )
        == "NumpyType('uint8', parameters={'__array__': 'byte'})"
    )


def test_RegularType():
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.unknowntype.UnknownType(), 10
            )
        )
        == "10 * unknown"
    )
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.unknowntype.UnknownType(), 0
            )
        )
        == "0 * unknown"
    )
    with pytest.raises(ValueError):
        ak._v2.types.regulartype.RegularType(ak._v2.types.unknowntype.UnknownType(), -1)
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.unknowntype.UnknownType(), 10, {"x": 123}
            )
        )
        == '[10 * unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.unknowntype.UnknownType(), 10, None, "override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.unknowntype.UnknownType(), 10, {"x": 123}, "override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.unknowntype.UnknownType(), 10, {"__categorical__": True}
            )
        )
        == "categorical[type=10 * unknown]"
    )
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.unknowntype.UnknownType(),
                10,
                {"__categorical__": True, "x": 123},
            )
        )
        == 'categorical[type=[10 * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.unknowntype.UnknownType(),
                10,
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.numpytype.NumpyType("uint8", {"__array__": "char"}),
                10,
                {"__array__": "string"},
            )
        )
        == "string[10]"
    )
    assert (
        str(
            ak._v2.types.regulartype.RegularType(
                ak._v2.types.numpytype.NumpyType("uint8", {"__array__": "byte"}),
                10,
                {"__array__": "bytestring"},
            )
        )
        == "bytes[10]"
    )

    assert (
        repr(
            ak._v2.types.regulartype.RegularType(
                content=ak._v2.types.unknowntype.UnknownType(), size=10
            )
        )
        == "RegularType(UnknownType(), 10)"
    )
    assert (
        repr(
            ak._v2.types.regulartype.RegularType(
                content=ak._v2.types.unknowntype.UnknownType(),
                size=10,
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "RegularType(UnknownType(), 10, parameters={'__categorical__': True}, typestr='override')"
    )
    assert (
        repr(
            ak._v2.types.regulartype.RegularType(
                content=ak._v2.types.numpytype.NumpyType(
                    primitive="uint8", parameters={"__array__": "char"}
                ),
                parameters={"__array__": "string"},
                size=10,
            )
        )
        == "RegularType(NumpyType('uint8', parameters={'__array__': 'char'}), 10, parameters={'__array__': 'string'})"
    )
    assert (
        repr(
            ak._v2.types.regulartype.RegularType(
                content=ak._v2.types.numpytype.NumpyType(
                    primitive="uint8", parameters={"__array__": "byte"}
                ),
                parameters={"__array__": "bytestring"},
                size=10,
            )
        )
        == "RegularType(NumpyType('uint8', parameters={'__array__': 'byte'}), 10, parameters={'__array__': 'bytestring'})"
    )


def test_ListType():
    assert (
        str(ak._v2.types.listtype.ListType(ak._v2.types.unknowntype.UnknownType()))
        == "var * unknown"
    )
    assert (
        str(
            ak._v2.types.listtype.ListType(
                ak._v2.types.unknowntype.UnknownType(), {"x": 123}
            )
        )
        == '[var * unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.listtype.ListType(
                ak._v2.types.unknowntype.UnknownType(), None, "override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.listtype.ListType(
                ak._v2.types.unknowntype.UnknownType(), {"x": 123}, "override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.listtype.ListType(
                ak._v2.types.unknowntype.UnknownType(), {"__categorical__": True}
            )
        )
        == "categorical[type=var * unknown]"
    )
    assert (
        str(
            ak._v2.types.listtype.ListType(
                ak._v2.types.unknowntype.UnknownType(),
                {"__categorical__": True, "x": 123},
            )
        )
        == 'categorical[type=[var * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.listtype.ListType(
                ak._v2.types.unknowntype.UnknownType(),
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.listtype.ListType(
                ak._v2.types.numpytype.NumpyType("uint8", {"__array__": "char"}),
                {"__array__": "string"},
            )
        )
        == "string"
    )
    assert (
        str(
            ak._v2.types.listtype.ListType(
                ak._v2.types.numpytype.NumpyType("uint8", {"__array__": "byte"}),
                {"__array__": "bytestring"},
            )
        )
        == "bytes"
    )

    assert (
        repr(
            ak._v2.types.listtype.ListType(
                content=ak._v2.types.unknowntype.UnknownType()
            )
        )
        == "ListType(UnknownType())"
    )
    assert (
        repr(
            ak._v2.types.listtype.ListType(
                content=ak._v2.types.unknowntype.UnknownType(),
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "ListType(UnknownType(), parameters={'__categorical__': True}, typestr='override')"
    )
    assert (
        repr(
            ak._v2.types.listtype.ListType(
                content=ak._v2.types.numpytype.NumpyType(
                    primitive="uint8", parameters={"__array__": "char"}
                ),
                parameters={"__array__": "string"},
            )
        )
        == "ListType(NumpyType('uint8', parameters={'__array__': 'char'}), parameters={'__array__': 'string'})"
    )
    assert (
        repr(
            ak._v2.types.listtype.ListType(
                content=ak._v2.types.numpytype.NumpyType(
                    primitive="uint8", parameters={"__array__": "byte"}
                ),
                parameters={"__array__": "bytestring"},
            )
        )
        == "ListType(NumpyType('uint8', parameters={'__array__': 'byte'}), parameters={'__array__': 'bytestring'})"
    )


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_RecordType():
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
            )
        )
        == "(unknown, bool)"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
            )
        )
        == "{x: unknown, y: bool}"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__record__": "Name"},
            )
        )
        == "Name[unknown, bool]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__record__": "Name"},
            )
        )
        == "Name[x: unknown, y: bool]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                None,
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                None,
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__record__": "Name"},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__record__": "Name"},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"x": 123},
            )
        )
        == 'tuple[[unknown, bool], parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"x": 123},
            )
        )
        == 'struct[["x", "y"], [unknown, bool], parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__record__": "Name", "x": 123},
            )
        )
        == 'Name[unknown, bool, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__record__": "Name", "x": 123},
            )
        )
        == 'Name[x: unknown, y: bool, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__record__": "Name", "x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__record__": "Name", "x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__categorical__": True},
            )
        )
        == "categorical[type=(unknown, bool)]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__categorical__": True},
            )
        )
        == "categorical[type={x: unknown, y: bool}]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__record__": "Name", "__categorical__": True},
            )
        )
        == "categorical[type=Name[unknown, bool]]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__record__": "Name", "__categorical__": True},
            )
        )
        == "categorical[type=Name[x: unknown, y: bool]]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__record__": "Name", "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__record__": "Name", "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=tuple[[unknown, bool], parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=struct[["x", "y"], [unknown, bool], parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__record__": "Name", "x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=Name[unknown, bool, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__record__": "Name", "x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=Name[x: unknown, y: bool, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                {"__record__": "Name", "x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.recordtype.RecordType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                {"__record__": "Name", "x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )

    assert (
        repr(
            ak._v2.types.recordtype.RecordType(
                contents=[
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                fields=None,
            )
        )
        == "RecordType([UnknownType(), NumpyType('bool')], None)"
    )
    assert (
        repr(
            ak._v2.types.recordtype.RecordType(
                contents=[
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                fields=["x", "y"],
            )
        )
        == "RecordType([UnknownType(), NumpyType('bool')], ['x', 'y'])"
    )
    assert (
        repr(
            ak._v2.types.recordtype.RecordType(
                contents=[
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                fields=None,
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "RecordType([UnknownType(), NumpyType('bool')], None, parameters={'__record__': 'Name', 'x': 123, '__categorical__': True}, typestr='override')"
    )
    assert (
        repr(
            ak._v2.types.recordtype.RecordType(
                contents=[
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                fields=None,
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
            )
        )
        == "RecordType([UnknownType(), NumpyType('bool')], None, parameters={'__record__': 'Name', 'x': 123, '__categorical__': True})"
    )
    assert (
        repr(
            ak._v2.types.recordtype.RecordType(
                contents=[
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                fields=["x", "y"],
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "RecordType([UnknownType(), NumpyType('bool')], ['x', 'y'], parameters={'__record__': 'Name', 'x': 123, '__categorical__': True}, typestr='override')"
    )


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_OptionType():
    assert (
        str(ak._v2.types.optiontype.OptionType(ak._v2.types.unknowntype.UnknownType()))
        == "?unknown"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.listtype.ListType(ak._v2.types.unknowntype.UnknownType())
            )
        )
        == "option[var * unknown]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.regulartype.RegularType(
                    ak._v2.types.unknowntype.UnknownType(), 10
                )
            )
        )
        == "option[10 * unknown]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.unknowntype.UnknownType(), {"x": 123}
            )
        )
        == 'option[unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.listtype.ListType(ak._v2.types.unknowntype.UnknownType()),
                {"x": 123},
            )
        )
        == 'option[var * unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.regulartype.RegularType(
                    ak._v2.types.unknowntype.UnknownType(), 10
                ),
                {"x": 123},
            )
        )
        == 'option[10 * unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.unknowntype.UnknownType(), None, "override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.listtype.ListType(ak._v2.types.unknowntype.UnknownType()),
                None,
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.regulartype.RegularType(
                    ak._v2.types.unknowntype.UnknownType(), 10
                ),
                None,
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.unknowntype.UnknownType(), {"x": 123}, "override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.listtype.ListType(ak._v2.types.unknowntype.UnknownType()),
                {"x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.regulartype.RegularType(
                    ak._v2.types.unknowntype.UnknownType(), 10
                ),
                {"x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.unknowntype.UnknownType(), {"__categorical__": True}
            )
        )
        == "categorical[type=?unknown]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.listtype.ListType(ak._v2.types.unknowntype.UnknownType()),
                {"__categorical__": True},
            )
        )
        == "categorical[type=option[var * unknown]]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.regulartype.RegularType(
                    ak._v2.types.unknowntype.UnknownType(), 10
                ),
                {"__categorical__": True},
            )
        )
        == "categorical[type=option[10 * unknown]]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.unknowntype.UnknownType(),
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=option[unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.listtype.ListType(ak._v2.types.unknowntype.UnknownType()),
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=option[var * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.regulartype.RegularType(
                    ak._v2.types.unknowntype.UnknownType(), 10
                ),
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=option[10 * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.unknowntype.UnknownType(),
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.listtype.ListType(ak._v2.types.unknowntype.UnknownType()),
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.regulartype.RegularType(
                    ak._v2.types.unknowntype.UnknownType(), 10
                ),
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.unknowntype.UnknownType(),
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.listtype.ListType(ak._v2.types.unknowntype.UnknownType()),
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.optiontype.OptionType(
                ak._v2.types.regulartype.RegularType(
                    ak._v2.types.unknowntype.UnknownType(), 10
                ),
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )

    assert (
        repr(
            ak._v2.types.optiontype.OptionType(
                content=ak._v2.types.unknowntype.UnknownType()
            )
        )
        == "OptionType(UnknownType())"
    )
    assert (
        repr(
            ak._v2.types.optiontype.OptionType(
                content=ak._v2.types.listtype.ListType(
                    ak._v2.types.unknowntype.UnknownType()
                )
            )
        )
        == "OptionType(ListType(UnknownType()))"
    )
    assert (
        repr(
            ak._v2.types.optiontype.OptionType(
                content=ak._v2.types.regulartype.RegularType(
                    ak._v2.types.unknowntype.UnknownType(), 10
                ),
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "OptionType(RegularType(UnknownType(), 10), parameters={'x': 123, '__categorical__': True}, typestr='override')"
    )


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_UnionType():
    assert (
        str(
            ak._v2.types.uniontype.UnionType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ]
            )
        )
        == "union[unknown, bool]"
    )
    assert (
        str(
            ak._v2.types.uniontype.UnionType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                {"x": 123},
            )
        )
        == 'union[unknown, bool, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.uniontype.UnionType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                None,
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.uniontype.UnionType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                {"x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.uniontype.UnionType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                {"__categorical__": True},
            )
        )
        == "categorical[type=union[unknown, bool]]"
    )
    assert (
        str(
            ak._v2.types.uniontype.UnionType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=union[unknown, bool, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.uniontype.UnionType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.uniontype.UnionType(
                [
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )

    assert (
        repr(
            ak._v2.types.uniontype.UnionType(
                contents=[
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ]
            )
        )
        == "UnionType([UnknownType(), NumpyType('bool')])"
    )
    assert (
        repr(
            ak._v2.types.uniontype.UnionType(
                contents=[
                    ak._v2.types.unknowntype.UnknownType(),
                    ak._v2.types.numpytype.NumpyType("bool"),
                ],
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "UnionType([UnknownType(), NumpyType('bool')], parameters={'x': 123, '__categorical__': True}, typestr='override')"
    )


def test_ArrayType():
    assert (
        str(
            ak._v2.types.arraytype.ArrayType(ak._v2.types.unknowntype.UnknownType(), 10)
        )
        == "10 * unknown"
    )
    assert (
        str(ak._v2.types.arraytype.ArrayType(ak._v2.types.unknowntype.UnknownType(), 0))
        == "0 * unknown"
    )
    with pytest.raises(ValueError):
        ak._v2.types.arraytype.ArrayType(ak._v2.types.unknowntype.UnknownType(), -1)

    # ArrayType should not have these arguments (should not be a Type subclass)
    with pytest.raises(TypeError):
        ak._v2.types.arraytype.ArrayType(
            ak._v2.types.unknowntype.UnknownType(), 10, {"x": 123}
        )
    with pytest.raises(TypeError):
        ak._v2.types.arraytype.ArrayType(
            ak._v2.types.unknowntype.UnknownType(), 10, None, "override"
        )

    assert (
        repr(
            ak._v2.types.arraytype.ArrayType(
                content=ak._v2.types.unknowntype.UnknownType(), length=10
            )
        )
        == "ArrayType(UnknownType(), 10)"
    )


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_EmptyForm():
    assert (
        str(ak._v2.forms.emptyform.EmptyForm())
        == """{
    "class": "EmptyArray"
}"""
    )
    assert (
        str(
            ak._v2.forms.emptyform.EmptyForm(
                has_identifier=True, parameters={"x": 123}, form_key="hello"
            )
        )
        == """{
    "class": "EmptyArray",
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert repr(ak._v2.forms.emptyform.EmptyForm()) == "EmptyForm()"
    assert (
        repr(
            ak._v2.forms.emptyform.EmptyForm(
                has_identifier=True, parameters={"x": 123}, form_key="hello"
            )
        )
        == "EmptyForm(has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.emptyform.EmptyForm().tolist(verbose=False) == {
        "class": "EmptyArray"
    }
    assert ak._v2.forms.emptyform.EmptyForm().tolist() == {
        "class": "EmptyArray",
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.emptyform.EmptyForm(
        has_identifier=True, parameters={"x": 123}, form_key="hello"
    ).tolist(verbose=False) == {
        "class": "EmptyArray",
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter({"class": "EmptyArray"}).tolist() == {
        "class": "EmptyArray",
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "EmptyArray",
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist() == {
        "class": "EmptyArray",
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
@pytest.mark.skipif(
    ak._util.win,
    reason="NumPy does not have float16, float128, and complex256 -- on Windows",
)
def test_NumpyForm():
    assert (
        str(ak._v2.forms.numpyform.NumpyForm("bool"))
        == """{
    "class": "NumpyArray",
    "primitive": "bool"
}"""
    )
    assert (
        repr(ak._v2.forms.numpyform.NumpyForm(primitive="bool")) == "NumpyForm('bool')"
    )
    assert (
        repr(
            ak._v2.forms.numpyform.NumpyForm(
                primitive="bool",
                inner_shape=[1, 2, 3],
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "NumpyForm('bool', inner_shape=(1, 2, 3), has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.numpyform.NumpyForm("bool").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak._v2.forms.numpyform.NumpyForm("bool").tolist() == {
        "class": "NumpyArray",
        "primitive": "bool",
        "inner_shape": [],
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.numpyform.NumpyForm(
        "bool",
        inner_shape=[1, 2, 3],
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
        "inner_shape": [1, 2, 3],
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }

    assert ak._v2.forms.numpyform.NumpyForm("bool").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak._v2.forms.numpyform.NumpyForm("int8").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int8",
    }
    assert ak._v2.forms.numpyform.NumpyForm("uint8").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint8",
    }
    assert ak._v2.forms.numpyform.NumpyForm("int16").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int16",
    }
    assert ak._v2.forms.numpyform.NumpyForm("uint16").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint16",
    }
    assert ak._v2.forms.numpyform.NumpyForm("int32").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int32",
    }
    assert ak._v2.forms.numpyform.NumpyForm("uint32").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint32",
    }
    assert ak._v2.forms.numpyform.NumpyForm("int64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int64",
    }
    assert ak._v2.forms.numpyform.NumpyForm("uint64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint64",
    }
    assert ak._v2.forms.numpyform.NumpyForm("float16").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float16",
    }
    assert ak._v2.forms.numpyform.NumpyForm("float32").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float32",
    }
    assert ak._v2.forms.numpyform.NumpyForm("float64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }
    assert ak._v2.forms.numpyform.NumpyForm("float128").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float128",
    }
    assert ak._v2.forms.numpyform.NumpyForm("complex64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex64",
    }
    assert ak._v2.forms.numpyform.NumpyForm("complex128").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex128",
    }
    assert ak._v2.forms.numpyform.NumpyForm("complex256").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex256",
    }
    assert ak._v2.forms.numpyform.NumpyForm("datetime64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
    }
    assert ak._v2.forms.numpyform.NumpyForm(
        "datetime64", parameters={"__unit__": "s"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s"},
    }
    assert ak._v2.forms.numpyform.NumpyForm(
        "datetime64", parameters={"__unit__": "s", "x": 123}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s", "x": 123},
    }
    assert ak._v2.forms.numpyform.NumpyForm("timedelta64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
    }
    assert ak._v2.forms.numpyform.NumpyForm(
        "timedelta64", parameters={"__unit__": "s"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s"},
    }
    assert ak._v2.forms.numpyform.NumpyForm(
        "timedelta64", parameters={"__unit__": "s", "x": 123}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s", "x": 123},
    }

    assert ak._v2.forms.numpyform.from_dtype(np.dtype("bool")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("int8")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int8",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("uint8")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "uint8",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("int16")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int16",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("uint16")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "uint16",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("int32")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int32",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("uint32")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "uint32",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("int64")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int64",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("uint64")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "uint64",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("float16")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "float16",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("float32")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "float32",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("float64")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("float128")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "float128",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("complex64")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "complex64",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("complex128")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "complex128",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("complex256")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "complex256",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("M8")).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("M8[s]")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s"},
    }
    assert ak._v2.forms.numpyform.from_dtype(
        np.dtype("M8[s]"), parameters={"x": 123}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s", "x": 123},
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("m8")).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype("m8[s]")).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s"},
    }
    assert ak._v2.forms.numpyform.from_dtype(
        np.dtype("m8[s]"), parameters={"x": 123}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s", "x": 123},
    }
    assert ak._v2.forms.numpyform.from_dtype(np.dtype(("bool", (1, 2, 3)))).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "bool",
        "inner_shape": [1, 2, 3],
    }
    with pytest.raises(TypeError):
        ak._v2.forms.from_dtype(np.dtype("O")).tolist(verbose=False)
    with pytest.raises(TypeError):
        ak._v2.forms.from_dtype(
            np.dtype([("one", np.int64), ("two", np.float64)])
        ).tolist(verbose=False)
    assert ak._v2.forms.from_iter("bool").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak._v2.forms.from_iter("int8").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int8",
    }
    assert ak._v2.forms.from_iter("uint8").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint8",
    }
    assert ak._v2.forms.from_iter("int16").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int16",
    }
    assert ak._v2.forms.from_iter("uint16").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint16",
    }
    assert ak._v2.forms.from_iter("int32").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int32",
    }
    assert ak._v2.forms.from_iter("uint32").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint32",
    }
    assert ak._v2.forms.from_iter("int64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int64",
    }
    assert ak._v2.forms.from_iter("uint64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint64",
    }
    assert ak._v2.forms.from_iter("float16").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float16",
    }
    assert ak._v2.forms.from_iter("float32").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float32",
    }
    assert ak._v2.forms.from_iter("float64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }
    assert ak._v2.forms.from_iter("float128").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float128",
    }
    assert ak._v2.forms.from_iter("complex64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex64",
    }
    assert ak._v2.forms.from_iter("complex128").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex128",
    }
    assert ak._v2.forms.from_iter("complex256").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex256",
    }
    assert ak._v2.forms.from_iter("datetime64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "NumpyArray",
            "primitive": "datetime64",
            "parameters": {"__unit__": "s"},
        }
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s"},
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "NumpyArray",
            "primitive": "datetime64",
            "parameters": {"__unit__": "s", "x": 123},
        }
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s", "x": 123},
    }
    assert ak._v2.forms.from_iter("timedelta64").tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "NumpyArray",
            "primitive": "timedelta64",
            "parameters": {"__unit__": "s"},
        }
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s"},
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "NumpyArray",
            "primitive": "timedelta64",
            "parameters": {"__unit__": "s", "x": 123},
        }
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s", "x": 123},
    }

    assert ak._v2.forms.from_iter({"class": "NumpyArray", "primitive": "bool"}).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak._v2.forms.from_iter({"class": "NumpyArray", "primitive": "int8"}).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int8",
    }
    assert ak._v2.forms.from_iter({"class": "NumpyArray", "primitive": "uint8"}).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "uint8",
    }
    assert ak._v2.forms.from_iter({"class": "NumpyArray", "primitive": "int16"}).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int16",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "uint16"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint16",
    }
    assert ak._v2.forms.from_iter({"class": "NumpyArray", "primitive": "int32"}).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int32",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "uint32"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint32",
    }
    assert ak._v2.forms.from_iter({"class": "NumpyArray", "primitive": "int64"}).tolist(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int64",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "uint64"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint64",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "float16"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float16",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "float32"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float32",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "float64"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "float128"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float128",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "complex64"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex64",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "complex128"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex128",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "complex256"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex256",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "datetime64"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
    }
    assert ak._v2.forms.from_iter(
        {"class": "NumpyArray", "primitive": "timedelta64"}
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
    }

    assert ak._v2.forms.from_iter(
        {
            "class": "NumpyArray",
            "primitive": "bool",
            "inner_shape": [1, 2, 3],
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
        "inner_shape": [1, 2, 3],
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_RegularForm():
    assert (
        str(
            ak._v2.forms.regularform.RegularForm(ak._v2.forms.emptyform.EmptyForm(), 10)
        )
        == """{
    "class": "RegularArray",
    "size": 10,
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.regularform.RegularForm(
                ak._v2.forms.emptyform.EmptyForm(),
                10,
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "RegularArray",
    "size": 10,
    "content": {
        "class": "EmptyArray"
    },
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak._v2.forms.regularform.RegularForm(
                content=ak._v2.forms.emptyform.EmptyForm(), size=10
            )
        )
        == "RegularForm(EmptyForm(), 10)"
    )
    assert (
        repr(
            ak._v2.forms.regularform.RegularForm(
                content=ak._v2.forms.emptyform.EmptyForm(),
                size=10,
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "RegularForm(EmptyForm(), 10, has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.regularform.RegularForm(
        ak._v2.forms.emptyform.EmptyForm(), 10
    ).tolist(verbose=False) == {
        "class": "RegularArray",
        "size": 10,
        "content": {"class": "EmptyArray"},
    }
    assert ak._v2.forms.regularform.RegularForm(
        ak._v2.forms.emptyform.EmptyForm(), 10
    ).tolist() == {
        "class": "RegularArray",
        "size": 10,
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.regularform.RegularForm(
        content=ak._v2.forms.emptyform.EmptyForm(),
        size=10,
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "RegularArray",
        "size": 10,
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {"class": "RegularArray", "size": 10, "content": {"class": "EmptyArray"}}
    ).tolist() == {
        "class": "RegularArray",
        "size": 10,
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "RegularArray",
            "size": 10,
            "content": {"class": "EmptyArray"},
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "RegularArray",
        "size": 10,
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.regularform.RegularForm(
        ak._v2.forms.numpyform.NumpyForm("bool"), 10
    ).tolist() == {
        "class": "RegularArray",
        "content": {
            "class": "NumpyArray",
            "primitive": "bool",
            "inner_shape": [],
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "size": 10,
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.regularform.RegularForm(
        ak._v2.forms.numpyform.NumpyForm("bool"), 10
    ).tolist(verbose=False) == {
        "class": "RegularArray",
        "content": "bool",
        "size": 10,
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_ListForm():
    assert (
        str(
            ak._v2.forms.listform.ListForm(
                "i32", "i32", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "ListArray",
    "starts": "i32",
    "stops": "i32",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.listform.ListForm(
                "u32", "u32", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "ListArray",
    "starts": "u32",
    "stops": "u32",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.listform.ListForm(
                "i64", "i64", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "ListArray",
    "starts": "i64",
    "stops": "i64",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.listform.ListForm(
                "i32",
                "i32",
                ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "ListArray",
    "starts": "i32",
    "stops": "i32",
    "content": {
        "class": "EmptyArray"
    },
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak._v2.forms.listform.ListForm(
                starts="i32", stops="i32", content=ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == "ListForm('i32', 'i32', EmptyForm())"
    )
    assert (
        repr(
            ak._v2.forms.listform.ListForm(
                starts="i32",
                stops="i32",
                content=ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "ListForm('i32', 'i32', EmptyForm(), has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.listform.ListForm(
        "i32", "i32", ak._v2.forms.emptyform.EmptyForm()
    ).tolist(verbose=False) == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {"class": "EmptyArray"},
    }
    assert ak._v2.forms.listform.ListForm(
        "i32", "i32", ak._v2.forms.emptyform.EmptyForm()
    ).tolist() == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.listform.ListForm(
        starts="i32",
        stops="i32",
        content=ak._v2.forms.emptyform.EmptyForm(),
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ListArray",
            "starts": "i32",
            "stops": "i32",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ListArray",
            "starts": "u32",
            "stops": "u32",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "ListArray",
        "starts": "u32",
        "stops": "u32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ListArray",
            "starts": "i64",
            "stops": "i64",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "ListArray",
        "starts": "i64",
        "stops": "i64",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ListArray",
            "starts": "i32",
            "stops": "i32",
            "content": {"class": "EmptyArray"},
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_ListOffsetForm():
    assert (
        str(
            ak._v2.forms.listoffsetform.ListOffsetForm(
                "i32", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "ListOffsetArray",
    "offsets": "i32",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.listoffsetform.ListOffsetForm(
                "u32", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "ListOffsetArray",
    "offsets": "u32",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.listoffsetform.ListOffsetForm(
                "i64", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "ListOffsetArray",
    "offsets": "i64",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.listoffsetform.ListOffsetForm(
                "i32",
                ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "ListOffsetArray",
    "offsets": "i32",
    "content": {
        "class": "EmptyArray"
    },
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak._v2.forms.listoffsetform.ListOffsetForm(
                offsets="i32", content=ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == "ListOffsetForm('i32', EmptyForm())"
    )
    assert (
        repr(
            ak._v2.forms.listoffsetform.ListOffsetForm(
                offsets="i32",
                content=ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "ListOffsetForm('i32', EmptyForm(), has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.listoffsetform.ListOffsetForm(
        "i32", ak._v2.forms.emptyform.EmptyForm()
    ).tolist(verbose=False) == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {"class": "EmptyArray"},
    }
    assert ak._v2.forms.listoffsetform.ListOffsetForm(
        "i32", ak._v2.forms.emptyform.EmptyForm()
    ).tolist() == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.listoffsetform.ListOffsetForm(
        offsets="i32",
        content=ak._v2.forms.emptyform.EmptyForm(),
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ListOffsetArray",
            "offsets": "i32",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ListOffsetArray",
            "offsets": "u32",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "ListOffsetArray",
        "offsets": "u32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ListOffsetArray",
            "offsets": "i32",
            "content": {"class": "EmptyArray"},
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_RecordForm():
    assert (
        str(
            ak._v2.forms.recordform.RecordForm(
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                None,
            )
        )
        == """{
    "class": "RecordArray",
    "contents": [
        {
            "class": "EmptyArray"
        },
        "bool"
    ]
}"""
    )
    assert (
        str(
            ak._v2.forms.recordform.RecordForm(
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                ["x", "y"],
            )
        )
        == """{
    "class": "RecordArray",
    "contents": {
        "x": {
            "class": "EmptyArray"
        },
        "y": "bool"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.recordform.RecordForm(
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                None,
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "RecordArray",
    "contents": [
        {
            "class": "EmptyArray"
        },
        "bool"
    ],
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        str(
            ak._v2.forms.recordform.RecordForm(
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                ["x", "y"],
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "RecordArray",
    "contents": {
        "x": {
            "class": "EmptyArray"
        },
        "y": "bool"
    },
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )

    assert (
        repr(
            ak._v2.forms.recordform.RecordForm(
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                None,
            )
        )
        == "RecordForm([EmptyForm(), NumpyForm('bool')], None)"
    )
    assert (
        repr(
            ak._v2.forms.recordform.RecordForm(
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                ["x", "y"],
            )
        )
        == "RecordForm([EmptyForm(), NumpyForm('bool')], ['x', 'y'])"
    )
    assert (
        repr(
            ak._v2.forms.recordform.RecordForm(
                contents=[
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                fields=None,
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "RecordForm([EmptyForm(), NumpyForm('bool')], None, has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )
    assert (
        repr(
            ak._v2.forms.recordform.RecordForm(
                contents=[
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                fields=["x", "y"],
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "RecordForm([EmptyForm(), NumpyForm('bool')], ['x', 'y'], has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.recordform.RecordForm(
        [ak._v2.forms.emptyform.EmptyForm(), ak._v2.forms.numpyform.NumpyForm("bool")],
        None,
    ).tolist(verbose=False) == {
        "class": "RecordArray",
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
    }
    assert ak._v2.forms.recordform.RecordForm(
        [ak._v2.forms.emptyform.EmptyForm(), ak._v2.forms.numpyform.NumpyForm("bool")],
        ["x", "y"],
    ).tolist(verbose=False) == {
        "class": "RecordArray",
        "contents": {
            "x": {"class": "EmptyArray"},
            "y": "bool",
        },
    }
    assert ak._v2.forms.recordform.RecordForm(
        [ak._v2.forms.emptyform.EmptyForm(), ak._v2.forms.numpyform.NumpyForm("bool")],
        None,
    ).tolist() == {
        "class": "RecordArray",
        "contents": [
            {
                "class": "EmptyArray",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.recordform.RecordForm(
        [ak._v2.forms.emptyform.EmptyForm(), ak._v2.forms.numpyform.NumpyForm("bool")],
        ["x", "y"],
    ).tolist() == {
        "class": "RecordArray",
        "contents": {
            "x": {
                "class": "EmptyArray",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            "y": {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.recordform.RecordForm(
        contents=[
            ak._v2.forms.emptyform.EmptyForm(),
            ak._v2.forms.numpyform.NumpyForm("bool"),
        ],
        fields=None,
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "RecordArray",
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.recordform.RecordForm(
        contents=[
            ak._v2.forms.emptyform.EmptyForm(),
            ak._v2.forms.numpyform.NumpyForm("bool"),
        ],
        fields=["x", "y"],
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "RecordArray",
        "contents": {
            "x": {"class": "EmptyArray"},
            "y": "bool",
        },
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "RecordArray",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
        }
    ).tolist() == {
        "class": "RecordArray",
        "contents": [
            {
                "class": "EmptyArray",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "RecordArray",
            "contents": {
                "x": {"class": "EmptyArray"},
                "y": "bool",
            },
        }
    ).tolist() == {
        "class": "RecordArray",
        "contents": {
            "x": {
                "class": "EmptyArray",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            "y": {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "RecordArray",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "RecordArray",
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "RecordArray",
            "contents": {
                "x": {"class": "EmptyArray"},
                "y": "bool",
            },
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "RecordArray",
        "contents": {
            "x": {"class": "EmptyArray"},
            "y": "bool",
        },
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_IndexedForm():
    assert (
        str(
            ak._v2.forms.indexedform.IndexedForm(
                "i32", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "IndexedArray",
    "index": "i32",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.indexedform.IndexedForm(
                "u32", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "IndexedArray",
    "index": "u32",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.indexedform.IndexedForm(
                "i64", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "IndexedArray",
    "index": "i64",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.indexedform.IndexedForm(
                "i32",
                ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "IndexedArray",
    "index": "i32",
    "content": {
        "class": "EmptyArray"
    },
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak._v2.forms.indexedform.IndexedForm(
                index="i32", content=ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == "IndexedForm('i32', EmptyForm())"
    )
    assert (
        repr(
            ak._v2.forms.indexedform.IndexedForm(
                index="i32",
                content=ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "IndexedForm('i32', EmptyForm(), has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.indexedform.IndexedForm(
        "i32", ak._v2.forms.emptyform.EmptyForm()
    ).tolist(verbose=False) == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
    }
    assert ak._v2.forms.indexedform.IndexedForm(
        "i32", ak._v2.forms.emptyform.EmptyForm()
    ).tolist() == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.indexedform.IndexedForm(
        index="i32",
        content=ak._v2.forms.emptyform.EmptyForm(),
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "IndexedArray",
            "index": "i32",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "IndexedArray",
            "index": "u32",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "IndexedArray",
        "index": "u32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "IndexedArray",
            "index": "i64",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "IndexedArray",
        "index": "i64",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "IndexedArray",
            "index": "i32",
            "content": {"class": "EmptyArray"},
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_IndexedOptionForm():
    assert (
        str(
            ak._v2.forms.indexedoptionform.IndexedOptionForm(
                "i32", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "IndexedOptionArray",
    "index": "i32",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.indexedoptionform.IndexedOptionForm(
                "i64", ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == """{
    "class": "IndexedOptionArray",
    "index": "i64",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.indexedoptionform.IndexedOptionForm(
                "i32",
                ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "IndexedOptionArray",
    "index": "i32",
    "content": {
        "class": "EmptyArray"
    },
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak._v2.forms.indexedoptionform.IndexedOptionForm(
                index="i32", content=ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == "IndexedOptionForm('i32', EmptyForm())"
    )
    assert (
        repr(
            ak._v2.forms.indexedoptionform.IndexedOptionForm(
                index="i32",
                content=ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "IndexedOptionForm('i32', EmptyForm(), has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.indexedoptionform.IndexedOptionForm(
        "i32", ak._v2.forms.emptyform.EmptyForm()
    ).tolist(verbose=False) == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
    }
    assert ak._v2.forms.indexedoptionform.IndexedOptionForm(
        "i32", ak._v2.forms.emptyform.EmptyForm()
    ).tolist() == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.indexedoptionform.IndexedOptionForm(
        index="i32",
        content=ak._v2.forms.emptyform.EmptyForm(),
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "IndexedOptionArray",
            "index": "i32",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "IndexedOptionArray",
            "index": "i64",
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "IndexedOptionArray",
        "index": "i64",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "IndexedOptionArray",
            "index": "i32",
            "content": {"class": "EmptyArray"},
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_ByteMaskedForm():
    assert (
        str(
            ak._v2.forms.bytemaskedform.ByteMaskedForm(
                "i8", ak._v2.forms.emptyform.EmptyForm(), True
            )
        )
        == """{
    "class": "ByteMaskedArray",
    "mask": "i8",
    "valid_when": true,
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.bytemaskedform.ByteMaskedForm(
                "i8", ak._v2.forms.emptyform.EmptyForm(), False
            )
        )
        == """{
    "class": "ByteMaskedArray",
    "mask": "i8",
    "valid_when": false,
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.bytemaskedform.ByteMaskedForm(
                "i8",
                ak._v2.forms.emptyform.EmptyForm(),
                True,
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "ByteMaskedArray",
    "mask": "i8",
    "valid_when": true,
    "content": {
        "class": "EmptyArray"
    },
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak._v2.forms.bytemaskedform.ByteMaskedForm(
                mask="i8", content=ak._v2.forms.emptyform.EmptyForm(), valid_when=True
            )
        )
        == "ByteMaskedForm('i8', EmptyForm(), True)"
    )
    assert (
        repr(
            ak._v2.forms.bytemaskedform.ByteMaskedForm(
                mask="i8",
                content=ak._v2.forms.emptyform.EmptyForm(),
                valid_when=True,
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "ByteMaskedForm('i8', EmptyForm(), True, has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.bytemaskedform.ByteMaskedForm(
        "i8", ak._v2.forms.emptyform.EmptyForm(), True
    ).tolist(verbose=False) == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {"class": "EmptyArray"},
    }
    assert ak._v2.forms.bytemaskedform.ByteMaskedForm(
        "i8", ak._v2.forms.emptyform.EmptyForm(), True
    ).tolist() == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.bytemaskedform.ByteMaskedForm(
        mask="i8",
        content=ak._v2.forms.emptyform.EmptyForm(),
        valid_when=True,
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ByteMaskedArray",
            "mask": "i8",
            "valid_when": True,
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ByteMaskedArray",
            "mask": "i64",
            "valid_when": True,
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "ByteMaskedArray",
        "mask": "i64",
        "valid_when": True,
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "ByteMaskedArray",
            "mask": "i8",
            "valid_when": True,
            "content": {"class": "EmptyArray"},
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_BitMaskedForm():
    assert (
        str(
            ak._v2.forms.bitmaskedform.BitMaskedForm(
                "u8", ak._v2.forms.emptyform.EmptyForm(), True, True
            )
        )
        == """{
    "class": "BitMaskedArray",
    "mask": "u8",
    "valid_when": true,
    "lsb_order": true,
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.bitmaskedform.BitMaskedForm(
                "u8", ak._v2.forms.emptyform.EmptyForm(), False, True
            )
        )
        == """{
    "class": "BitMaskedArray",
    "mask": "u8",
    "valid_when": false,
    "lsb_order": true,
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.bitmaskedform.BitMaskedForm(
                "u8", ak._v2.forms.emptyform.EmptyForm(), True, False
            )
        )
        == """{
    "class": "BitMaskedArray",
    "mask": "u8",
    "valid_when": true,
    "lsb_order": false,
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.bitmaskedform.BitMaskedForm(
                "u8", ak._v2.forms.emptyform.EmptyForm(), False, False
            )
        )
        == """{
    "class": "BitMaskedArray",
    "mask": "u8",
    "valid_when": false,
    "lsb_order": false,
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.bitmaskedform.BitMaskedForm(
                "u8",
                ak._v2.forms.emptyform.EmptyForm(),
                True,
                False,
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "BitMaskedArray",
    "mask": "u8",
    "valid_when": true,
    "lsb_order": false,
    "content": {
        "class": "EmptyArray"
    },
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak._v2.forms.bitmaskedform.BitMaskedForm(
                mask="u8",
                content=ak._v2.forms.emptyform.EmptyForm(),
                valid_when=True,
                lsb_order=False,
            )
        )
        == "BitMaskedForm('u8', EmptyForm(), True, False)"
    )
    assert (
        repr(
            ak._v2.forms.bitmaskedform.BitMaskedForm(
                mask="u8",
                content=ak._v2.forms.emptyform.EmptyForm(),
                valid_when=True,
                lsb_order=False,
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "BitMaskedForm('u8', EmptyForm(), True, False, has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.bitmaskedform.BitMaskedForm(
        "u8", ak._v2.forms.emptyform.EmptyForm(), True, False
    ).tolist(verbose=False) == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {"class": "EmptyArray"},
    }
    assert ak._v2.forms.bitmaskedform.BitMaskedForm(
        "u8", ak._v2.forms.emptyform.EmptyForm(), True, False
    ).tolist() == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.bitmaskedform.BitMaskedForm(
        mask="u8",
        content=ak._v2.forms.emptyform.EmptyForm(),
        valid_when=True,
        lsb_order=False,
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "BitMaskedArray",
            "mask": "u8",
            "valid_when": True,
            "lsb_order": False,
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "BitMaskedArray",
            "mask": "i64",
            "valid_when": True,
            "lsb_order": False,
            "content": {"class": "EmptyArray"},
        }
    ).tolist() == {
        "class": "BitMaskedArray",
        "mask": "i64",
        "valid_when": True,
        "lsb_order": False,
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "BitMaskedArray",
            "mask": "u8",
            "valid_when": True,
            "lsb_order": False,
            "content": {"class": "EmptyArray"},
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_UnmaskedForm():
    assert (
        str(ak._v2.forms.unmaskedform.UnmaskedForm(ak._v2.forms.emptyform.EmptyForm()))
        == """{
    "class": "UnmaskedArray",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak._v2.forms.unmaskedform.UnmaskedForm(
                ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "UnmaskedArray",
    "content": {
        "class": "EmptyArray"
    },
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak._v2.forms.unmaskedform.UnmaskedForm(
                content=ak._v2.forms.emptyform.EmptyForm()
            )
        )
        == "UnmaskedForm(EmptyForm())"
    )
    assert (
        repr(
            ak._v2.forms.unmaskedform.UnmaskedForm(
                content=ak._v2.forms.emptyform.EmptyForm(),
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "UnmaskedForm(EmptyForm(), has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.unmaskedform.UnmaskedForm(
        ak._v2.forms.emptyform.EmptyForm()
    ).tolist(verbose=False) == {
        "class": "UnmaskedArray",
        "content": {"class": "EmptyArray"},
    }
    assert ak._v2.forms.unmaskedform.UnmaskedForm(
        ak._v2.forms.emptyform.EmptyForm()
    ).tolist() == {
        "class": "UnmaskedArray",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.unmaskedform.UnmaskedForm(
        content=ak._v2.forms.emptyform.EmptyForm(),
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "UnmaskedArray",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {"class": "UnmaskedArray", "content": {"class": "EmptyArray"}}
    ).tolist() == {
        "class": "UnmaskedArray",
        "content": {
            "class": "EmptyArray",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "UnmaskedArray",
            "content": {"class": "EmptyArray"},
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "UnmaskedArray",
        "content": {"class": "EmptyArray"},
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="Python 2.7, 3.5 have unstable dict order."
)
def test_UnionForm():
    assert (
        str(
            ak._v2.forms.unionform.UnionForm(
                "i8",
                "i32",
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
            )
        )
        == """{
    "class": "UnionArray",
    "tags": "i8",
    "index": "i32",
    "contents": [
        {
            "class": "EmptyArray"
        },
        "bool"
    ]
}"""
    )
    assert (
        str(
            ak._v2.forms.unionform.UnionForm(
                "i8",
                "u32",
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
            )
        )
        == """{
    "class": "UnionArray",
    "tags": "i8",
    "index": "u32",
    "contents": [
        {
            "class": "EmptyArray"
        },
        "bool"
    ]
}"""
    )
    assert (
        str(
            ak._v2.forms.unionform.UnionForm(
                "i8",
                "i64",
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
            )
        )
        == """{
    "class": "UnionArray",
    "tags": "i8",
    "index": "i64",
    "contents": [
        {
            "class": "EmptyArray"
        },
        "bool"
    ]
}"""
    )
    assert (
        str(
            ak._v2.forms.unionform.UnionForm(
                "i8",
                "i32",
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "UnionArray",
    "tags": "i8",
    "index": "i32",
    "contents": [
        {
            "class": "EmptyArray"
        },
        "bool"
    ],
    "has_identifier": true,
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )

    assert (
        repr(
            ak._v2.forms.unionform.UnionForm(
                "i8",
                "i32",
                [
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
            )
        )
        == "UnionForm('i8', 'i32', [EmptyForm(), NumpyForm('bool')])"
    )
    assert (
        repr(
            ak._v2.forms.unionform.UnionForm(
                tags="i8",
                index="i32",
                contents=[
                    ak._v2.forms.emptyform.EmptyForm(),
                    ak._v2.forms.numpyform.NumpyForm("bool"),
                ],
                has_identifier=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "UnionForm('i8', 'i32', [EmptyForm(), NumpyForm('bool')], has_identifier=True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak._v2.forms.unionform.UnionForm(
        "i8",
        "i32",
        [ak._v2.forms.emptyform.EmptyForm(), ak._v2.forms.numpyform.NumpyForm("bool")],
    ).tolist(verbose=False) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
    }
    assert ak._v2.forms.unionform.UnionForm(
        "i8",
        "i32",
        [ak._v2.forms.emptyform.EmptyForm(), ak._v2.forms.numpyform.NumpyForm("bool")],
    ).tolist() == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {
                "class": "EmptyArray",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.unionform.UnionForm(
        tags="i8",
        index="i32",
        contents=[
            ak._v2.forms.emptyform.EmptyForm(),
            ak._v2.forms.numpyform.NumpyForm("bool"),
        ],
        has_identifier=True,
        parameters={"x": 123},
        form_key="hello",
    ).tolist(verbose=False) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i32",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
        }
    ).tolist() == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {
                "class": "EmptyArray",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "u32",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
        }
    ).tolist() == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "u32",
        "contents": [
            {
                "class": "EmptyArray",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
        }
    ).tolist() == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i64",
        "contents": [
            {
                "class": "EmptyArray",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert ak._v2.forms.from_iter(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i32",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
            "has_identifier": True,
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).tolist(verbose=False) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "has_identifier": True,
        "parameters": {"x": 123},
        "form_key": "hello",
    }
