# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


@pytest.mark.skip(reason="unimplemented UnknownType")
def test_UnknownType():
    assert str(ak._v2.types.UnknownType()) == "unknown"
    assert str(ak._v2.types.UnknownType({"x": 123})) == 'unknown[parameters={"x": 123}]'
    assert str(ak._v2.types.UnknownType(None, "override")) == "override"
    assert str(ak._v2.types.UnknownType({"x": 123}, "override")) == "override"
    assert (
        str(ak._v2.types.UnknownType({"__categorical__": True}))
        == "categorical[type=unknown]"
    )
    assert (
        str(ak._v2.types.UnknownType({"__categorical__": True, "x": 123}))
        == 'categorical[type=unknown[parameters={"x": 123}]]'
    )
    assert (
        str(ak._v2.types.UnknownType({"__categorical__": True}, "override"))
        == "categorical[type=override]"
    )

    assert repr(ak._v2.types.UnknownType()) == "UnknownType()"
    assert (
        repr(
            ak._v2.types.UnknownType(
                parameters={"__categorical__": True}, typestr="override"
            )
        )
        == "UnknownType(parameters={'__categorical__': True}, typestr='override')"
    )


@pytest.mark.skip(reason="unimplemented NumpyType")
def test_NumpyType():
    assert str(ak._v2.types.NumpyType("bool")) == "bool"
    assert str(ak._v2.types.NumpyType("int8")) == "int8"
    assert str(ak._v2.types.NumpyType("uint8")) == "uint8"
    assert str(ak._v2.types.NumpyType("int16")) == "int16"
    assert str(ak._v2.types.NumpyType("uint16")) == "uint16"
    assert str(ak._v2.types.NumpyType("int32")) == "int32"
    assert str(ak._v2.types.NumpyType("uint32")) == "uint32"
    assert str(ak._v2.types.NumpyType("int64")) == "int64"
    assert str(ak._v2.types.NumpyType("uint64")) == "uint64"
    assert str(ak._v2.types.NumpyType("float16")) == "float16"
    assert str(ak._v2.types.NumpyType("float32")) == "float32"
    assert str(ak._v2.types.NumpyType("float64")) == "float64"
    assert str(ak._v2.types.NumpyType("float128")) == "float128"
    assert str(ak._v2.types.NumpyType("complex64")) == "complex64"
    assert str(ak._v2.types.NumpyType("complex128")) == "complex128"
    assert str(ak._v2.types.NumpyType("complex256")) == "complex256"
    assert str(ak._v2.types.NumpyType("bool", {"x": 123})) == 'bool[parameters={"x": 123}]'
    assert str(ak._v2.types.NumpyType("bool", None, "override")) == "override"
    assert str(ak._v2.types.NumpyType("bool", {"x": 123}, "override")) == "override"
    assert (
        str(ak._v2.types.NumpyType("bool", {"__categorical__": True}))
        == "categorical[type=bool]"
    )
    assert (
        str(ak._v2.types.NumpyType("bool", {"__categorical__": True, "x": 123}))
        == 'categorical[type=bool[parameters={"x": 123}]]'
    )
    assert (
        str(ak._v2.types.NumpyType("bool", {"__categorical__": True}, "override"))
        == "categorical[type=override]"
    )
    assert str(ak._v2.types.NumpyType("datetime64")) == "datetime64"
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "Y"}))
        == 'datetime64[unit="Y"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "M"}))
        == 'datetime64[unit="M"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "W"}))
        == 'datetime64[unit="W"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "D"}))
        == 'datetime64[unit="D"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "h"}))
        == 'datetime64[unit="h"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "m"}))
        == 'datetime64[unit="m"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "s"}))
        == 'datetime64[unit="s"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "ms"}))
        == 'datetime64[unit="ms"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "us"}))
        == 'datetime64[unit="us"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "ns"}))
        == 'datetime64[unit="ns"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "ps"}))
        == 'datetime64[unit="ps"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "fs"}))
        == 'datetime64[unit="fs"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "as"}))
        == 'datetime64[unit="as"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "10s"}))
        == 'datetime64[unit="10s"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "1s"}))
        == 'datetime64[unit="s"]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"__unit__": "s", "x": 123}))
        == 'datetime64[unit="s", parameters={"x": 123}]'
    )
    assert (
        str(ak._v2.types.NumpyType("datetime64", {"x": 123}))
        == 'datetime64[parameters={"x": 123}]'
    )
    assert str(ak._v2.types.NumpyType("timedelta64")) == "timedelta64"
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "Y"}))
        == 'timedelta64[unit="Y"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "M"}))
        == 'timedelta64[unit="M"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "W"}))
        == 'timedelta64[unit="W"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "D"}))
        == 'timedelta64[unit="D"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "h"}))
        == 'timedelta64[unit="h"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "m"}))
        == 'timedelta64[unit="m"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "s"}))
        == 'timedelta64[unit="s"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "ms"}))
        == 'timedelta64[unit="ms"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "us"}))
        == 'timedelta64[unit="us"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "ns"}))
        == 'timedelta64[unit="ns"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "ps"}))
        == 'timedelta64[unit="ps"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "fs"}))
        == 'timedelta64[unit="fs"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "as"}))
        == 'timedelta64[unit="as"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "10s"}))
        == 'timedelta64[unit="10s"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "1s"}))
        == 'timedelta64[unit="s"]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"__unit__": "s", "x": 123}))
        == 'timedelta64[unit="s", parameters={"x": 123}]'
    )
    assert (
        str(ak._v2.types.NumpyType("timedelta64", {"x": 123}))
        == 'timedelta64[parameters={"x": 123}]'
    )
    assert str(ak._v2.types.NumpyType("uint8", {"__array__": "char"})) == "char"
    assert str(ak._v2.types.NumpyType("uint8", {"__array__": "byte"})) == "byte"

    assert repr(ak._v2.types.NumpyType(dtype="bool")) == "NumpyType('bool')"
    assert (
        repr(
            ak._v2.types.NumpyType(
                dtype="bool", parameters={"__categorical__": True}, typestr="override"
            )
        )
        == "NumpyType('bool', parameters={'__categorical__': True}, typestr='override')"
    )
    assert (
        repr(ak._v2.types.NumpyType(dtype="datetime64", parameters={"__unit__": "s"}))
        == 'NumpyType("datetime64", parameters={"__unit__": "s"})'
    )
    assert (
        repr(ak._v2.types.NumpyType(dtype="uint8", parameters={"__array__": "char"}))
        == 'NumpyType("uint8", parameters={"__array__": "char"})'
    )
    assert (
        repr(ak._v2.types.NumpyType(dtype="uint8", parameters={"__array__": "byte"}))
        == 'NumpyType("uint8", parameters={"__array__": "byte"})'
    )


@pytest.mark.skip(reason="unimplemented RegularType")
def test_RegularType():
    assert str(ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10)) == "10 * unknown"
    assert str(ak._v2.types.RegularType(ak._v2.types.UnknownType(), 0)) == "0 * unknown"
    with pytest.raises(ValueError):
        ak._v2.types.RegularType(ak._v2.types.UnknownType(), -1)
    assert (
        str(ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10, {"x": 123}))
        == '[10 * unknown, parameters={"x": 123}]'
    )
    assert (
        str(ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10, None, "override"))
        == "override"
    )
    assert (
        str(ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10, {"x": 123}, "override"))
        == "override"
    )
    assert (
        str(ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10, {"__categorical__": True}))
        == "categorical[type=10 * unknown]"
    )
    assert (
        str(
            ak._v2.types.RegularType(
                ak._v2.types.UnknownType(), 10, {"__categorical__": True, "x": 123}
            )
        )
        == 'categorical[type=[10 * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.RegularType(
                ak._v2.types.UnknownType(), 10, {"__categorical__": True}, "override"
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.RegularType(
                ak._v2.types.NumpyType("uint8", {"__array__": "char"}),
                {"__array__": "string"},
                10,
            )
        )
        == "string[10]"
    )
    assert (
        str(
            ak._v2.types.RegularType(
                ak._v2.types.NumpyType("uint8", {"__array__": "byte"}),
                {"__array__": "bytestring"},
                10,
            )
        )
        == "bytes[10]"
    )

    assert (
        repr(ak._v2.types.RegularType(type=ak._v2.types.UnknownType(), size=10))
        == "RegularType(UnknownType(), 10)"
    )
    assert (
        repr(
            ak._v2.types.RegularType(
                type=ak._v2.types.UnknownType(),
                size=10,
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == 'RegularType(UnknownType(), 10, parameters={"__categorical__": True}, typestr="override")'
    )
    assert (
        repr(
            ak._v2.types.RegularType(
                type=ak._v2.types.NumpyType(
                    dtype="uint8", parameters={"__array__": "char"}
                ),
                parameters={"__array__": "string"},
                size=10,
            )
        )
        == 'RegularType(NumpyType("uint8", parameters={"__array__": "char"}), parameters={"__array__": "string"}, 10)'
    )
    assert (
        repr(
            ak._v2.types.RegularType(
                type=ak._v2.types.NumpyType(
                    dtype="uint8", parameters={"__array__": "byte"}
                ),
                parameters={"__array__": "bytestring"},
                size=10,
            )
        )
        == 'RegularType(NumpyType("uint8", parameters={"__array__": "byte"}), parameters={"__array__": "bytestring"}, 10)'
    )


@pytest.mark.skip(reason="unimplemented ListType")
def test_ListType():
    assert str(ak._v2.types.ListType(ak._v2.types.UnknownType())) == "var * unknown"
    assert (
        str(ak._v2.types.ListType(ak._v2.types.UnknownType(), {"x": 123}))
        == '[var * unknown, parameters={"x": 123}]'
    )
    assert (
        str(ak._v2.types.ListType(ak._v2.types.UnknownType(), None, "override")) == "override"
    )
    assert (
        str(ak._v2.types.ListType(ak._v2.types.UnknownType(), {"x": 123}, "override"))
        == "override"
    )
    assert (
        str(ak._v2.types.ListType(ak._v2.types.UnknownType(), {"__categorical__": True}))
        == "categorical[type=var * unknown]"
    )
    assert (
        str(
            ak._v2.types.ListType(
                ak._v2.types.UnknownType(), {"__categorical__": True, "x": 123}
            )
        )
        == 'categorical[type=[var * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.ListType(
                ak._v2.types.UnknownType(), {"__categorical__": True}, "override"
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.ListType(
                ak._v2.types.NumpyType("uint8", {"__array__": "char"}),
                {"__array__": "string"},
            )
        )
        == "string"
    )
    assert (
        str(
            ak._v2.types.ListType(
                ak._v2.types.NumpyType("uint8", {"__array__": "byte"}),
                {"__array__": "bytestring"},
            )
        )
        == "bytes"
    )

    assert (
        repr(ak._v2.types.ListType(type=ak._v2.types.UnknownType()))
        == "ListType(UnknownType())"
    )
    assert (
        repr(
            ak._v2.types.ListType(
                type=ak._v2.types.UnknownType(),
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == 'ListType(UnknownType(), parameters={"__categorical__": True}, typestr="override")'
    )
    assert (
        repr(
            ak._v2.types.ListType(
                type=ak._v2.types.NumpyType(
                    dtype="uint8", parameters={"__array__": "char"}
                ),
                parameters={"__array__": "string"},
            )
        )
        == 'ListType(NumpyType("uint8", parameters={"__array__": "char"}), parameters={"__array__": "string"})'
    )
    assert (
        repr(
            ak._v2.types.ListType(
                type=ak._v2.types.NumpyType(
                    dtype="uint8", parameters={"__array__": "byte"}
                ),
                parameters={"__array__": "bytestring"},
            )
        )
        == 'ListType(NumpyType("uint8", parameters={"__array__": "byte"}), parameters={"__array__": "bytestring"})'
    )


@pytest.mark.skip(reason="unimplemented RecordType")
def test_RecordType():
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")], None
            )
        )
        == "(unknown, bool)"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")], ["x", "y"]
            )
        )
        == '{"x": unknown, "y": bool}'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__record__": "Name"},
            )
        )
        == "Name[unknown, bool]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__record__": "Name"},
            )
        )
        == 'Name["x": unknown, "y": bool]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                None,
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                None,
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__record__": "Name"},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__record__": "Name"},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")], None, {"x": 123}
            )
        )
        == 'tuple[[unknown, bool], parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"x": 123},
            )
        )
        == 'struct[["x", "y"], [unknown, bool], parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__record__": "Name", "x": 123},
            )
        )
        == 'Name[unknown, bool, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__record__": "Name", "x": 123},
            )
        )
        == 'Name["x": unknown, "y": bool, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__record__": "Name", "x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__record__": "Name", "x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__categorical__": "True"},
            )
        )
        == "categorical[type=(unknown, bool)]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__categorical__": "True"},
            )
        )
        == 'categorical[type={"x": unknown, "y": bool}]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__record__": "Name", "__categorical__": True},
            )
        )
        == "categorical[type=Name[unknown, bool]]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__record__": "Name", "__categorical__": True},
            )
        )
        == 'categorical[type=Name["x": unknown, "y": bool]]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__record__": "Name", "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__record__": "Name", "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=tuple[[unknown, bool], parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=struct[["x", "y"], [unknown, bool], parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__record__": "Name", "x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=Name[unknown, bool, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__record__": "Name", "x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=Name["x": unknown, "y": bool, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                None,
                {"__record__": "Name", "x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.RecordType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                ["x", "y"],
                {"__record__": "Name", "x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )

    assert (
        repr(
            ak._v2.types.RecordType(
                types=[ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                recordlookup=None,
            )
        )
        == 'RecordType([UnknownType(), NumpyType("bool")], None)'
    )
    assert (
        repr(
            ak._v2.types.RecordType(
                types=[ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                recordlookup=["x", "y"],
            )
        )
        == 'RecordType([UnknownType(), NumpyType("bool")], ["x", "y"])'
    )
    assert (
        repr(
            ak._v2.types.RecordType(
                types=[ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                recordlookup=None,
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == 'RecordType([UnknownType(), NumpyType("bool")], None, parameters={"__record__": "Name", "x": 123, "__categorical__": True}, typestr="override")'
    )
    assert (
        repr(
            ak._v2.types.RecordType(
                types=[ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                recordlookup=["x", "y"],
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == 'RecordType([UnknownType(), NumpyType("bool")], ["x", "y"], parameters={"__record__": "Name", "x": 123, "__categorical__": True}, typestr="override")'
    )


@pytest.mark.skip(reason="unimplemented OptionType")
def test_OptionType():
    assert str(ak._v2.types.OptionType(ak._v2.types.UnknownType())) == "?unknown"
    assert (
        str(ak._v2.types.OptionType(ak._v2.types.ListType(ak._v2.types.UnknownType())))
        == "option[var * unknown]"
    )
    assert (
        str(ak._v2.types.OptionType(ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10)))
        == "option[10 * unknown]"
    )
    assert (
        str(ak._v2.types.OptionType(ak._v2.types.UnknownType(), {"x": 123}))
        == 'option[unknown, parameters={"x": 123}]'
    )
    assert (
        str(ak._v2.types.OptionType(ak._v2.types.ListType(ak._v2.types.UnknownType()), {"x": 123}))
        == 'option[var * unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10), {"x": 123}
            )
        )
        == 'option[10 * unknown, parameters={"x": 123}]'
    )
    assert (
        str(ak._v2.types.OptionType(ak._v2.types.UnknownType(), None, "override")) == "override"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.ListType(ak._v2.types.UnknownType()), None, "override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10), None, "override"
            )
        )
        == "override"
    )
    assert (
        str(ak._v2.types.OptionType(ak._v2.types.UnknownType(), {"x": 123}, "override"))
        == "override"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.ListType(ak._v2.types.UnknownType()), {"x": 123}, "override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10), {"x": 123}, "override"
            )
        )
        == "override"
    )
    assert (
        str(ak._v2.types.OptionType(ak._v2.types.UnknownType(), {"__categorical__": True}))
        == "categorical[type=?unknown]"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.ListType(ak._v2.types.UnknownType()), {"__categorical__": True}
            )
        )
        == "categorical[type=option[var * unknown]]"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10),
                {"__categorical__": True},
            )
        )
        == "categorical[type=option[10 * unknown]]"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.UnknownType(), {"x": 123, "__categorical__": True}
            )
        )
        == 'categorical[type=option[unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.ListType(ak._v2.types.UnknownType()),
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=option[var * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10),
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=option[10 * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.UnknownType(), {"__categorical__": True}, "override"
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.ListType(ak._v2.types.UnknownType()),
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10),
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.UnknownType(), {"x": 123, "__categorical__": True}, "override"
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.ListType(ak._v2.types.UnknownType()),
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.OptionType(
                ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10),
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )

    assert (
        repr(ak._v2.types.OptionType(type=ak._v2.types.UnknownType()))
        == "OptionType(UnknownType())"
    )
    assert (
        repr(ak._v2.types.OptionType(type=ak._v2.types.ListType(ak._v2.types.UnknownType())))
        == "OptionType(ListType(UnknownType()))"
    )
    assert (
        repr(
            ak._v2.types.OptionType(
                type=ak._v2.types.RegularType(ak._v2.types.UnknownType(), 10),
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == 'OptionType(RegularType(UnknownType(), 10), parameters={"x": 123, "__categorical__": True}, typestr="override")'
    )


@pytest.mark.skip(reason="unimplemented UnionType")
def test_UnionType():
    assert (
        str(ak._v2.types.UnionType([ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")]))
        == "union[unknown, bool]"
    )
    assert (
        str(
            ak._v2.types.UnionType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")], {"x": 123}
            )
        )
        == 'union[unknown, bool, parameters={"x": 123}]'
    )
    assert (
        str(
            ak._v2.types.UnionType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")], None, "override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.UnionType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                {"x": 123},
                "override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak._v2.types.UnionType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                {"__categorical__": True},
            )
        )
        == "categorical[type=union[unknown, bool]]"
    )
    assert (
        str(
            ak._v2.types.UnionType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                {"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=union[unknown, bool, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak._v2.types.UnionType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                {"__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak._v2.types.UnionType(
                [ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                {"x": 123, "__categorical__": True},
                "override",
            )
        )
        == "categorical[type=override]"
    )

    assert (
        repr(
            ak._v2.types.UnionType(
                types=[ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")]
            )
        )
        == 'UnionType([UnknownType(), NumpyType("bool")])'
    )
    assert (
        repr(
            ak._v2.types.UnionType(
                types=[ak._v2.types.UnknownType(), ak._v2.types.NumpyType("bool")],
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == 'UnionType([UnknownType(), NumpyType("bool")], parameters={"x": 123, "__categorical__": True}, typestr="override")'
    )


@pytest.mark.skip(reason="unimplemented ArrayType")
def test_ArrayType():
    assert str(ak._v2.types.ArrayType(ak._v2.types.UnknownType(), 10)) == "10 * unknown"
    assert str(ak._v2.types.ArrayType(ak._v2.types.UnknownType(), 0)) == "0 * unknown"
    with pytest.raises(ValueError):
        ak._v2.types.ArrayType(ak._v2.types.UnknownType(), -1)

    # ArrayType should not have these arguments (should not be a Type subclass)
    with pytest.raises(TypeError):
        ak._v2.types.ArrayType(ak._v2.types.UnknownType(), 10, {"x": 123})
    with pytest.raises(TypeError):
        ak._v2.types.ArrayType(ak._v2.types.UnknownType(), 10, None, "override")

    assert (
        repr(ak._v2.types.ArrayType(type=ak._v2.types.UnknownType(), length=10))
        == "ArrayType(UnknownType(), 10)"
    )


def test_EmptyForm():
    pass


def test_NumpyForm():
    pass


def test_RegularForm():
    pass


def test_ListForm():
    pass


def test_ListOffsetForm():
    pass


def test_RecordForm():
    pass


def test_IndexedForm():
    pass


def test_IndexedOptionForm():
    pass


def test_ByteMaskedForm():
    pass


def test_BitMaskedForm():
    pass


def test_UnmaskedForm():
    pass


def test_UnionForm():
    pass


def test_VirtualForm():
    pass
