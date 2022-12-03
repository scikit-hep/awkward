# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_UnknownType():
    assert str(ak.types.unknowntype.UnknownType()) == "unknown"
    assert (
        str(ak.types.unknowntype.UnknownType(parameters={"x": 123}))
        == 'unknown[parameters={"x": 123}]'
    )
    assert (
        str(ak.types.unknowntype.UnknownType(parameters=None, typestr="override"))
        == "override"
    )
    assert (
        str(ak.types.unknowntype.UnknownType(parameters={"x": 123}, typestr="override"))
        == "override"
    )
    assert (
        str(ak.types.unknowntype.UnknownType(parameters={"__categorical__": True}))
        == "categorical[type=unknown]"
    )
    assert (
        str(
            ak.types.unknowntype.UnknownType(
                parameters={"__categorical__": True, "x": 123}
            )
        )
        == 'categorical[type=unknown[parameters={"x": 123}]]'
    )
    assert (
        str(
            ak.types.unknowntype.UnknownType(
                parameters={"__categorical__": True}, typestr="override"
            )
        )
        == "categorical[type=override]"
    )

    assert repr(ak.types.unknowntype.UnknownType()) == "UnknownType()"
    assert (
        repr(
            ak.types.unknowntype.UnknownType(
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
    assert str(ak.types.numpytype.NumpyType("bool")) == "bool"
    assert str(ak.types.numpytype.NumpyType("int8")) == "int8"
    assert str(ak.types.numpytype.NumpyType("uint8")) == "uint8"
    assert str(ak.types.numpytype.NumpyType("int16")) == "int16"
    assert str(ak.types.numpytype.NumpyType("uint16")) == "uint16"
    assert str(ak.types.numpytype.NumpyType("int32")) == "int32"
    assert str(ak.types.numpytype.NumpyType("uint32")) == "uint32"
    assert str(ak.types.numpytype.NumpyType("int64")) == "int64"
    assert str(ak.types.numpytype.NumpyType("uint64")) == "uint64"
    if hasattr(np, "float16"):
        assert str(ak.types.numpytype.NumpyType("float16")) == "float16"
    assert str(ak.types.numpytype.NumpyType("float32")) == "float32"
    assert str(ak.types.numpytype.NumpyType("float64")) == "float64"
    if hasattr(np, "float128"):
        assert str(ak.types.numpytype.NumpyType("float128")) == "float128"
    assert str(ak.types.numpytype.NumpyType("complex64")) == "complex64"
    assert str(ak.types.numpytype.NumpyType("complex128")) == "complex128"
    if hasattr(np, "complex256"):
        assert str(ak.types.numpytype.NumpyType("complex256")) == "complex256"
    assert (
        str(ak.types.numpytype.NumpyType("bool", parameters={"x": 123}))
        == 'bool[parameters={"x": 123}]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("bool", parameters=None, typestr="override"))
        == "override"
    )
    assert (
        str(
            ak.types.numpytype.NumpyType(
                "bool", parameters={"x": 123}, typestr="override"
            )
        )
        == "override"
    )
    assert (
        str(ak.types.numpytype.NumpyType("bool", parameters={"__categorical__": True}))
        == "categorical[type=bool]"
    )
    assert (
        str(
            ak.types.numpytype.NumpyType(
                "bool", parameters={"__categorical__": True, "x": 123}
            )
        )
        == 'categorical[type=bool[parameters={"x": 123}]]'
    )
    assert (
        str(
            ak.types.numpytype.NumpyType(
                "bool", parameters={"__categorical__": True}, typestr="override"
            )
        )
        == "categorical[type=override]"
    )
    assert str(ak.types.numpytype.NumpyType("datetime64")) == "datetime64"
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "Y"}))
        == 'datetime64[unit="Y"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "M"}))
        == 'datetime64[unit="M"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "W"}))
        == 'datetime64[unit="W"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "D"}))
        == 'datetime64[unit="D"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "h"}))
        == 'datetime64[unit="h"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "m"}))
        == 'datetime64[unit="m"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "s"}))
        == 'datetime64[unit="s"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "ms"}))
        == 'datetime64[unit="ms"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "us"}))
        == 'datetime64[unit="us"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "ns"}))
        == 'datetime64[unit="ns"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "ps"}))
        == 'datetime64[unit="ps"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "fs"}))
        == 'datetime64[unit="fs"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "as"}))
        == 'datetime64[unit="as"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "10s"}))
        == 'datetime64[unit="10s"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"__unit__": "1s"}))
        == 'datetime64[unit="s"]'
    )
    assert (
        str(
            ak.types.numpytype.NumpyType(
                "datetime64", parameters={"__unit__": "s", "x": 123}
            )
        )
        == 'datetime64[unit="s", parameters={"x": 123}]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("datetime64", parameters={"x": 123}))
        == 'datetime64[parameters={"x": 123}]'
    )
    assert str(ak.types.numpytype.NumpyType("timedelta64")) == "timedelta64"
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "Y"}))
        == 'timedelta64[unit="Y"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "M"}))
        == 'timedelta64[unit="M"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "W"}))
        == 'timedelta64[unit="W"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "D"}))
        == 'timedelta64[unit="D"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "h"}))
        == 'timedelta64[unit="h"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "m"}))
        == 'timedelta64[unit="m"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "s"}))
        == 'timedelta64[unit="s"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "ms"}))
        == 'timedelta64[unit="ms"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "us"}))
        == 'timedelta64[unit="us"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "ns"}))
        == 'timedelta64[unit="ns"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "ps"}))
        == 'timedelta64[unit="ps"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "fs"}))
        == 'timedelta64[unit="fs"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "as"}))
        == 'timedelta64[unit="as"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "10s"}))
        == 'timedelta64[unit="10s"]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"__unit__": "1s"}))
        == 'timedelta64[unit="s"]'
    )
    assert (
        str(
            ak.types.numpytype.NumpyType(
                "timedelta64", parameters={"__unit__": "s", "x": 123}
            )
        )
        == 'timedelta64[unit="s", parameters={"x": 123}]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("timedelta64", parameters={"x": 123}))
        == 'timedelta64[parameters={"x": 123}]'
    )
    assert (
        str(ak.types.numpytype.NumpyType("uint8", parameters={"__array__": "char"}))
        == "char"
    )
    assert (
        str(ak.types.numpytype.NumpyType("uint8", parameters={"__array__": "byte"}))
        == "byte"
    )
    assert repr(ak.types.numpytype.NumpyType(primitive="bool")) == "NumpyType('bool')"
    assert (
        repr(
            ak.types.numpytype.NumpyType(
                primitive="bool",
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "NumpyType('bool', parameters={'__categorical__': True}, typestr='override')"
    )
    assert (
        repr(
            ak.types.numpytype.NumpyType(
                primitive="datetime64", parameters={"__unit__": "s"}
            )
        )
        == "NumpyType('datetime64', parameters={'__unit__': 's'})"
    )
    assert (
        repr(
            ak.types.numpytype.NumpyType(
                primitive="uint8", parameters={"__array__": "char"}
            )
        )
        == "NumpyType('uint8', parameters={'__array__': 'char'})"
    )
    assert (
        repr(
            ak.types.numpytype.NumpyType(
                primitive="uint8", parameters={"__array__": "byte"}
            )
        )
        == "NumpyType('uint8', parameters={'__array__': 'byte'})"
    )


def test_RegularType():
    assert (
        str(ak.types.regulartype.RegularType(ak.types.unknowntype.UnknownType(), 10))
        == "10 * unknown"
    )
    assert (
        str(ak.types.regulartype.RegularType(ak.types.unknowntype.UnknownType(), 0))
        == "0 * unknown"
    )
    with pytest.raises(ValueError):
        ak.types.regulartype.RegularType(ak.types.unknowntype.UnknownType(), -1)
    assert (
        str(
            ak.types.regulartype.RegularType(
                ak.types.unknowntype.UnknownType(), 10, parameters={"x": 123}
            )
        )
        == '[10 * unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.regulartype.RegularType(
                ak.types.unknowntype.UnknownType(),
                10,
                parameters=None,
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.regulartype.RegularType(
                ak.types.unknowntype.UnknownType(),
                10,
                parameters={"x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.regulartype.RegularType(
                ak.types.unknowntype.UnknownType(),
                10,
                parameters={"__categorical__": True},
            )
        )
        == "categorical[type=10 * unknown]"
    )
    assert (
        str(
            ak.types.regulartype.RegularType(
                ak.types.unknowntype.UnknownType(),
                10,
                parameters={"__categorical__": True, "x": 123},
            )
        )
        == 'categorical[type=[10 * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak.types.regulartype.RegularType(
                ak.types.unknowntype.UnknownType(),
                10,
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.regulartype.RegularType(
                ak.types.numpytype.NumpyType("uint8", parameters={"__array__": "char"}),
                10,
                parameters={"__array__": "string"},
            )
        )
        == "string[10]"
    )
    assert (
        str(
            ak.types.regulartype.RegularType(
                ak.types.numpytype.NumpyType("uint8", parameters={"__array__": "byte"}),
                10,
                parameters={"__array__": "bytestring"},
            )
        )
        == "bytes[10]"
    )

    assert (
        repr(
            ak.types.regulartype.RegularType(
                content=ak.types.unknowntype.UnknownType(), size=10
            )
        )
        == "RegularType(UnknownType(), 10)"
    )
    assert (
        repr(
            ak.types.regulartype.RegularType(
                content=ak.types.unknowntype.UnknownType(),
                size=10,
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "RegularType(UnknownType(), 10, parameters={'__categorical__': True}, typestr='override')"
    )
    assert (
        repr(
            ak.types.regulartype.RegularType(
                content=ak.types.numpytype.NumpyType(
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
            ak.types.regulartype.RegularType(
                content=ak.types.numpytype.NumpyType(
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
        str(ak.types.listtype.ListType(ak.types.unknowntype.UnknownType()))
        == "var * unknown"
    )
    assert (
        str(
            ak.types.listtype.ListType(
                ak.types.unknowntype.UnknownType(), parameters={"x": 123}
            )
        )
        == '[var * unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.listtype.ListType(
                ak.types.unknowntype.UnknownType(), parameters=None, typestr="override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.listtype.ListType(
                ak.types.unknowntype.UnknownType(),
                parameters={"x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.listtype.ListType(
                ak.types.unknowntype.UnknownType(), parameters={"__categorical__": True}
            )
        )
        == "categorical[type=var * unknown]"
    )
    assert (
        str(
            ak.types.listtype.ListType(
                ak.types.unknowntype.UnknownType(),
                parameters={"__categorical__": True, "x": 123},
            )
        )
        == 'categorical[type=[var * unknown, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak.types.listtype.ListType(
                ak.types.unknowntype.UnknownType(),
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.listtype.ListType(
                ak.types.numpytype.NumpyType("uint8", parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
            )
        )
        == "string"
    )
    assert (
        str(
            ak.types.listtype.ListType(
                ak.types.numpytype.NumpyType("uint8", parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
            )
        )
        == "bytes"
    )

    assert (
        repr(ak.types.listtype.ListType(content=ak.types.unknowntype.UnknownType()))
        == "ListType(UnknownType())"
    )
    assert (
        repr(
            ak.types.listtype.ListType(
                content=ak.types.unknowntype.UnknownType(),
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "ListType(UnknownType(), parameters={'__categorical__': True}, typestr='override')"
    )
    assert (
        repr(
            ak.types.listtype.ListType(
                content=ak.types.numpytype.NumpyType(
                    primitive="uint8", parameters={"__array__": "char"}
                ),
                parameters={"__array__": "string"},
            )
        )
        == "ListType(NumpyType('uint8', parameters={'__array__': 'char'}), parameters={'__array__': 'string'})"
    )
    assert (
        repr(
            ak.types.listtype.ListType(
                content=ak.types.numpytype.NumpyType(
                    primitive="uint8", parameters={"__array__": "byte"}
                ),
                parameters={"__array__": "bytestring"},
            )
        )
        == "ListType(NumpyType('uint8', parameters={'__array__': 'byte'}), parameters={'__array__': 'bytestring'})"
    )


def test_RecordType():
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
            )
        )
        == "(unknown, bool)"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
            )
        )
        == "{x: unknown, y: bool}"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__record__": "Name"},
            )
        )
        == "Name[unknown, bool]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__record__": "Name"},
            )
        )
        == "Name[x: unknown, y: bool]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters=None,
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters=None,
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__record__": "Name"},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__record__": "Name"},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"x": 123},
            )
        )
        == 'tuple[[unknown, bool], parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"x": 123},
            )
        )
        == 'struct[{x: unknown, y: bool}, parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__record__": "Name", "x": 123},
            )
        )
        == 'Name[unknown, bool, parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__record__": "Name", "x": 123},
            )
        )
        == 'Name[x: unknown, y: bool, parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__record__": "Name", "x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__record__": "Name", "x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__categorical__": True},
            )
        )
        == "categorical[type=(unknown, bool)]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__categorical__": True},
            )
        )
        == "categorical[type={x: unknown, y: bool}]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__record__": "Name", "__categorical__": True},
            )
        )
        == "categorical[type=Name[unknown, bool]]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__record__": "Name", "__categorical__": True},
            )
        )
        == "categorical[type=Name[x: unknown, y: bool]]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__record__": "Name", "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__record__": "Name", "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=tuple[[unknown, bool], parameters={"x": 123}]]'
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=struct[{x: unknown, y: bool}, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=Name[unknown, bool, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=Name[x: unknown, y: bool, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                None,
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.recordtype.RecordType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                ["x", "y"],
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )

    assert (
        repr(
            ak.types.recordtype.RecordType(
                contents=[
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                fields=None,
            )
        )
        == "RecordType([UnknownType(), NumpyType('bool')], None)"
    )
    assert (
        repr(
            ak.types.recordtype.RecordType(
                contents=[
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                fields=["x", "y"],
            )
        )
        == "RecordType([UnknownType(), NumpyType('bool')], ['x', 'y'])"
    )
    assert (
        repr(
            ak.types.recordtype.RecordType(
                contents=[
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
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
            ak.types.recordtype.RecordType(
                contents=[
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                fields=None,
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
            )
        )
        == "RecordType([UnknownType(), NumpyType('bool')], None, parameters={'__record__': 'Name', 'x': 123, '__categorical__': True})"
    )
    assert (
        repr(
            ak.types.recordtype.RecordType(
                contents=[
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                fields=["x", "y"],
                parameters={"__record__": "Name", "x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "RecordType([UnknownType(), NumpyType('bool')], ['x', 'y'], parameters={'__record__': 'Name', 'x': 123, '__categorical__': True}, typestr='override')"
    )


def test_OptionType():
    assert (
        str(ak.types.optiontype.OptionType(ak.types.unknowntype.UnknownType()))
        == "?unknown"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.listtype.ListType(ak.types.unknowntype.UnknownType())
            )
        )
        == "option[var * unknown]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.regulartype.RegularType(ak.types.unknowntype.UnknownType(), 10)
            )
        )
        == "option[10 * unknown]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.unknowntype.UnknownType(), parameters={"x": 123}
            )
        )
        == 'option[unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.listtype.ListType(ak.types.unknowntype.UnknownType()),
                parameters={"x": 123},
            )
        )
        == 'option[var * unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.regulartype.RegularType(
                    ak.types.unknowntype.UnknownType(), 10
                ),
                parameters={"x": 123},
            )
        )
        == 'option[10 * unknown, parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.unknowntype.UnknownType(), parameters=None, typestr="override"
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.listtype.ListType(ak.types.unknowntype.UnknownType()),
                parameters=None,
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.regulartype.RegularType(
                    ak.types.unknowntype.UnknownType(), 10
                ),
                parameters=None,
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.unknowntype.UnknownType(),
                parameters={"x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.listtype.ListType(ak.types.unknowntype.UnknownType()),
                parameters={"x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.regulartype.RegularType(
                    ak.types.unknowntype.UnknownType(), 10
                ),
                parameters={"x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.unknowntype.UnknownType(), parameters={"__categorical__": True}
            )
        )
        == "?categorical[type=unknown]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.listtype.ListType(ak.types.unknowntype.UnknownType()),
                parameters={"__categorical__": True},
            )
        )
        == "option[categorical[type=var * unknown]]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.regulartype.RegularType(
                    ak.types.unknowntype.UnknownType(), 10
                ),
                parameters={"__categorical__": True},
            )
        )
        == "option[categorical[type=10 * unknown]]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.unknowntype.UnknownType(),
                parameters={"x": 123, "__categorical__": True},
            )
        )
        == 'option[categorical[type=unknown], parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.listtype.ListType(ak.types.unknowntype.UnknownType()),
                parameters={"x": 123, "__categorical__": True},
            )
        )
        == 'option[categorical[type=var * unknown], parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.regulartype.RegularType(
                    ak.types.unknowntype.UnknownType(), 10
                ),
                parameters={"x": 123, "__categorical__": True},
            )
        )
        == 'option[categorical[type=10 * unknown], parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.unknowntype.UnknownType(),
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.listtype.ListType(ak.types.unknowntype.UnknownType()),
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.regulartype.RegularType(
                    ak.types.unknowntype.UnknownType(), 10
                ),
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.unknowntype.UnknownType(),
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.listtype.ListType(ak.types.unknowntype.UnknownType()),
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.optiontype.OptionType(
                ak.types.regulartype.RegularType(
                    ak.types.unknowntype.UnknownType(), 10
                ),
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )

    assert (
        repr(ak.types.optiontype.OptionType(content=ak.types.unknowntype.UnknownType()))
        == "OptionType(UnknownType())"
    )
    assert (
        repr(
            ak.types.optiontype.OptionType(
                content=ak.types.listtype.ListType(ak.types.unknowntype.UnknownType())
            )
        )
        == "OptionType(ListType(UnknownType()))"
    )
    assert (
        repr(
            ak.types.optiontype.OptionType(
                content=ak.types.regulartype.RegularType(
                    ak.types.unknowntype.UnknownType(), 10
                ),
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "OptionType(RegularType(UnknownType(), 10), parameters={'x': 123, '__categorical__': True}, typestr='override')"
    )


def test_UnionType():
    assert (
        str(
            ak.types.uniontype.UnionType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ]
            )
        )
        == "union[unknown, bool]"
    )
    assert (
        str(
            ak.types.uniontype.UnionType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                parameters={"x": 123},
            )
        )
        == 'union[unknown, bool, parameters={"x": 123}]'
    )
    assert (
        str(
            ak.types.uniontype.UnionType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                parameters=None,
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.uniontype.UnionType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                parameters={"x": 123},
                typestr="override",
            )
        )
        == "override"
    )
    assert (
        str(
            ak.types.uniontype.UnionType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                parameters={"__categorical__": True},
            )
        )
        == "categorical[type=union[unknown, bool]]"
    )
    assert (
        str(
            ak.types.uniontype.UnionType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                parameters={"x": 123, "__categorical__": True},
            )
        )
        == 'categorical[type=union[unknown, bool, parameters={"x": 123}]]'
    )
    assert (
        str(
            ak.types.uniontype.UnionType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                parameters={"__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )
    assert (
        str(
            ak.types.uniontype.UnionType(
                [
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "categorical[type=override]"
    )

    assert (
        repr(
            ak.types.uniontype.UnionType(
                contents=[
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ]
            )
        )
        == "UnionType([UnknownType(), NumpyType('bool')])"
    )
    assert (
        repr(
            ak.types.uniontype.UnionType(
                contents=[
                    ak.types.unknowntype.UnknownType(),
                    ak.types.numpytype.NumpyType("bool"),
                ],
                parameters={"x": 123, "__categorical__": True},
                typestr="override",
            )
        )
        == "UnionType([UnknownType(), NumpyType('bool')], parameters={'x': 123, '__categorical__': True}, typestr='override')"
    )


def test_ArrayType():
    assert (
        str(ak.types.arraytype.ArrayType(ak.types.unknowntype.UnknownType(), 10))
        == "10 * unknown"
    )
    assert (
        str(ak.types.arraytype.ArrayType(ak.types.unknowntype.UnknownType(), 0))
        == "0 * unknown"
    )
    with pytest.raises(ValueError):
        ak.types.arraytype.ArrayType(ak.types.unknowntype.UnknownType(), -1)

    # ArrayType should not have these arguments (should not be a Type subclass)
    with pytest.raises(TypeError):
        ak.types.arraytype.ArrayType(ak.types.unknowntype.UnknownType(), 10, {"x": 123})
    with pytest.raises(TypeError):
        ak.types.arraytype.ArrayType(
            ak.types.unknowntype.UnknownType(), 10, None, typestr="override"
        )

    assert (
        repr(
            ak.types.arraytype.ArrayType(
                content=ak.types.unknowntype.UnknownType(), length=10
            )
        )
        == "ArrayType(UnknownType(), 10)"
    )


def test_EmptyForm():
    assert (
        str(ak.forms.emptyform.EmptyForm())
        == """{
    "class": "EmptyArray"
}"""
    )
    assert (
        str(ak.forms.emptyform.EmptyForm(parameters={"x": 123}, form_key="hello"))
        == """{
    "class": "EmptyArray",
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert repr(ak.forms.emptyform.EmptyForm()) == "EmptyForm()"
    assert (
        repr(ak.forms.emptyform.EmptyForm(parameters={"x": 123}, form_key="hello"))
        == "EmptyForm(parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.emptyform.EmptyForm().to_dict(verbose=False) == {
        "class": "EmptyArray"
    }
    assert ak.forms.emptyform.EmptyForm().to_dict() == {
        "class": "EmptyArray",
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.emptyform.EmptyForm(
        parameters={"x": 123}, form_key="hello"
    ).to_dict(verbose=False) == {
        "class": "EmptyArray",
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict({"class": "EmptyArray"}).to_dict() == {
        "class": "EmptyArray",
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "EmptyArray",
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict() == {
        "class": "EmptyArray",
        "parameters": {"x": 123},
        "form_key": "hello",
    }


@pytest.mark.skipif(
    ak._util.win,
    reason="NumPy does not have float16, float128, and complex256 -- on Windows",
)
def test_NumpyForm():
    assert (
        str(ak.forms.numpyform.NumpyForm("bool"))
        == """{
    "class": "NumpyArray",
    "primitive": "bool"
}"""
    )
    assert repr(ak.forms.numpyform.NumpyForm(primitive="bool")) == "NumpyForm('bool')"
    assert (
        repr(
            ak.forms.numpyform.NumpyForm(
                primitive="bool",
                inner_shape=[1, 2, 3],
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "NumpyForm('bool', inner_shape=(1, 2, 3), parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.numpyform.NumpyForm("bool").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak.forms.numpyform.NumpyForm("bool").to_dict() == {
        "class": "NumpyArray",
        "primitive": "bool",
        "inner_shape": [],
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.numpyform.NumpyForm(
        "bool",
        inner_shape=[1, 2, 3],
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
        "inner_shape": [1, 2, 3],
        "parameters": {"x": 123},
        "form_key": "hello",
    }

    assert ak.forms.numpyform.NumpyForm("bool").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak.forms.numpyform.NumpyForm("int8").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int8",
    }
    assert ak.forms.numpyform.NumpyForm("uint8").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint8",
    }
    assert ak.forms.numpyform.NumpyForm("int16").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int16",
    }
    assert ak.forms.numpyform.NumpyForm("uint16").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint16",
    }
    assert ak.forms.numpyform.NumpyForm("int32").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int32",
    }
    assert ak.forms.numpyform.NumpyForm("uint32").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint32",
    }
    assert ak.forms.numpyform.NumpyForm("int64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int64",
    }
    assert ak.forms.numpyform.NumpyForm("uint64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint64",
    }
    if hasattr(np, "float16"):
        assert ak.forms.numpyform.NumpyForm("float16").to_dict(verbose=False) == {
            "class": "NumpyArray",
            "primitive": "float16",
        }
    assert ak.forms.numpyform.NumpyForm("float32").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float32",
    }
    assert ak.forms.numpyform.NumpyForm("float64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }
    if hasattr(np, "float128"):
        assert ak.forms.numpyform.NumpyForm("float128").to_dict(verbose=False) == {
            "class": "NumpyArray",
            "primitive": "float128",
        }
    assert ak.forms.numpyform.NumpyForm("complex64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex64",
    }
    assert ak.forms.numpyform.NumpyForm("complex128").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex128",
    }
    if hasattr(np, "complex256"):
        assert ak.forms.numpyform.NumpyForm("complex256").to_dict(verbose=False) == {
            "class": "NumpyArray",
            "primitive": "complex256",
        }
    assert ak.forms.numpyform.NumpyForm("datetime64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
    }
    assert ak.forms.numpyform.NumpyForm(
        "datetime64", parameters={"__unit__": "s"}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s"},
    }
    assert ak.forms.numpyform.NumpyForm(
        "datetime64", parameters={"__unit__": "s", "x": 123}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s", "x": 123},
    }
    assert ak.forms.numpyform.NumpyForm("timedelta64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
    }
    assert ak.forms.numpyform.NumpyForm(
        "timedelta64", parameters={"__unit__": "s"}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s"},
    }
    assert ak.forms.numpyform.NumpyForm(
        "timedelta64", parameters={"__unit__": "s", "x": 123}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s", "x": 123},
    }

    assert ak.forms.numpyform.from_dtype(np.dtype("bool")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("int8")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int8",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("uint8")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint8",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("int16")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int16",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("uint16")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint16",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("int32")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int32",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("uint32")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint32",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("int64")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int64",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("uint64")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint64",
    }
    if hasattr(np, "float16"):
        assert ak.forms.numpyform.from_dtype(np.dtype("float16")).to_dict(
            verbose=False
        ) == {
            "class": "NumpyArray",
            "primitive": "float16",
        }
    assert ak.forms.numpyform.from_dtype(np.dtype("float32")).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "float32",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("float64")).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }
    if hasattr(np, "float128"):
        assert ak.forms.numpyform.from_dtype(np.dtype("float128")).to_dict(
            verbose=False
        ) == {
            "class": "NumpyArray",
            "primitive": "float128",
        }
    assert ak.forms.numpyform.from_dtype(np.dtype("complex64")).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "complex64",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("complex128")).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "complex128",
    }
    if hasattr(np, "complex256"):
        assert ak.forms.numpyform.from_dtype(np.dtype("complex256")).to_dict(
            verbose=False
        ) == {
            "class": "NumpyArray",
            "primitive": "complex256",
        }
    assert ak.forms.numpyform.from_dtype(np.dtype("M8")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("M8[s]")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s"},
    }
    assert ak.forms.numpyform.from_dtype(
        np.dtype("M8[s]"), parameters={"x": 123}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s", "x": 123},
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("m8")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
    }
    assert ak.forms.numpyform.from_dtype(np.dtype("m8[s]")).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s"},
    }
    assert ak.forms.numpyform.from_dtype(
        np.dtype("m8[s]"), parameters={"x": 123}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s", "x": 123},
    }
    assert ak.forms.numpyform.from_dtype(np.dtype(("bool", (1, 2, 3)))).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "bool",
        "inner_shape": [1, 2, 3],
    }
    with pytest.raises(TypeError):
        ak.forms.from_dtype(np.dtype("O")).to_dict(verbose=False)
    with pytest.raises(TypeError):
        ak.forms.from_dtype(np.dtype([("one", np.int64), ("two", np.float64)])).to_dict(
            verbose=False
        )
    assert ak.forms.from_dict("bool").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak.forms.from_dict("int8").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int8",
    }
    assert ak.forms.from_dict("uint8").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint8",
    }
    assert ak.forms.from_dict("int16").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int16",
    }
    assert ak.forms.from_dict("uint16").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint16",
    }
    assert ak.forms.from_dict("int32").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int32",
    }
    assert ak.forms.from_dict("uint32").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint32",
    }
    assert ak.forms.from_dict("int64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "int64",
    }
    assert ak.forms.from_dict("uint64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "uint64",
    }
    if hasattr(np, "float16"):
        assert ak.forms.from_dict("float16").to_dict(verbose=False) == {
            "class": "NumpyArray",
            "primitive": "float16",
        }
    assert ak.forms.from_dict("float32").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float32",
    }
    assert ak.forms.from_dict("float64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }
    if hasattr(np, "float128"):
        assert ak.forms.from_dict("float128").to_dict(verbose=False) == {
            "class": "NumpyArray",
            "primitive": "float128",
        }
    assert ak.forms.from_dict("complex64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex64",
    }
    assert ak.forms.from_dict("complex128").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex128",
    }
    if hasattr(np, "complex256"):
        assert ak.forms.from_dict("complex256").to_dict(verbose=False) == {
            "class": "NumpyArray",
            "primitive": "complex256",
        }
    assert ak.forms.from_dict("datetime64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
    }
    assert ak.forms.from_dict(
        {
            "class": "NumpyArray",
            "primitive": "datetime64",
            "parameters": {"__unit__": "s"},
        }
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s"},
    }
    assert ak.forms.from_dict(
        {
            "class": "NumpyArray",
            "primitive": "datetime64",
            "parameters": {"__unit__": "s", "x": 123},
        }
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
        "parameters": {"__unit__": "s", "x": 123},
    }
    assert ak.forms.from_dict("timedelta64").to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
    }
    assert ak.forms.from_dict(
        {
            "class": "NumpyArray",
            "primitive": "timedelta64",
            "parameters": {"__unit__": "s"},
        }
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s"},
    }
    assert ak.forms.from_dict(
        {
            "class": "NumpyArray",
            "primitive": "timedelta64",
            "parameters": {"__unit__": "s", "x": 123},
        }
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
        "parameters": {"__unit__": "s", "x": 123},
    }

    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "bool"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "bool",
    }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "int8"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int8",
    }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "uint8"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "uint8",
    }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "int16"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int16",
    }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "uint16"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "uint16",
    }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "int32"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int32",
    }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "uint32"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "uint32",
    }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "int64"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "int64",
    }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "uint64"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "uint64",
    }
    if hasattr(np, "float16"):
        assert ak.forms.from_dict(
            {"class": "NumpyArray", "primitive": "float16"}
        ).to_dict(verbose=False) == {
            "class": "NumpyArray",
            "primitive": "float16",
        }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "float32"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "float32",
    }
    assert ak.forms.from_dict({"class": "NumpyArray", "primitive": "float64"}).to_dict(
        verbose=False
    ) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }
    if hasattr(np, "float128"):
        assert ak.forms.from_dict(
            {"class": "NumpyArray", "primitive": "float128"}
        ).to_dict(verbose=False) == {
            "class": "NumpyArray",
            "primitive": "float128",
        }
    assert ak.forms.from_dict(
        {"class": "NumpyArray", "primitive": "complex64"}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex64",
    }
    assert ak.forms.from_dict(
        {"class": "NumpyArray", "primitive": "complex128"}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "complex128",
    }
    if hasattr(np, "complex256"):
        assert ak.forms.from_dict(
            {"class": "NumpyArray", "primitive": "complex256"}
        ).to_dict(verbose=False) == {
            "class": "NumpyArray",
            "primitive": "complex256",
        }
    assert ak.forms.from_dict(
        {"class": "NumpyArray", "primitive": "datetime64"}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "datetime64",
    }
    assert ak.forms.from_dict(
        {"class": "NumpyArray", "primitive": "timedelta64"}
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "timedelta64",
    }

    assert ak.forms.from_dict(
        {
            "class": "NumpyArray",
            "primitive": "bool",
            "inner_shape": [1, 2, 3],
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "NumpyArray",
        "primitive": "bool",
        "inner_shape": [1, 2, 3],
        "parameters": {"x": 123},
        "form_key": "hello",
    }


def test_RegularForm():
    assert (
        str(ak.forms.regularform.RegularForm(ak.forms.emptyform.EmptyForm(), 10))
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
            ak.forms.regularform.RegularForm(
                ak.forms.emptyform.EmptyForm(),
                10,
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
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak.forms.regularform.RegularForm(
                content=ak.forms.emptyform.EmptyForm(), size=10
            )
        )
        == "RegularForm(EmptyForm(), 10)"
    )
    assert (
        repr(
            ak.forms.regularform.RegularForm(
                content=ak.forms.emptyform.EmptyForm(),
                size=10,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "RegularForm(EmptyForm(), 10, parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.regularform.RegularForm(ak.forms.emptyform.EmptyForm(), 10).to_dict(
        verbose=False
    ) == {
        "class": "RegularArray",
        "size": 10,
        "content": {"class": "EmptyArray"},
    }
    assert ak.forms.regularform.RegularForm(
        ak.forms.emptyform.EmptyForm(), 10
    ).to_dict() == {
        "class": "RegularArray",
        "size": 10,
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.regularform.RegularForm(
        content=ak.forms.emptyform.EmptyForm(),
        size=10,
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "RegularArray",
        "size": 10,
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {"class": "RegularArray", "size": 10, "content": {"class": "EmptyArray"}}
    ).to_dict() == {
        "class": "RegularArray",
        "size": 10,
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "RegularArray",
            "size": 10,
            "content": {"class": "EmptyArray"},
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "RegularArray",
        "size": 10,
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.regularform.RegularForm(
        ak.forms.numpyform.NumpyForm("bool"), 10
    ).to_dict() == {
        "class": "RegularArray",
        "content": {
            "class": "NumpyArray",
            "primitive": "bool",
            "inner_shape": [],
            "parameters": {},
            "form_key": None,
        },
        "size": 10,
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.regularform.RegularForm(
        ak.forms.numpyform.NumpyForm("bool"), 10
    ).to_dict(verbose=False) == {
        "class": "RegularArray",
        "content": "bool",
        "size": 10,
    }


def test_ListForm():
    assert (
        str(ak.forms.listform.ListForm("i32", "i32", ak.forms.emptyform.EmptyForm()))
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
        str(ak.forms.listform.ListForm("u32", "u32", ak.forms.emptyform.EmptyForm()))
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
        str(ak.forms.listform.ListForm("i64", "i64", ak.forms.emptyform.EmptyForm()))
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
            ak.forms.listform.ListForm(
                "i32",
                "i32",
                ak.forms.emptyform.EmptyForm(),
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
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak.forms.listform.ListForm(
                starts="i32", stops="i32", content=ak.forms.emptyform.EmptyForm()
            )
        )
        == "ListForm('i32', 'i32', EmptyForm())"
    )
    assert (
        repr(
            ak.forms.listform.ListForm(
                starts="i32",
                stops="i32",
                content=ak.forms.emptyform.EmptyForm(),
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "ListForm('i32', 'i32', EmptyForm(), parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.listform.ListForm(
        "i32", "i32", ak.forms.emptyform.EmptyForm()
    ).to_dict(verbose=False) == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {"class": "EmptyArray"},
    }
    assert ak.forms.listform.ListForm(
        "i32", "i32", ak.forms.emptyform.EmptyForm()
    ).to_dict() == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.listform.ListForm(
        starts="i32",
        stops="i32",
        content=ak.forms.emptyform.EmptyForm(),
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {
            "class": "ListArray",
            "starts": "i32",
            "stops": "i32",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "ListArray",
            "starts": "u32",
            "stops": "u32",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "ListArray",
        "starts": "u32",
        "stops": "u32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "ListArray",
            "starts": "i64",
            "stops": "i64",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "ListArray",
        "starts": "i64",
        "stops": "i64",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "ListArray",
            "starts": "i32",
            "stops": "i32",
            "content": {"class": "EmptyArray"},
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "ListArray",
        "starts": "i32",
        "stops": "i32",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }


def test_ListOffsetForm():
    assert (
        str(
            ak.forms.listoffsetform.ListOffsetForm(
                "i32", ak.forms.emptyform.EmptyForm()
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
            ak.forms.listoffsetform.ListOffsetForm(
                "u32", ak.forms.emptyform.EmptyForm()
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
            ak.forms.listoffsetform.ListOffsetForm(
                "i64", ak.forms.emptyform.EmptyForm()
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
            ak.forms.listoffsetform.ListOffsetForm(
                "i32",
                ak.forms.emptyform.EmptyForm(),
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
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak.forms.listoffsetform.ListOffsetForm(
                offsets="i32", content=ak.forms.emptyform.EmptyForm()
            )
        )
        == "ListOffsetForm('i32', EmptyForm())"
    )
    assert (
        repr(
            ak.forms.listoffsetform.ListOffsetForm(
                offsets="i32",
                content=ak.forms.emptyform.EmptyForm(),
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "ListOffsetForm('i32', EmptyForm(), parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.listoffsetform.ListOffsetForm(
        "i32", ak.forms.emptyform.EmptyForm()
    ).to_dict(verbose=False) == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {"class": "EmptyArray"},
    }
    assert ak.forms.listoffsetform.ListOffsetForm(
        "i32", ak.forms.emptyform.EmptyForm()
    ).to_dict() == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.listoffsetform.ListOffsetForm(
        offsets="i32",
        content=ak.forms.emptyform.EmptyForm(),
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {
            "class": "ListOffsetArray",
            "offsets": "i32",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "ListOffsetArray",
            "offsets": "u32",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "ListOffsetArray",
        "offsets": "u32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "ListOffsetArray",
            "offsets": "i32",
            "content": {"class": "EmptyArray"},
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "ListOffsetArray",
        "offsets": "i32",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }


def test_RecordForm():
    assert (
        str(
            ak.forms.recordform.RecordForm(
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
                None,
            )
        )
        == """{
    "class": "RecordArray",
    "fields": null,
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
            ak.forms.recordform.RecordForm(
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
                ["x", "y"],
            )
        )
        == """{
    "class": "RecordArray",
    "fields": [
        "x",
        "y"
    ],
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
            ak.forms.recordform.RecordForm(
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
                None,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "RecordArray",
    "fields": null,
    "contents": [
        {
            "class": "EmptyArray"
        },
        "bool"
    ],
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        str(
            ak.forms.recordform.RecordForm(
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
                ["x", "y"],
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "RecordArray",
    "fields": [
        "x",
        "y"
    ],
    "contents": [
        {
            "class": "EmptyArray"
        },
        "bool"
    ],
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )

    assert (
        repr(
            ak.forms.recordform.RecordForm(
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
                None,
            )
        )
        == "RecordForm([EmptyForm(), NumpyForm('bool')], None)"
    )
    assert (
        repr(
            ak.forms.recordform.RecordForm(
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
                ["x", "y"],
            )
        )
        == "RecordForm([EmptyForm(), NumpyForm('bool')], ['x', 'y'])"
    )
    assert (
        repr(
            ak.forms.recordform.RecordForm(
                contents=[
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
                fields=None,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "RecordForm([EmptyForm(), NumpyForm('bool')], None, parameters={'x': 123}, form_key='hello')"
    )
    assert (
        repr(
            ak.forms.recordform.RecordForm(
                contents=[
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
                fields=["x", "y"],
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "RecordForm([EmptyForm(), NumpyForm('bool')], ['x', 'y'], parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.recordform.RecordForm(
        [ak.forms.emptyform.EmptyForm(), ak.forms.numpyform.NumpyForm("bool")],
        None,
    ).to_dict(verbose=False) == {
        "class": "RecordArray",
        "fields": None,
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
    }
    assert ak.forms.recordform.RecordForm(
        [ak.forms.emptyform.EmptyForm(), ak.forms.numpyform.NumpyForm("bool")],
        ["x", "y"],
    ).to_dict(verbose=False) == {
        "class": "RecordArray",
        "fields": ["x", "y"],
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
    }
    assert ak.forms.recordform.RecordForm(
        [ak.forms.emptyform.EmptyForm(), ak.forms.numpyform.NumpyForm("bool")],
        None,
    ).to_dict() == {
        "class": "RecordArray",
        "fields": None,
        "contents": [
            {
                "class": "EmptyArray",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.recordform.RecordForm(
        [ak.forms.emptyform.EmptyForm(), ak.forms.numpyform.NumpyForm("bool")],
        ["x", "y"],
    ).to_dict() == {
        "class": "RecordArray",
        "fields": ["x", "y"],
        "contents": [
            {
                "class": "EmptyArray",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.recordform.RecordForm(
        contents=[
            ak.forms.emptyform.EmptyForm(),
            ak.forms.numpyform.NumpyForm("bool"),
        ],
        fields=None,
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "RecordArray",
        "fields": None,
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.recordform.RecordForm(
        contents=[
            ak.forms.emptyform.EmptyForm(),
            ak.forms.numpyform.NumpyForm("bool"),
        ],
        fields=["x", "y"],
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "RecordArray",
        "fields": ["x", "y"],
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": None,
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
        }
    ).to_dict() == {
        "class": "RecordArray",
        "fields": None,
        "contents": [
            {
                "class": "EmptyArray",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": ["x", "y"],
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
        }
    ).to_dict() == {
        "class": "RecordArray",
        "fields": ["x", "y"],
        "contents": [
            {
                "class": "EmptyArray",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": None,
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "RecordArray",
        "fields": None,
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": ["x", "y"],
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "RecordArray",
        "fields": ["x", "y"],
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "parameters": {"x": 123},
        "form_key": "hello",
    }


def test_IndexedForm():
    assert (
        str(ak.forms.indexedform.IndexedForm("i32", ak.forms.emptyform.EmptyForm()))
        == """{
    "class": "IndexedArray",
    "index": "i32",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(ak.forms.indexedform.IndexedForm("u32", ak.forms.emptyform.EmptyForm()))
        == """{
    "class": "IndexedArray",
    "index": "u32",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(ak.forms.indexedform.IndexedForm("i64", ak.forms.emptyform.EmptyForm()))
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
            ak.forms.indexedform.IndexedForm(
                "i32",
                ak.forms.emptyform.EmptyForm(),
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
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak.forms.indexedform.IndexedForm(
                index="i32", content=ak.forms.emptyform.EmptyForm()
            )
        )
        == "IndexedForm('i32', EmptyForm())"
    )
    assert (
        repr(
            ak.forms.indexedform.IndexedForm(
                index="i32",
                content=ak.forms.emptyform.EmptyForm(),
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "IndexedForm('i32', EmptyForm(), parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.indexedform.IndexedForm(
        "i32", ak.forms.emptyform.EmptyForm()
    ).to_dict(verbose=False) == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
    }
    assert ak.forms.indexedform.IndexedForm(
        "i32", ak.forms.emptyform.EmptyForm()
    ).to_dict() == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.indexedform.IndexedForm(
        index="i32",
        content=ak.forms.emptyform.EmptyForm(),
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {
            "class": "IndexedArray",
            "index": "i32",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "IndexedArray",
            "index": "u32",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "IndexedArray",
        "index": "u32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "IndexedArray",
            "index": "i64",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "IndexedArray",
        "index": "i64",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "IndexedArray",
            "index": "i32",
            "content": {"class": "EmptyArray"},
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "IndexedArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }


def test_IndexedOptionForm():
    assert (
        str(
            ak.forms.indexedoptionform.IndexedOptionForm(
                "i32", ak.forms.emptyform.EmptyForm()
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
            ak.forms.indexedoptionform.IndexedOptionForm(
                "i64", ak.forms.emptyform.EmptyForm()
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
            ak.forms.indexedoptionform.IndexedOptionForm(
                "i32",
                ak.forms.emptyform.EmptyForm(),
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
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak.forms.indexedoptionform.IndexedOptionForm(
                index="i32", content=ak.forms.emptyform.EmptyForm()
            )
        )
        == "IndexedOptionForm('i32', EmptyForm())"
    )
    assert (
        repr(
            ak.forms.indexedoptionform.IndexedOptionForm(
                index="i32",
                content=ak.forms.emptyform.EmptyForm(),
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "IndexedOptionForm('i32', EmptyForm(), parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.indexedoptionform.IndexedOptionForm(
        "i32", ak.forms.emptyform.EmptyForm()
    ).to_dict(verbose=False) == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
    }
    assert ak.forms.indexedoptionform.IndexedOptionForm(
        "i32", ak.forms.emptyform.EmptyForm()
    ).to_dict() == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.indexedoptionform.IndexedOptionForm(
        index="i32",
        content=ak.forms.emptyform.EmptyForm(),
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {
            "class": "IndexedOptionArray",
            "index": "i32",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "IndexedOptionArray",
            "index": "i64",
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "IndexedOptionArray",
        "index": "i64",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "IndexedOptionArray",
            "index": "i32",
            "content": {"class": "EmptyArray"},
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "IndexedOptionArray",
        "index": "i32",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }


def test_ByteMaskedForm():
    assert (
        str(
            ak.forms.bytemaskedform.ByteMaskedForm(
                "i8", ak.forms.emptyform.EmptyForm(), True
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
            ak.forms.bytemaskedform.ByteMaskedForm(
                "i8", ak.forms.emptyform.EmptyForm(), False
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
            ak.forms.bytemaskedform.ByteMaskedForm(
                "i8",
                ak.forms.emptyform.EmptyForm(),
                True,
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
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak.forms.bytemaskedform.ByteMaskedForm(
                mask="i8", content=ak.forms.emptyform.EmptyForm(), valid_when=True
            )
        )
        == "ByteMaskedForm('i8', EmptyForm(), True)"
    )
    assert (
        repr(
            ak.forms.bytemaskedform.ByteMaskedForm(
                mask="i8",
                content=ak.forms.emptyform.EmptyForm(),
                valid_when=True,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "ByteMaskedForm('i8', EmptyForm(), True, parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.bytemaskedform.ByteMaskedForm(
        "i8", ak.forms.emptyform.EmptyForm(), True
    ).to_dict(verbose=False) == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {"class": "EmptyArray"},
    }
    assert ak.forms.bytemaskedform.ByteMaskedForm(
        "i8", ak.forms.emptyform.EmptyForm(), True
    ).to_dict() == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.bytemaskedform.ByteMaskedForm(
        mask="i8",
        content=ak.forms.emptyform.EmptyForm(),
        valid_when=True,
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {
            "class": "ByteMaskedArray",
            "mask": "i8",
            "valid_when": True,
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "ByteMaskedArray",
            "mask": "i64",
            "valid_when": True,
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "ByteMaskedArray",
        "mask": "i64",
        "valid_when": True,
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "ByteMaskedArray",
            "mask": "i8",
            "valid_when": True,
            "content": {"class": "EmptyArray"},
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "ByteMaskedArray",
        "mask": "i8",
        "valid_when": True,
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }


def test_BitMaskedForm():
    assert (
        str(
            ak.forms.bitmaskedform.BitMaskedForm(
                "u8", ak.forms.emptyform.EmptyForm(), True, True
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
            ak.forms.bitmaskedform.BitMaskedForm(
                "u8", ak.forms.emptyform.EmptyForm(), False, True
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
            ak.forms.bitmaskedform.BitMaskedForm(
                "u8", ak.forms.emptyform.EmptyForm(), True, False
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
            ak.forms.bitmaskedform.BitMaskedForm(
                "u8", ak.forms.emptyform.EmptyForm(), False, False
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
            ak.forms.bitmaskedform.BitMaskedForm(
                "u8",
                ak.forms.emptyform.EmptyForm(),
                True,
                False,
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
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(
            ak.forms.bitmaskedform.BitMaskedForm(
                mask="u8",
                content=ak.forms.emptyform.EmptyForm(),
                valid_when=True,
                lsb_order=False,
            )
        )
        == "BitMaskedForm('u8', EmptyForm(), True, False)"
    )
    assert (
        repr(
            ak.forms.bitmaskedform.BitMaskedForm(
                mask="u8",
                content=ak.forms.emptyform.EmptyForm(),
                valid_when=True,
                lsb_order=False,
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "BitMaskedForm('u8', EmptyForm(), True, False, parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.bitmaskedform.BitMaskedForm(
        "u8", ak.forms.emptyform.EmptyForm(), True, False
    ).to_dict(verbose=False) == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {"class": "EmptyArray"},
    }
    assert ak.forms.bitmaskedform.BitMaskedForm(
        "u8", ak.forms.emptyform.EmptyForm(), True, False
    ).to_dict() == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.bitmaskedform.BitMaskedForm(
        mask="u8",
        content=ak.forms.emptyform.EmptyForm(),
        valid_when=True,
        lsb_order=False,
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {
            "class": "BitMaskedArray",
            "mask": "u8",
            "valid_when": True,
            "lsb_order": False,
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "BitMaskedArray",
            "mask": "i64",
            "valid_when": True,
            "lsb_order": False,
            "content": {"class": "EmptyArray"},
        }
    ).to_dict() == {
        "class": "BitMaskedArray",
        "mask": "i64",
        "valid_when": True,
        "lsb_order": False,
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "BitMaskedArray",
            "mask": "u8",
            "valid_when": True,
            "lsb_order": False,
            "content": {"class": "EmptyArray"},
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "BitMaskedArray",
        "mask": "u8",
        "valid_when": True,
        "lsb_order": False,
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }


def test_UnmaskedForm():
    assert (
        str(ak.forms.unmaskedform.UnmaskedForm(ak.forms.emptyform.EmptyForm()))
        == """{
    "class": "UnmaskedArray",
    "content": {
        "class": "EmptyArray"
    }
}"""
    )
    assert (
        str(
            ak.forms.unmaskedform.UnmaskedForm(
                ak.forms.emptyform.EmptyForm(),
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == """{
    "class": "UnmaskedArray",
    "content": {
        "class": "EmptyArray"
    },
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )
    assert (
        repr(ak.forms.unmaskedform.UnmaskedForm(content=ak.forms.emptyform.EmptyForm()))
        == "UnmaskedForm(EmptyForm())"
    )
    assert (
        repr(
            ak.forms.unmaskedform.UnmaskedForm(
                content=ak.forms.emptyform.EmptyForm(),
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "UnmaskedForm(EmptyForm(), parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.unmaskedform.UnmaskedForm(ak.forms.emptyform.EmptyForm()).to_dict(
        verbose=False
    ) == {
        "class": "UnmaskedArray",
        "content": {"class": "EmptyArray"},
    }
    assert ak.forms.unmaskedform.UnmaskedForm(
        ak.forms.emptyform.EmptyForm()
    ).to_dict() == {
        "class": "UnmaskedArray",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.unmaskedform.UnmaskedForm(
        content=ak.forms.emptyform.EmptyForm(),
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "UnmaskedArray",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {"class": "UnmaskedArray", "content": {"class": "EmptyArray"}}
    ).to_dict() == {
        "class": "UnmaskedArray",
        "content": {
            "class": "EmptyArray",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "UnmaskedArray",
            "content": {"class": "EmptyArray"},
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "UnmaskedArray",
        "content": {"class": "EmptyArray"},
        "parameters": {"x": 123},
        "form_key": "hello",
    }


def test_UnionForm():
    assert (
        str(
            ak.forms.unionform.UnionForm(
                "i8",
                "i32",
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
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
            ak.forms.unionform.UnionForm(
                "i8",
                "u32",
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
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
            ak.forms.unionform.UnionForm(
                "i8",
                "i64",
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
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
            ak.forms.unionform.UnionForm(
                "i8",
                "i32",
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
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
    "parameters": {
        "x": 123
    },
    "form_key": "hello"
}"""
    )

    assert (
        repr(
            ak.forms.unionform.UnionForm(
                "i8",
                "i32",
                [
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
            )
        )
        == "UnionForm('i8', 'i32', [EmptyForm(), NumpyForm('bool')])"
    )
    assert (
        repr(
            ak.forms.unionform.UnionForm(
                tags="i8",
                index="i32",
                contents=[
                    ak.forms.emptyform.EmptyForm(),
                    ak.forms.numpyform.NumpyForm("bool"),
                ],
                parameters={"x": 123},
                form_key="hello",
            )
        )
        == "UnionForm('i8', 'i32', [EmptyForm(), NumpyForm('bool')], parameters={'x': 123}, form_key='hello')"
    )

    assert ak.forms.unionform.UnionForm(
        "i8",
        "i32",
        [ak.forms.emptyform.EmptyForm(), ak.forms.numpyform.NumpyForm("bool")],
    ).to_dict(verbose=False) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
    }
    assert ak.forms.unionform.UnionForm(
        "i8",
        "i32",
        [ak.forms.emptyform.EmptyForm(), ak.forms.numpyform.NumpyForm("bool")],
    ).to_dict() == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {
                "class": "EmptyArray",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.unionform.UnionForm(
        tags="i8",
        index="i32",
        contents=[
            ak.forms.emptyform.EmptyForm(),
            ak.forms.numpyform.NumpyForm("bool"),
        ],
        parameters={"x": 123},
        form_key="hello",
    ).to_dict(verbose=False) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "parameters": {"x": 123},
        "form_key": "hello",
    }
    assert ak.forms.from_dict(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i32",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
        }
    ).to_dict() == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {
                "class": "EmptyArray",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "u32",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
        }
    ).to_dict() == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "u32",
        "contents": [
            {
                "class": "EmptyArray",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
        }
    ).to_dict() == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i64",
        "contents": [
            {
                "class": "EmptyArray",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "bool",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {},
        "form_key": None,
    }
    assert ak.forms.from_dict(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i32",
            "contents": [
                {"class": "EmptyArray"},
                "bool",
            ],
            "parameters": {"x": 123},
            "form_key": "hello",
        }
    ).to_dict(verbose=False) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i32",
        "contents": [
            {"class": "EmptyArray"},
            "bool",
        ],
        "parameters": {"x": 123},
        "form_key": "hello",
    }
