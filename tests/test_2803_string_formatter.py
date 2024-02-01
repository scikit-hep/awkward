# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pytest

import awkward as ak


def test_precision():
    stream = io.StringIO()
    ak.Array([1.0, 2.3456789]).show(stream=stream, precision=1)
    assert stream.getvalue() == "[1,\n 2]\n"

    stream.seek(0)
    ak.Array([1.0, 2.3456789]).show(stream=stream, precision=7)
    assert stream.getvalue() == "[1,\n 2.345679]\n"


def test_float():
    stream = io.StringIO()
    ak.Array([1.0, 2.3456789]).show(
        stream=stream, formatter={"float": "<FLOAT {}>".format}
    )
    assert stream.getvalue() == "[<FLOAT 1.0>,\n <FLOAT 2.3456789>]\n"

    stream.seek(0)
    ak.Array([1.0, 2.3456789]).show(
        stream=stream, formatter={"float_kind": "<FLOAT {}>".format}
    )
    assert stream.getvalue() == "[<FLOAT 1.0>,\n <FLOAT 2.3456789>]\n"


@pytest.mark.skipif(
    not hasattr(np, "float128"), reason="Only for 128-float supporting platforms"
)
def test_longfloat():
    stream = io.StringIO()
    ak.values_astype([1.0, 2.3456789], "float128").show(
        stream=stream, formatter={"longfloat": "<FLOAT {}>".format}
    )
    assert stream.getvalue() == "[<FLOAT 1.0>,\n <FLOAT 2.3456789>]\n"

    stream.seek(0)
    ak.values_astype([1.0, 2.3456789], "float128").show(
        stream=stream, formatter={"float_kind": "<FLOAT {}>".format}
    )
    assert stream.getvalue() == "[<FLOAT 1.0>,\n <FLOAT 2.3456789>]\n"


def test_complex():
    stream = io.StringIO()
    ak.Array([1.0j, 2.3456789j + 2]).show(
        stream=stream, formatter={"complexfloat": "<COMPLEX {}>".format}
    )
    assert stream.getvalue() == "[<COMPLEX 1j>,\n <COMPLEX (2+2.3456789j)>]\n"

    stream.seek(0)
    ak.Array([1.0j, 2.3456789j + 2]).show(
        stream=stream, formatter={"complex_kind": "<COMPLEX {}>".format}
    )
    assert stream.getvalue() == "[<COMPLEX 1j>,\n <COMPLEX (2+2.3456789j)>]\n"


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="Only for 256-complex supporting platforms"
)
def test_longcomplex():
    stream = io.StringIO()
    ak.values_astype([1.0j, 2.3456789j + 2], "complex256").show(
        stream=stream, formatter={"longcomplexfloat": "<COMPLEX {}>".format}
    )
    assert stream.getvalue() == "[<COMPLEX 1j>,\n <COMPLEX (2+2.3456789j)>]\n"

    stream.seek(0)
    ak.values_astype([1.0j, 2.3456789j + 2], "complex256").show(
        stream=stream, formatter={"complex_kind": "<COMPLEX {}>".format}
    )
    assert stream.getvalue() == "[<COMPLEX 1j>,\n <COMPLEX (2+2.3456789j)>]\n"


def test_int():
    stream = io.StringIO()
    ak.Array([1, 2]).show(stream=stream, formatter={"int": "<INT {}>".format})
    assert stream.getvalue() == "[<INT 1>,\n <INT 2>]\n"

    stream.seek(0)
    ak.Array([1, 2]).show(stream=stream, formatter={"int_kind": "<INT {}>".format})
    assert stream.getvalue() == "[<INT 1>,\n <INT 2>]\n"


def test_bool():
    stream = io.StringIO()
    ak.Array([True, False]).show(stream=stream, formatter={"bool": "<BOOL {}>".format})
    assert stream.getvalue() == "[<BOOL True>,\n <BOOL False>]\n"


def test_str():
    stream = io.StringIO()
    ak.Array(["hello", "world"]).show(
        stream=stream, formatter={"str": "<STRING {!r}>".format}
    )
    assert stream.getvalue() == "[<STRING 'hello'>,\n <STRING 'world'>]\n"

    stream.seek(0)
    ak.Array(["hello", "world"]).show(
        stream=stream, formatter={"str_kind": "<STRING {!r}>".format}
    )
    assert stream.getvalue() == "[<STRING 'hello'>,\n <STRING 'world'>]\n"


def test_bytes():
    stream = io.StringIO()
    ak.Array([b"hello", b"world"]).show(
        stream=stream, formatter={"bytes": "<BYTES {!r}>".format}
    )
    assert stream.getvalue() == "[<BYTES b'hello'>,\n <BYTES b'world'>]\n"

    stream.seek(0)
    ak.Array([b"hello", b"world"]).show(
        stream=stream, formatter={"str_kind": "<STRING {!r}>".format}
    )
    assert stream.getvalue() == "[<STRING b'hello'>,\n <STRING b'world'>]\n"


def test_datetime64():
    stream = io.StringIO()
    ak.Array([datetime(year=2023, month=1, day=1, hour=12, minute=1, second=30)]).show(
        stream=stream, formatter={"datetime": "<DT {}>".format}
    )
    assert stream.getvalue() == "[<DT 2023-01-01T12:01:30.000000>]\n"


def test_timedelta64():
    stream = io.StringIO()
    ak.Array([timedelta(days=1, hours=12, minutes=1, seconds=30)]).show(
        stream=stream, formatter={"datetime": "<TD {}>".format}
    )
    assert stream.getvalue() == "[129690000000 microseconds]\n"
