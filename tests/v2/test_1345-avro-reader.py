import pytest  # noqa: F401
import awkward._v2 as ak
from awkward._v2.operations.convert.ak_from_avro import from_avro_file  # noqa: F401


def test_int():
    data = [34, 45, 67, 78, 23, 89, 6, 33, 96, 73]
    assert (
        ak.from_avro_file(
            file_name="tests/samples/int_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )


def test_boolean():
    data = [True, False, False, True, True, True, False, False, False, False]
    assert (
        ak.from_avro_file(
            file_name="tests/samples/bool_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )


def test_long():
    data = [12, 435, 56, 12, 67, 34, 89, 2345, 536, 8769]
    assert (
        ak.from_avro_file(
            file_name="tests/samples/long_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )


def test_float():
    data = [
        12.456,
        57.1234,
        798.23,
        345.687,
        908.23,
        65.89,
        43.57,
        745.79,
        532.68,
        3387.684,
    ]
    assert (
        pytest.approx(
            ak.from_avro_file(
                file_name="tests/samples/float_test_data.avro", reader_lang="py"
            ).to_list()
        )
        == data
    )


def test_double():
    data = [
        12.456,
        57.1234,
        798.23,
        345.687,
        908.23,
        65.89,
        43.57,
        745.79,
        532.68,
        3387.684,
    ]
    assert (
        pytest.approx(
            ak.from_avro_file(
                file_name="tests/samples/double_test_data.avro", reader_lang="py"
            ).to_list()
        )
        == data
    )


def test_bytes():
    data = [
        bytes("hello", "utf-8"),
        bytes("hii", "utf-8"),
        bytes("byee", "utf-8"),
        bytes("pink", "utf-8"),
        bytes("blue", "utf-8"),
        bytes("red", "utf-8"),
        bytes("chrome", "utf-8"),
        bytes("green", "utf-8"),
        bytes("black", "utf-8"),
        bytes("peach", "utf-8"),
    ]
    assert (
        pytest.approx(
            ak.from_avro_file(
                file_name="tests/samples/bytes_test_data.avro", reader_lang="py"
            ).to_list()
        )
        == data
    )


def test_string():
    raise NotImplementedError


def test_null():
    raise NotImplementedError
