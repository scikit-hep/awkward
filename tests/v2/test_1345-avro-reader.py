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
    data = [
        "Hello",
        "what",
        "should",
        "we",
        "do",
        "for",
        "this",
        "period",
        "of",
        "time",
    ]
    assert (
        ak.from_avro_file(
            file_name="tests/samples/string_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )


def test_fixed():
    data = [
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
    ]
    assert (
        ak.from_avro_file(
            file_name="tests/samples/fixed_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )


def test_null():
    raise NotImplementedError


def test_enum():
    data = ["TWO", "ONE", "FOUR", "THREE", "TWO", "ONE", "FOUR", "THREE", "TWO", "ONE"]
    raise NotImplementedError


def test_arrays_int():
    data = [
        [34, 556, 12],
        [34, 556, 12],
        [34, 532, 657],
        [236, 568, 12],
        [34, 556, 12],
        [34, 54, 967],
        [34, 556, 12],
        [34, 647, 12],
    ]
    raise NotImplementedError


def test_array_string():
    raise NotImplementedError


def test_array_enum():
    raise NotImplementedError


def test_Unions_x_null():
    raise NotImplementedError


def test_Unions_record_null():
    raise NotImplementedError


def test_null_X_Y():
    raise NotImplementedError


def test_records():
    data = [
        {"name": "Pierre-Simon Laplace", "age": 77, "Numbers": "TWO"},
        {"name": "Henry", "age": 36, "Numbers": "THREE"},
        {"name": "Harry", "age": 769, "Numbers": "ONE"},
        {"name": "Jim", "age": 3215, "Numbers": "FOUR"},
        {"name": "Lindsey", "age": 658, "Numbers": "TWO"},
        {"name": "Eduardo", "age": 25, "Numbers": "THREE"},
        {"name": "Aryan", "age": 6478, "Numbers": "FOUR"},
    ]
    raise NotImplementedError
