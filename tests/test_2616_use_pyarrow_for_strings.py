# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")

string = ak.Array(
    [
        ["\u03b1\u03b2\u03b3", ""],
        [],
        ["\u2192\u03b4\u03b5\u2190", "\u03b6z z\u03b6", "abc"],
    ]
)
bytestring = ak.Array(
    [
        ["\u03b1\u03b2\u03b3".encode(), b""],
        [],
        ["\u2192\u03b4\u03b5\u2190".encode(), "\u03b6z z\u03b6".encode(), b"abc"],
    ]
)

string_padded = ak.Array(
    [
        ["      αβγ      ", "               "],
        [],
        ["     →δε←      ", "     ζz zζ     ", "      abc      "],
    ]
)
bytestring_padded = ak.Array(
    [
        [b"    \xce\xb1\xce\xb2\xce\xb3     ", b"               "],
        [],
        [
            b"  \xe2\x86\x92\xce\xb4\xce\xb5\xe2\x86\x90   ",
            b"    \xce\xb6z z\xce\xb6    ",
            b"      abc      ",
        ],
    ]
)

string_repeats = ak.Array(
    [["foo123bar123baz", "foo", "bar"], ["123foo", "456bar", "foo123456bar"], []]
)

bytestring_repeats = ak.Array(
    [[b"foo123bar123baz", b"foo", b"bar"], [b"123foo", b"456bar", b"foo123456bar"], []]
)


def test_is_alnum():
    assert ak.str.is_alnum(string).tolist() == [
        [True, False],
        [],
        [False, False, True],
    ]
    assert ak.str.is_alnum(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, True],
    ]
    assert (
        ak.str.is_alnum(string).layout.form
        == ak.str.is_alnum(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_alnum(bytestring).layout.form
        == ak.str.is_alnum(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_alpha():
    assert ak.str.is_alpha(string).tolist() == [
        [True, False],
        [],
        [False, False, True],
    ]
    assert ak.str.is_alpha(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, True],
    ]
    assert (
        ak.str.is_alpha(string).layout.form
        == ak.str.is_alpha(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_alpha(bytestring).layout.form
        == ak.str.is_alpha(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_decimal():
    assert ak.str.is_decimal(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_decimal(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert (
        ak.str.is_decimal(string).layout.form
        == ak.str.is_decimal(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_decimal(bytestring).layout.form
        == ak.str.is_decimal(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_digit():
    assert ak.str.is_digit(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_digit(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]

    assert (
        ak.str.is_digit(string).layout.form
        == ak.str.is_digit(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_digit(bytestring).layout.form
        == ak.str.is_digit(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_lower():
    assert ak.str.is_lower(string).tolist() == [
        [True, False],
        [],
        [True, True, True],
    ]
    assert ak.str.is_lower(bytestring).tolist() == [
        [False, False],
        [],
        [False, True, True],
    ]
    assert (
        ak.str.is_lower(string).layout.form
        == ak.str.is_lower(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_lower(bytestring).layout.form
        == ak.str.is_lower(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_numeric():
    assert ak.str.is_numeric(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_numeric(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert (
        ak.str.is_numeric(string).layout.form
        == ak.str.is_numeric(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_numeric(bytestring).layout.form
        == ak.str.is_numeric(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_printable():
    assert ak.str.is_printable(string).tolist() == [
        [True, True],
        [],
        [True, True, True],
    ]
    assert ak.str.is_printable(bytestring).tolist() == [
        [False, True],
        [],
        [False, False, True],
    ]
    assert (
        ak.str.is_printable(string).layout.form
        == ak.str.is_printable(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_printable(bytestring).layout.form
        == ak.str.is_printable(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_space():
    assert ak.str.is_space(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_space(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert (
        ak.str.is_space(string).layout.form
        == ak.str.is_space(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_space(bytestring).layout.form
        == ak.str.is_space(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_upper():
    assert ak.str.is_upper(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_upper(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert (
        ak.str.is_upper(string).layout.form
        == ak.str.is_upper(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_upper(bytestring).layout.form
        == ak.str.is_upper(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_title():
    assert ak.str.is_title(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_title(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert (
        ak.str.is_title(string).layout.form
        == ak.str.is_title(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_title(bytestring).layout.form
        == ak.str.is_title(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_is_ascii():
    assert ak.str.is_ascii(string).tolist() == [
        [False, True],
        [],
        [False, False, True],
    ]
    assert ak.str.is_ascii(bytestring).tolist() == [
        [False, True],
        [],
        [False, False, True],
    ]
    assert (
        ak.str.is_ascii(string).layout.form
        == ak.str.is_ascii(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.is_ascii(bytestring).layout.form
        == ak.str.is_ascii(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_capitalize():
    assert ak.str.capitalize(string).tolist() == [
        ["Αβγ", ""],
        [],
        ["→δε←", "Ζz zζ", "Abc"],  # noqa: RUF001, RUF003 (we care about Ζ vs Z)
    ]
    assert ak.str.capitalize(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"Abc"],
    ]
    assert (
        ak.str.capitalize(string).layout.form
        == ak.str.capitalize(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.capitalize(bytestring).layout.form
        == ak.str.capitalize(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_length():
    assert ak.str.length(string).tolist() == [
        [3, 0],
        [],
        [4, 5, 3],
    ]
    assert ak.str.length(bytestring).tolist() == [
        [6, 0],
        [],
        [10, 7, 3],
    ]
    assert (
        ak.str.length(string).layout.form
        == ak.str.length(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.length(bytestring).layout.form
        == ak.str.length(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_lower():
    assert ak.str.lower(string).tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.lower(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]
    assert (
        ak.str.lower(string).layout.form
        == ak.str.lower(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.lower(bytestring).layout.form
        == ak.str.lower(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_swapcase():
    assert ak.str.swapcase(string).tolist() == [
        ["ΑΒΓ", ""],
        [],
        ["→ΔΕ←", "ΖZ ZΖ", "ABC"],  # noqa: RUF001, RUF003 (we care about Ζ vs Z)
    ]
    assert ak.str.swapcase(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζZ Zζ".encode(), b"ABC"],
    ]
    assert (
        ak.str.swapcase(string).layout.form
        == ak.str.swapcase(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.swapcase(bytestring).layout.form
        == ak.str.swapcase(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_title():
    assert ak.str.title(string).tolist() == [
        ["Αβγ", ""],
        [],
        ["→Δε←", "Ζz Zζ", "Abc"],  # noqa: RUF001, RUF003 (we care about Ζ vs Z)
    ]
    assert ak.str.title(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζZ Zζ".encode(), b"Abc"],
    ]
    assert (
        ak.str.title(string).layout.form
        == ak.str.title(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.title(bytestring).layout.form
        == ak.str.title(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_upper():
    assert ak.str.upper(string).tolist() == [
        ["ΑΒΓ", ""],
        [],
        ["→ΔΕ←", "ΖZ ZΖ", "ABC"],  # noqa: RUF001, RUF003 (we care about Ζ vs Z)
    ]
    assert ak.str.upper(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζZ Zζ".encode(), b"ABC"],
    ]
    assert (
        ak.str.upper(string).layout.form
        == ak.str.upper(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.upper(bytestring).layout.form
        == ak.str.upper(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_repeat():
    assert ak.str.repeat(string, 3).tolist() == [
        ["αβγαβγαβγ", ""],
        [],
        ["→δε←→δε←→δε←", "ζz zζζz zζζz zζ", "abcabcabc"],
    ]
    assert ak.str.repeat(bytestring, 3).tolist() == [
        ["αβγαβγαβγ".encode(), b""],
        [],
        ["→δε←→δε←→δε←".encode(), "ζz zζζz zζζz zζ".encode(), b"abcabcabc"],
    ]
    assert (
        ak.str.repeat(string, 3).layout.form
        == ak.str.repeat(ak.to_backend(string, "typetracer"), 3).layout.form
    )
    assert (
        ak.str.repeat(bytestring, 3).layout.form
        == ak.str.repeat(ak.to_backend(bytestring, "typetracer"), 3).layout.form
    )

    assert ak.str.repeat(string, [[3, 3], [], [2, 0, 1]]).tolist() == [
        ["αβγαβγαβγ", ""],
        [],
        ["→δε←→δε←", "", "abc"],
    ]
    assert ak.str.repeat(bytestring, [[3, 3], [], [2, 0, 1]]).tolist() == [
        ["αβγαβγαβγ".encode(), b""],
        [],
        ["→δε←→δε←".encode(), b"", b"abc"],
    ]
    assert (
        ak.str.repeat(string, [[3, 3], [], [2, 0, 1]]).layout.form
        == ak.str.repeat(
            ak.to_backend(string, "typetracer"), [[3, 3], [], [2, 0, 1]]
        ).layout.form
    )
    assert (
        ak.str.repeat(bytestring, [[3, 3], [], [2, 0, 1]]).layout.form
        == ak.str.repeat(
            ak.to_backend(bytestring, "typetracer"), [[3, 3], [], [2, 0, 1]]
        ).layout.form
    )


def test_replace_slice():
    assert ak.str.replace_slice(string[:, :1], 1, 2, "qj").tolist() == [
        ["αqjγ"],  # noqa: RUF001
        [],
        ["→qjε←"],
    ]
    assert ak.str.replace_slice(bytestring[:, :1], 1, 2, b"qj").tolist() == [
        [b"\xceqj\xce\xb2\xce\xb3"],
        [],
        [b"\xe2qj\x92\xce\xb4\xce\xb5\xe2\x86\x90"],
    ]
    assert (
        ak.str.replace_slice(string[:, :1], 1, 2, "qj").layout.form
        == ak.str.replace_slice(
            ak.to_backend(string, "typetracer")[:, :1], 1, 2, "qj"
        ).layout.form
    )
    assert (
        ak.str.replace_slice(bytestring[:, :1], 1, 2, b"qj").layout.form
        == ak.str.replace_slice(
            ak.to_backend(bytestring, "typetracer")[:, :1], 1, 2, b"qj"
        ).layout.form
    )


def test_reverse():
    assert ak.str.reverse(string).tolist() == [
        ["αβγ"[::-1], ""],
        [],
        ["→δε←"[::-1], "ζz zζ"[::-1], "abc"[::-1]],
    ]
    assert ak.str.reverse(bytestring).tolist() == [
        ["αβγ".encode()[::-1], b""],
        [],
        ["→δε←".encode()[::-1], "ζz zζ".encode()[::-1], b"abc"[::-1]],
    ]
    assert (
        ak.str.reverse(string).layout.form
        == ak.str.reverse(ak.to_backend(string, "typetracer")).layout.form
    )
    assert (
        ak.str.reverse(bytestring).layout.form
        == ak.str.reverse(ak.to_backend(bytestring, "typetracer")).layout.form
    )


def test_replace_substring():
    assert ak.str.replace_substring(string, "βγ", "HELLO").tolist() == [
        ["αHELLO", ""],  # noqa: RUF001
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.replace_substring(bytestring, "βγ".encode(), b"HELLO").tolist() == [
        ["αHELLO".encode(), b""],  # noqa: RUF001
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]
    assert (
        ak.str.replace_substring(string, "βγ", "HELLO").layout.form
        == ak.str.replace_substring(
            ak.to_backend(string, "typetracer"), "βγ", "HELLO"
        ).layout.form
    )
    assert (
        ak.str.replace_substring(bytestring, "βγ".encode(), b"HELLO").layout.form
        == ak.str.replace_substring(
            ak.to_backend(bytestring, "typetracer"), "βγ".encode(), b"HELLO"
        ).layout.form
    )

    assert ak.str.replace_substring(
        string, "βγ", "HELLO", max_replacements=0
    ).tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.replace_substring(
        bytestring, "βγ".encode(), b"HELLO", max_replacements=0
    ).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]
    assert (
        ak.str.replace_substring(string, "βγ", "HELLO", max_replacements=0).layout.form
        == ak.str.replace_substring(
            ak.to_backend(string, "typetracer"), "βγ", "HELLO", max_replacements=0
        ).layout.form
    )
    assert (
        ak.str.replace_substring(
            bytestring, "βγ".encode(), b"HELLO", max_replacements=0
        ).layout.form
        == ak.str.replace_substring(
            ak.to_backend(bytestring, "typetracer"),
            "βγ".encode(),
            b"HELLO",
            max_replacements=0,
        ).layout.form
    )


def test_replace_substring_regex():
    assert ak.str.replace_substring_regex(string, "βγ", "HELLO").tolist() == [
        ["αHELLO", ""],  # noqa: RUF001
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.replace_substring_regex(
        bytestring, "βγ".encode(), b"HELLO"
    ).tolist() == [
        ["αHELLO".encode(), b""],  # noqa: RUF001
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]
    assert (
        ak.str.replace_substring(string, "βγ", "HELLO").layout.form
        == ak.str.replace_substring(
            ak.to_backend(string, "typetracer"), "βγ", "HELLO"
        ).layout.form
    )
    assert (
        ak.str.replace_substring(bytestring, "βγ".encode(), b"HELLO").layout.form
        == ak.str.replace_substring(
            ak.to_backend(bytestring, "typetracer"), "βγ".encode(), b"HELLO"
        ).layout.form
    )

    assert ak.str.replace_substring_regex(
        string, "βγ", "HELLO", max_replacements=0
    ).tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.replace_substring_regex(
        bytestring, "βγ".encode(), b"HELLO", max_replacements=0
    ).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]
    assert (
        ak.str.replace_substring(string, "βγ", "HELLO", max_replacements=0).layout.form
        == ak.str.replace_substring(
            ak.to_backend(string, "typetracer"), "βγ", "HELLO", max_replacements=0
        ).layout.form
    )
    assert (
        ak.str.replace_substring(
            bytestring, "βγ".encode(), b"HELLO", max_replacements=0
        ).layout.form
        == ak.str.replace_substring(
            ak.to_backend(bytestring, "typetracer"),
            "βγ".encode(),
            b"HELLO",
            max_replacements=0,
        ).layout.form
    )

    # Check regex
    assert ak.str.replace_substring_regex(
        [["aaaa1bb1c2ddddd"], ["fff"], []], r"[a-zA-Z]+", "FOO"
    ).tolist() == [["FOO1FOO1FOO2FOO"], ["FOO"], []]
    assert ak.str.replace_substring_regex(
        [[b"aaaa1bb1c2ddddd"], [b"fff"], []], rb"[a-zA-Z]+", b"FOO"
    ).tolist() == [[b"FOO1FOO1FOO2FOO"], [b"FOO"], []]
    assert (
        ak.str.replace_substring_regex(
            [["aaaa1bb1c2ddddd"], ["fff"], []], r"[a-zA-Z]+", "FOO"
        ).layout.form
        == ak.str.replace_substring_regex(
            ak.to_backend([["aaaa1bb1c2ddddd"], ["fff"], []], "typetracer"),
            r"[a-zA-Z]+",
            "FOO",
        ).layout.form
    )
    assert (
        ak.str.replace_substring_regex(
            [[b"aaaa1bb1c2ddddd"], [b"fff"], []], rb"[a-zA-Z]+", b"FOO"
        ).layout.form
        == ak.str.replace_substring_regex(
            ak.to_backend([[b"aaaa1bb1c2ddddd"], [b"fff"], []], "typetracer"),
            rb"[a-zA-Z]+",
            b"FOO",
        ).layout.form
    )


def test_center():
    assert ak.str.center(string, 15, " ").tolist() == [
        ["      αβγ      ", "               "],
        [],
        ["     →δε←      ", "     ζz zζ     ", "      abc      "],
    ]
    assert ak.str.center(bytestring, 15, b" ").tolist() == [
        [b"    \xce\xb1\xce\xb2\xce\xb3     ", b"               "],
        [],
        [
            b"  \xe2\x86\x92\xce\xb4\xce\xb5\xe2\x86\x90   ",
            b"    \xce\xb6z z\xce\xb6    ",
            b"      abc      ",
        ],
    ]
    assert (
        ak.str.center(string, 15, " ").layout.form
        == ak.str.center(ak.to_backend(string, "typetracer"), 15, " ").layout.form
    )
    assert (
        ak.str.center(bytestring, 15, b" ").layout.form
        == ak.str.center(ak.to_backend(bytestring, "typetracer"), 15, b" ").layout.form
    )


def test_lpad():
    assert ak.str.lpad(string, 15, " ").tolist() == [
        ["            αβγ", "               "],
        [],
        ["           →δε←", "          ζz zζ", "            abc"],
    ]
    assert ak.str.lpad(bytestring, 15, b" ").tolist() == [
        [b"         \xce\xb1\xce\xb2\xce\xb3", b"               "],
        [],
        [
            b"     \xe2\x86\x92\xce\xb4\xce\xb5\xe2\x86\x90",
            b"        \xce\xb6z z\xce\xb6",
            b"            abc",
        ],
    ]
    assert (
        ak.str.lpad(string, 15, " ").layout.form
        == ak.str.lpad(ak.to_backend(string, "typetracer"), 15, " ").layout.form
    )
    assert (
        ak.str.lpad(bytestring, 15, b" ").layout.form
        == ak.str.lpad(ak.to_backend(bytestring, "typetracer"), 15, b" ").layout.form
    )


def test_rpad():
    assert ak.str.rpad(string, 15, " ").tolist() == [
        ["αβγ            ", "               "],
        [],
        ["→δε←           ", "ζz zζ          ", "abc            "],
    ]
    assert ak.str.rpad(bytestring, 15, b" ").tolist() == [
        [b"\xce\xb1\xce\xb2\xce\xb3         ", b"               "],
        [],
        [
            b"\xe2\x86\x92\xce\xb4\xce\xb5\xe2\x86\x90     ",
            b"\xce\xb6z z\xce\xb6        ",
            b"abc            ",
        ],
    ]
    assert (
        ak.str.rpad(string, 15, " ").layout.form
        == ak.str.rpad(ak.to_backend(string, "typetracer"), 15, " ").layout.form
    )
    assert (
        ak.str.rpad(bytestring, 15, b" ").layout.form
        == ak.str.rpad(ak.to_backend(bytestring, "typetracer"), 15, b" ").layout.form
    )


def test_ltrim():
    assert ak.str.ltrim(string_padded, " ").tolist() == [
        ["αβγ      ", ""],
        [],
        ["→δε←      ", "ζz zζ     ", "abc      "],
    ]
    assert ak.str.ltrim(bytestring_padded, b" ").tolist() == [
        ["αβγ     ".encode(), b""],
        [],
        ["→δε←   ".encode(), "ζz zζ    ".encode(), b"abc      "],
    ]
    assert (
        ak.str.ltrim(string_padded, " ").layout.form
        == ak.str.ltrim(ak.to_backend(string_padded, "typetracer"), " ").layout.form
    )
    assert (
        ak.str.ltrim(bytestring_padded, b" ").layout.form
        == ak.str.ltrim(
            ak.to_backend(bytestring_padded, "typetracer"), b" "
        ).layout.form
    )


def test_ltrim_whitespace():
    assert ak.str.ltrim_whitespace(string_padded).tolist() == [
        ["αβγ      ", ""],
        [],
        ["→δε←      ", "ζz zζ     ", "abc      "],
    ]
    assert ak.str.ltrim_whitespace(bytestring_padded).tolist() == [
        ["αβγ     ".encode(), b""],
        [],
        ["→δε←   ".encode(), "ζz zζ    ".encode(), b"abc      "],
    ]
    assert (
        ak.str.ltrim_whitespace(string_padded).layout.form
        == ak.str.ltrim_whitespace(
            ak.to_backend(string_padded, "typetracer")
        ).layout.form
    )
    assert (
        ak.str.ltrim_whitespace(bytestring_padded).layout.form
        == ak.str.ltrim_whitespace(
            ak.to_backend(bytestring_padded, "typetracer")
        ).layout.form
    )


def test_rtrim():
    assert ak.str.rtrim(string_padded, " ").tolist() == [
        ["      αβγ", ""],
        [],
        ["     →δε←", "     ζz zζ", "      abc"],
    ]
    assert ak.str.rtrim(bytestring_padded, b" ").tolist() == [
        ["    αβγ".encode(), b""],
        [],
        ["  →δε←".encode(), "    ζz zζ".encode(), b"      abc"],
    ]
    assert (
        ak.str.rtrim(string_padded, " ").layout.form
        == ak.str.rtrim(ak.to_backend(string_padded, "typetracer"), " ").layout.form
    )
    assert (
        ak.str.rtrim(bytestring_padded, b" ").layout.form
        == ak.str.rtrim(
            ak.to_backend(bytestring_padded, "typetracer"), b" "
        ).layout.form
    )


def test_rtrim_whitespace():
    assert ak.str.rtrim_whitespace(string_padded).tolist() == [
        ["      αβγ", ""],
        [],
        ["     →δε←", "     ζz zζ", "      abc"],
    ]
    assert ak.str.rtrim_whitespace(bytestring_padded).tolist() == [
        ["    αβγ".encode(), b""],
        [],
        ["  →δε←".encode(), "    ζz zζ".encode(), b"      abc"],
    ]
    assert (
        ak.str.rtrim_whitespace(string_padded).layout.form
        == ak.str.rtrim_whitespace(
            ak.to_backend(string_padded, "typetracer")
        ).layout.form
    )
    assert (
        ak.str.rtrim_whitespace(bytestring_padded).layout.form
        == ak.str.rtrim_whitespace(
            ak.to_backend(bytestring_padded, "typetracer")
        ).layout.form
    )


def test_trim():
    assert ak.str.trim(string_padded, " ").tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.trim(bytestring_padded, b" ").tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]
    assert (
        ak.str.trim(string_padded, " ").layout.form
        == ak.str.trim(ak.to_backend(string_padded, "typetracer"), " ").layout.form
    )
    assert (
        ak.str.trim(bytestring_padded, b" ").layout.form
        == ak.str.trim(ak.to_backend(bytestring_padded, "typetracer"), b" ").layout.form
    )


def test_trim_whitespace():
    assert ak.str.trim_whitespace(string_padded).tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.trim_whitespace(bytestring_padded).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]
    assert (
        ak.str.trim_whitespace(string_padded).layout.form
        == ak.str.trim_whitespace(
            ak.to_backend(string_padded, "typetracer")
        ).layout.form
    )

    assert (
        ak.str.trim_whitespace(bytestring_padded).layout.form
        == ak.str.trim_whitespace(
            ak.to_backend(bytestring_padded, "typetracer")
        ).layout.form
    )


def test_slice():
    assert ak.str.slice(string, 1, 3).tolist() == [
        ["αβγ"[1:3], ""[1:3]],
        [],
        ["→δε←"[1:3], "ζz zζ"[1:3], "abc"[1:3]],
    ]
    assert ak.str.slice(bytestring, 1, 3).tolist() == [
        ["αβγ".encode()[1:3], b""[1:3]],
        [],
        ["→δε←".encode()[1:3], "ζz zζ".encode()[1:3], b"abc"[1:3]],
    ]
    assert (
        ak.str.slice(string, 1, 3).layout.form
        == ak.str.slice(ak.to_backend(string, "typetracer"), 1, 3).layout.form
    )
    assert (
        ak.str.slice(bytestring, 1, 3).layout.form
        == ak.str.slice(ak.to_backend(bytestring, "typetracer"), 1, 3).layout.form
    )

    # ArrowInvalid: Negative buffer resize: -40 (looks like an Arrow bug)
    # assert ak.str.slice(string, 1).tolist() == [
    #     ["αβγ"[1:], ""[1:]],
    #     [],
    #     ["→δε←"[1:], "ζz zζ"[1:], "abc"[1:]],
    # ]
    # assert (
    #     ak.str.slice(string, 1).layout.form
    #     == ak.str.slice(ak.to_backend(string, "typetracer"), 1).layout.form
    # )
    assert ak.str.slice(bytestring, 1).tolist() == [
        ["αβγ".encode()[1:], b""[1:]],
        [],
        ["→δε←".encode()[1:], "ζz zζ".encode()[1:], b"abc"[1:]],
    ]
    assert (
        ak.str.slice(bytestring, 1).layout.form
        == ak.str.slice(ak.to_backend(bytestring, "typetracer"), 1).layout.form
    )


def test_split_whitespace():
    assert ak.str.split_whitespace(string_padded, max_splits=1).tolist() == [
        [["", "αβγ      "], ["", " "]],
        [],
        [["", "→δε←      "], ["", "ζz zζ     "], ["", "abc      "]],
    ]
    assert (
        ak.str.split_whitespace(string_padded, max_splits=1).layout.form
        == ak.str.split_whitespace(
            ak.to_backend(string_padded, "typetracer"), max_splits=1
        ).layout.form
    )

    assert ak.str.split_whitespace(
        string_padded, max_splits=1, reverse=True
    ).tolist() == [
        [["      αβγ", ""], [" ", ""]],
        [],
        [["     →δε←", ""], ["     ζz zζ", ""], ["      abc", ""]],
    ]
    assert (
        ak.str.split_whitespace(string_padded, max_splits=1, reverse=True).layout.form
        == ak.str.split_whitespace(
            ak.to_backend(string_padded, "typetracer"), max_splits=1, reverse=True
        ).layout.form
    )

    assert ak.str.split_whitespace(string_padded, max_splits=None).tolist() == [
        [["", "αβγ", "", ""], ["", "", ""]],
        [],
        [["", "→δε←", "", ""], ["", "ζz", "zζ", "", ""], ["", "abc", "", ""]],
    ]
    assert (
        ak.str.split_whitespace(string_padded, max_splits=None).layout.form
        == ak.str.split_whitespace(
            ak.to_backend(string_padded, "typetracer"), max_splits=None
        ).layout.form
    )

    # Bytestrings
    assert ak.str.split_whitespace(bytestring_padded, max_splits=1).tolist() == [
        [[b"", "αβγ     ".encode()], [b"", b""]],
        [],
        [
            [b"", "→δε←   ".encode()],
            [b"", "ζz zζ    ".encode()],
            [b"", b"abc      "],
        ],
    ]
    assert (
        ak.str.split_whitespace(bytestring_padded, max_splits=1).layout.form
        == ak.str.split_whitespace(
            ak.to_backend(bytestring_padded, "typetracer"), max_splits=1
        ).layout.form
    )

    assert ak.str.split_whitespace(
        bytestring_padded, max_splits=1, reverse=True
    ).tolist() == [
        [["    αβγ".encode(), b""], [b"", b""]],
        [],
        [
            ["  →δε←".encode(), b""],
            ["    ζz zζ".encode(), b""],
            [b"      abc", b""],
        ],
    ]
    assert (
        ak.str.split_whitespace(
            bytestring_padded, max_splits=1, reverse=True
        ).layout.form
        == ak.str.split_whitespace(
            ak.to_backend(bytestring_padded, "typetracer"), max_splits=1, reverse=True
        ).layout.form
    )

    assert ak.str.split_whitespace(bytestring_padded, max_splits=None).tolist() == [
        [[b"", "αβγ".encode(), b""], [b"", b""]],
        [],
        [
            [b"", "→δε←".encode(), b""],
            [b"", "ζz".encode(), "zζ".encode(), b""],
            [b"", b"abc", b""],
        ],
    ]
    assert (
        ak.str.split_whitespace(bytestring_padded, max_splits=None).layout.form
        == ak.str.split_whitespace(
            ak.to_backend(bytestring_padded, "typetracer"), max_splits=None
        ).layout.form
    )


def test_split_pattern():
    assert ak.str.split_pattern(string_repeats, "123", max_splits=1).tolist() == [
        [["foo", "bar123baz"], ["foo"], ["bar"]],
        [["", "foo"], ["456bar"], ["foo", "456bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern(string_repeats, "123", max_splits=1).layout.form
        == ak.str.split_pattern(
            ak.to_backend(string_repeats, "typetracer"), "123", max_splits=1
        ).layout.form
    )

    assert ak.str.split_pattern(
        string_repeats, "123", max_splits=1, reverse=True
    ).tolist() == [
        [["foo123bar", "baz"], ["foo"], ["bar"]],
        [["", "foo"], ["456bar"], ["foo", "456bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern(
            string_repeats, "123", max_splits=1, reverse=True
        ).layout.form
        == ak.str.split_pattern(
            ak.to_backend(string_repeats, "typetracer"),
            "123",
            max_splits=1,
            reverse=True,
        ).layout.form
    )
    assert ak.str.split_pattern(string_repeats, "123", max_splits=None).tolist() == [
        [["foo", "bar", "baz"], ["foo"], ["bar"]],
        [["", "foo"], ["456bar"], ["foo", "456bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern(string_repeats, "123", max_splits=None).layout.form
        == ak.str.split_pattern(
            ak.to_backend(string_repeats, "typetracer"), "123", max_splits=None
        ).layout.form
    )

    # Bytestrings
    assert ak.str.split_pattern(bytestring_repeats, b"123", max_splits=1).tolist() == [
        [[b"foo", b"bar123baz"], [b"foo"], [b"bar"]],
        [[b"", b"foo"], [b"456bar"], [b"foo", b"456bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern(bytestring_repeats, b"123", max_splits=1).layout.form
        == ak.str.split_pattern(
            ak.to_backend(bytestring_repeats, "typetracer"), b"123", max_splits=1
        ).layout.form
    )

    assert ak.str.split_pattern(
        bytestring_repeats, b"123", max_splits=1, reverse=True
    ).tolist() == [
        [[b"foo123bar", b"baz"], [b"foo"], [b"bar"]],
        [[b"", b"foo"], [b"456bar"], [b"foo", b"456bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern(
            bytestring_repeats, b"123", max_splits=1, reverse=True
        ).layout.form
        == ak.str.split_pattern(
            ak.to_backend(bytestring_repeats, "typetracer"),
            b"123",
            max_splits=1,
            reverse=True,
        ).layout.form
    )

    assert ak.str.split_pattern(
        bytestring_repeats, b"123", max_splits=None
    ).tolist() == [
        [[b"foo", b"bar", b"baz"], [b"foo"], [b"bar"]],
        [[b"", b"foo"], [b"456bar"], [b"foo", b"456bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern(bytestring_repeats, b"123", max_splits=None).layout.form
        == ak.str.split_pattern(
            ak.to_backend(bytestring_repeats, "typetracer"), b"123", max_splits=None
        ).layout.form
    )


def test_split_pattern_regex():
    assert ak.str.split_pattern_regex(
        string_repeats, r"\d{3}", max_splits=1
    ).tolist() == [
        [["foo", "bar123baz"], ["foo"], ["bar"]],
        [["", "foo"], ["", "bar"], ["foo", "456bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern_regex(string_repeats, r"\d{3}", max_splits=1).layout.form
        == ak.str.split_pattern_regex(
            ak.to_backend(string_repeats, "typetracer"), r"\d{3}", max_splits=1
        ).layout.form
    )

    with pytest.raises(ValueError, match=r"split in reverse with regex"):
        assert ak.str.split_pattern_regex(
            string_repeats, r"\d{3}", max_splits=1, reverse=True
        ).tolist() == [
            [["foo123bar", "baz"], ["foo"], ["bar"]],
            [["", "foo"], ["", "bar"], ["foo", "456bar"]],
            [],
        ]
    with pytest.raises(ValueError, match=r"split in reverse with regex"):
        ak.str.split_pattern_regex(
            ak.to_backend(string_repeats, "typetracer"),
            r"\d{3}",
            max_splits=1,
            reverse=True,
        )

    assert ak.str.split_pattern_regex(
        string_repeats, r"\d{3}", max_splits=None
    ).tolist() == [
        [["foo", "bar", "baz"], ["foo"], ["bar"]],
        [["", "foo"], ["", "bar"], ["foo", "", "bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern_regex(
            string_repeats, r"\d{3}", max_splits=None
        ).layout.form
        == ak.str.split_pattern_regex(
            ak.to_backend(string_repeats, "typetracer"), r"\d{3}", max_splits=None
        ).layout.form
    )

    # Bytestrings
    assert ak.str.split_pattern_regex(
        bytestring_repeats, rb"\d{3}", max_splits=1
    ).tolist() == [
        [[b"foo", b"bar123baz"], [b"foo"], [b"bar"]],
        [[b"", b"foo"], [b"", b"bar"], [b"foo", b"456bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern_regex(
            bytestring_repeats, r"\d{3}", max_splits=1
        ).layout.form
        == ak.str.split_pattern_regex(
            ak.to_backend(bytestring_repeats, "typetracer"), r"\d{3}", max_splits=1
        ).layout.form
    )

    with pytest.raises(ValueError, match=r"split in reverse with regex"):
        assert ak.str.split_pattern_regex(
            bytestring_repeats, rb"\d{3}", max_splits=1, reverse=True
        ).tolist() == [
            [[b"foo123bar", b"baz"], [b"foo"], [b"bar"]],
            [[b"", b"foo"], [b"", b"bar"], [b"foo", b"456bar"]],
            [],
        ]
    with pytest.raises(ValueError, match=r"split in reverse with regex"):
        ak.str.split_pattern_regex(
            ak.to_backend(bytestring_repeats, "typetracer"),
            r"\d{3}",
            max_splits=1,
            reverse=True,
        )

    assert ak.str.split_pattern_regex(
        bytestring_repeats, rb"\d{3}", max_splits=None
    ).tolist() == [
        [[b"foo", b"bar", b"baz"], [b"foo"], [b"bar"]],
        [[b"", b"foo"], [b"", b"bar"], [b"foo", b"", b"bar"]],
        [],
    ]
    assert (
        ak.str.split_pattern_regex(
            bytestring_repeats, r"\d{3}", max_splits=None
        ).layout.form
        == ak.str.split_pattern_regex(
            ak.to_backend(bytestring_repeats, "typetracer"), r"\d{3}", max_splits=None
        ).layout.form
    )


def test_extract_regex():
    assert ak.str.extract_regex(
        ak.Array([["one1", "two2", "three3"], [], ["four4", "five5"]]),
        "(?P<vowel>[aeiou])(?P<number>[0-9]+)",
    ).tolist() == [
        [
            {"vowel": "e", "number": "1"},
            {"vowel": "o", "number": "2"},
            {"vowel": "e", "number": "3"},
        ],
        [],
        [None, {"vowel": "e", "number": "5"}],
    ]
    assert (
        ak.str.extract_regex(
            ak.Array([["one1", "two2", "three3"], [], ["four4", "five5"]]),
            "(?P<vowel>[aeiou])(?P<number>[0-9]+)",
        ).layout.form
        == ak.str.extract_regex(
            ak.Array(
                [["one1", "two2", "three3"], [], ["four4", "five5"]],
                backend="typetracer",
            ),
            "(?P<vowel>[aeiou])(?P<number>[0-9]+)",
        ).layout.form
    )

    assert ak.str.extract_regex(
        ak.Array([[b"one1", b"two2", b"three3"], [], [b"four4", b"five5"]]),
        b"(?P<vowel>[aeiou])(?P<number>[0-9]+)",
    ).tolist() == [
        [
            {"vowel": b"e", "number": b"1"},
            {"vowel": b"o", "number": b"2"},
            {"vowel": b"e", "number": b"3"},
        ],
        [],
        [None, {"vowel": b"e", "number": b"5"}],
    ]
    assert (
        ak.str.extract_regex(
            ak.Array([[b"one1", b"two2", b"three3"], [], [b"four4", b"five5"]]),
            b"(?P<vowel>[aeiou])(?P<number>[0-9]+)",
        ).layout.form
        == ak.str.extract_regex(
            ak.Array(
                [[b"one1", b"two2", b"three3"], [], [b"four4", b"five5"]],
                backend="typetracer",
            ),
            b"(?P<vowel>[aeiou])(?P<number>[0-9]+)",
        ).layout.form
    )


def test_join():
    array1 = ak.Array(
        [
            ["this", "that"],
            [],
            ["foo", "bar", "baz"],
        ]
    )
    assert ak.str.join(array1, "-").tolist() == ["this-that", "", "foo-bar-baz"]
    assert (
        ak.str.join(array1, "-").layout.form
        == ak.str.join(ak.to_backend(array1, "typetracer"), "-").layout.form
    )

    separator = ak.Array(["→", "↑", "←"])
    assert ak.str.join(array1, separator).tolist() == ["this→that", "", "foo←bar←baz"]
    assert (
        ak.str.join(array1, separator).layout.form
        == ak.str.join(ak.to_backend(array1, "typetracer"), separator).layout.form
    )

    array2 = ak.Array(
        [
            [b"this", b"that"],
            [],
            [b"foo", b"bar", b"baz"],
        ]
    )
    assert ak.str.join(array2, b"-").tolist() == [b"this-that", b"", b"foo-bar-baz"]
    assert (
        ak.str.join(array2, b"-").layout.form
        == ak.str.join(ak.to_backend(array2, "typetracer"), b"-").layout.form
    )

    separator2 = ak.Array(["→".encode(), "↑".encode(), "←".encode()])
    assert ak.str.join(array2, separator2).tolist() == [
        "this→that".encode(),
        b"",
        "foo←bar←baz".encode(),
    ]
    assert (
        ak.str.join(array2, separator2).layout.form
        == ak.str.join(ak.to_backend(array2, "typetracer"), separator2).layout.form
    )


def test_join_element_wise():
    first1 = ak.Array([["one", "two", "three"], [], ["four", "five"]])
    second1 = ak.Array([["111", "222", "333"], [], ["444", "555"]])
    separator1 = ak.Array(["→", "↑", "←"])

    assert ak.str.join_element_wise(first1, second1, separator1).tolist() == [
        ["one→111", "two→222", "three→333"],
        [],
        ["four←444", "five←555"],
    ]
    assert (
        ak.str.join_element_wise(first1, second1, separator1).layout.form
        == ak.str.join_element_wise(
            ak.to_backend(first1, "typetracer"),
            ak.to_backend(second1, "typetracer"),
            ak.to_backend(separator1, "typetracer"),
        ).layout.form
    )

    first2 = ak.Array([[b"one", b"two", b"three"], [], [b"four", b"five"]])
    second2 = ak.Array([[b"111", b"222", b"333"], [], [b"444", b"555"]])
    separator2 = ak.Array(["→".encode(), "↑".encode(), "←".encode()])

    assert ak.str.join_element_wise(first2, second2, separator2).tolist() == [
        ["one→111".encode(), "two→222".encode(), "three→333".encode()],
        [],
        ["four←444".encode(), "five←555".encode()],
    ]
    assert (
        ak.str.join_element_wise(first2, second2, separator2).layout.form
        == ak.str.join_element_wise(
            ak.to_backend(first2, "typetracer"),
            ak.to_backend(second2, "typetracer"),
            ak.to_backend(separator2, "typetracer"),
        ).layout.form
    )


def test_count_substring():
    assert ak.str.count_substring(string_repeats, "BA").tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [],
    ]
    assert (
        ak.str.count_substring(string_repeats, "BA").layout.form
        == ak.str.count_substring(
            ak.to_backend(string_repeats, "typetracer"), "BA"
        ).layout.form
    )

    assert ak.str.count_substring(string_repeats, "BA", ignore_case=True).tolist() == [
        [2, 0, 1],
        [0, 1, 1],
        [],
    ]
    assert (
        ak.str.count_substring(string_repeats, "BA", ignore_case=True).layout.form
        == ak.str.count_substring(
            ak.to_backend(string_repeats, "typetracer"), "BA", ignore_case=True
        ).layout.form
    )

    # Bytestrings
    assert ak.str.count_substring(bytestring_repeats, b"BA").tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [],
    ]
    assert (
        ak.str.count_substring(bytestring_repeats, b"BA").layout.form
        == ak.str.count_substring(
            ak.to_backend(bytestring_repeats, "typetracer"), b"BA"
        ).layout.form
    )

    assert ak.str.count_substring(
        bytestring_repeats, b"BA", ignore_case=True
    ).tolist() == [
        [2, 0, 1],
        [0, 1, 1],
        [],
    ]
    assert (
        ak.str.count_substring(bytestring_repeats, b"BA", ignore_case=True).layout.form
        == ak.str.count_substring(
            ak.to_backend(bytestring_repeats, "typetracer"), b"BA", ignore_case=True
        ).layout.form
    )


def test_count_substring_regex():
    assert ak.str.count_substring_regex(string_repeats, r"BA\d*").tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [],
    ]
    assert (
        ak.str.count_substring_regex(string_repeats, r"BA\d*").layout.form
        == ak.str.count_substring_regex(
            ak.to_backend(string_repeats, "typetracer"), r"BA\d*"
        ).layout.form
    )

    assert ak.str.count_substring_regex(
        string_repeats, r"BA\d*", ignore_case=True
    ).tolist() == [
        [2, 0, 1],
        [0, 1, 1],
        [],
    ]
    assert (
        ak.str.count_substring_regex(
            string_repeats, r"BA\d*", ignore_case=True
        ).layout.form
        == ak.str.count_substring_regex(
            ak.to_backend(string_repeats, "typetracer"), r"BA\d*", ignore_case=True
        ).layout.form
    )

    assert ak.str.count_substring_regex(string_repeats, r"\d{1,}").tolist() == [
        [2, 0, 0],
        [1, 1, 1],
        [],
    ]
    assert (
        ak.str.count_substring_regex(string_repeats, r"\d{1,}").layout.form
        == ak.str.count_substring_regex(
            ak.to_backend(string_repeats, "typetracer"), r"\d{1,}"
        ).layout.form
    )

    # Bytestrings
    assert ak.str.count_substring_regex(bytestring_repeats, rb"BA\d*").tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [],
    ]
    assert (
        ak.str.count_substring_regex(bytestring_repeats, rb"BA\d*").layout.form
        == ak.str.count_substring_regex(
            ak.to_backend(bytestring_repeats, "typetracer"), rb"BA\d*"
        ).layout.form
    )

    assert ak.str.count_substring_regex(
        bytestring_repeats, rb"BA\d*", ignore_case=True
    ).tolist() == [
        [2, 0, 1],
        [0, 1, 1],
        [],
    ]
    assert (
        ak.str.count_substring_regex(
            bytestring_repeats, rb"BA\d*", ignore_case=True
        ).layout.form
        == ak.str.count_substring_regex(
            ak.to_backend(bytestring_repeats, "typetracer"), rb"BA\d*", ignore_case=True
        ).layout.form
    )

    assert ak.str.count_substring_regex(bytestring_repeats, rb"\d{1,}").tolist() == [
        [2, 0, 0],
        [1, 1, 1],
        [],
    ]
    assert (
        ak.str.count_substring_regex(bytestring_repeats, rb"\d{1,}").layout.form
        == ak.str.count_substring_regex(
            ak.to_backend(bytestring_repeats, "typetracer"), rb"\d{1,}"
        ).layout.form
    )


def test_ends_with():
    assert ak.str.ends_with(string_repeats, "BAR").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.ends_with(string_repeats, "BAR").layout.form
        == ak.str.ends_with(
            ak.to_backend(string_repeats, "typetracer"), "BAR"
        ).layout.form
    )

    assert ak.str.ends_with(string_repeats, "BAR", ignore_case=True).tolist() == [
        [False, False, True],
        [False, True, True],
        [],
    ]
    assert (
        ak.str.ends_with(string_repeats, "BAR", ignore_case=True).layout.form
        == ak.str.ends_with(
            ak.to_backend(string_repeats, "typetracer"), "BAR", ignore_case=True
        ).layout.form
    )

    # Bytestrings
    assert ak.str.ends_with(bytestring_repeats, b"BAR").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.ends_with(bytestring_repeats, b"BAR").layout.form
        == ak.str.ends_with(
            ak.to_backend(bytestring_repeats, "typetracer"), b"BAR"
        ).layout.form
    )

    assert ak.str.ends_with(bytestring_repeats, b"BAR", ignore_case=True).tolist() == [
        [False, False, True],
        [False, True, True],
        [],
    ]
    assert (
        ak.str.ends_with(bytestring_repeats, b"BAR", ignore_case=True).layout.form
        == ak.str.ends_with(
            ak.to_backend(bytestring_repeats, "typetracer"), b"BAR", ignore_case=True
        ).layout.form
    )


def test_starts_with():
    assert ak.str.starts_with(string_repeats, "FOO").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.starts_with(string_repeats, "FOO").layout.form
        == ak.str.starts_with(
            ak.to_backend(string_repeats, "typetracer"), "FOO"
        ).layout.form
    )

    assert ak.str.starts_with(string_repeats, "FOO", ignore_case=True).tolist() == [
        [True, True, False],
        [False, False, True],
        [],
    ]
    assert (
        ak.str.starts_with(string_repeats, "FOO", ignore_case=True).layout.form
        == ak.str.starts_with(
            ak.to_backend(string_repeats, "typetracer"), "FOO", ignore_case=True
        ).layout.form
    )

    # Bytestrings
    assert ak.str.starts_with(bytestring_repeats, b"FOO").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.starts_with(bytestring_repeats, b"FOO").layout.form
        == ak.str.starts_with(
            ak.to_backend(bytestring_repeats, "typetracer"), b"FOO"
        ).layout.form
    )

    assert ak.str.starts_with(
        bytestring_repeats, b"FOO", ignore_case=True
    ).tolist() == [
        [True, True, False],
        [False, False, True],
        [],
    ]
    assert (
        ak.str.starts_with(bytestring_repeats, b"FOO", ignore_case=True).layout.form
        == ak.str.starts_with(
            ak.to_backend(bytestring_repeats, "typetracer"), b"FOO", ignore_case=True
        ).layout.form
    )


def test_find_substring():
    assert ak.str.find_substring(string_repeats, "FOO").tolist() == [
        [-1, -1, -1],
        [-1, -1, -1],
        [],
    ]
    assert (
        ak.str.find_substring(string_repeats, "FOO").layout.form
        == ak.str.find_substring(
            ak.to_backend(string_repeats, "typetracer"), "FOO"
        ).layout.form
    )

    assert ak.str.find_substring(string_repeats, "FOO", ignore_case=True).tolist() == [
        [0, 0, -1],
        [3, -1, 0],
        [],
    ]
    assert (
        ak.str.find_substring(string_repeats, "FOO", ignore_case=True).layout.form
        == ak.str.find_substring(
            ak.to_backend(string_repeats, "typetracer"), "FOO", ignore_case=True
        ).layout.form
    )

    # Bytestrings
    assert ak.str.find_substring(bytestring_repeats, b"FOO").tolist() == [
        [-1, -1, -1],
        [-1, -1, -1],
        [],
    ]
    assert (
        ak.str.find_substring(bytestring_repeats, b"FOO").layout.form
        == ak.str.find_substring(
            ak.to_backend(bytestring_repeats, "typetracer"), b"FOO"
        ).layout.form
    )

    assert ak.str.find_substring(
        bytestring_repeats, b"FOO", ignore_case=True
    ).tolist() == [
        [0, 0, -1],
        [3, -1, 0],
        [],
    ]
    assert (
        ak.str.find_substring(bytestring_repeats, b"FOO", ignore_case=True).layout.form
        == ak.str.find_substring(
            ak.to_backend(bytestring_repeats, "typetracer"), b"FOO", ignore_case=True
        ).layout.form
    )


def test_find_substring_regex():
    assert ak.str.find_substring_regex(string_repeats, r"FOO\d+").tolist() == [
        [-1, -1, -1],
        [-1, -1, -1],
        [],
    ]
    assert (
        ak.str.find_substring_regex(string_repeats, r"FOO\d+").layout.form
        == ak.str.find_substring_regex(
            ak.to_backend(string_repeats, "typetracer"), r"FOO\d+"
        ).layout.form
    )

    assert ak.str.find_substring_regex(
        string_repeats, r"FOO\d+", ignore_case=True
    ).tolist() == [
        [0, -1, -1],
        [-1, -1, 0],
        [],
    ]
    assert (
        ak.str.find_substring_regex(
            string_repeats, r"FOO\d+", ignore_case=True
        ).layout.form
        == ak.str.find_substring_regex(
            ak.to_backend(string_repeats, "typetracer"), r"FOO\d+", ignore_case=True
        ).layout.form
    )

    # Bytestrings
    assert ak.str.find_substring_regex(bytestring_repeats, rb"FOO\d+").tolist() == [
        [-1, -1, -1],
        [-1, -1, -1],
        [],
    ]
    assert (
        ak.str.find_substring_regex(bytestring_repeats, rb"FOO\d+").layout.form
        == ak.str.find_substring_regex(
            ak.to_backend(bytestring_repeats, "typetracer"), rb"FOO\d+"
        ).layout.form
    )

    assert ak.str.find_substring_regex(
        bytestring_repeats, rb"FOO\d+", ignore_case=True
    ).tolist() == [
        [0, -1, -1],
        [-1, -1, 0],
        [],
    ]
    assert (
        ak.str.find_substring_regex(
            bytestring_repeats, rb"FOO\d+", ignore_case=True
        ).layout.form
        == ak.str.find_substring_regex(
            ak.to_backend(bytestring_repeats, "typetracer"),
            rb"FOO\d+",
            ignore_case=True,
        ).layout.form
    )


def test_match_like():
    assert ak.str.match_like(string_repeats, "FOO%BA%").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.match_like(string_repeats, "FOO%BA%").layout.form
        == ak.str.match_like(
            ak.to_backend(string_repeats, "typetracer"), "FOO%BA%"
        ).layout.form
    )

    assert ak.str.match_like(string_repeats, "FOO%BA%", ignore_case=True).tolist() == [
        [True, False, False],
        [False, False, True],
        [],
    ]
    assert (
        ak.str.match_like(string_repeats, "FOO%BA%", ignore_case=True).layout.form
        == ak.str.match_like(
            ak.to_backend(string_repeats, "typetracer"), "FOO%BA%", ignore_case=True
        ).layout.form
    )

    # Bytestrings
    assert ak.str.match_like(bytestring_repeats, b"FOO%BA%").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.match_like(bytestring_repeats, b"FOO%BA%").layout.form
        == ak.str.match_like(
            ak.to_backend(bytestring_repeats, "typetracer"), b"FOO%BA%"
        ).layout.form
    )

    assert ak.str.match_like(
        bytestring_repeats, b"FOO%BA%", ignore_case=True
    ).tolist() == [
        [True, False, False],
        [False, False, True],
        [],
    ]
    assert (
        ak.str.match_like(bytestring_repeats, b"FOO%BA^", ignore_case=True).layout.form
        == ak.str.match_like(
            ak.to_backend(bytestring_repeats, "typetracer"),
            b"FOO%BA^",
            ignore_case=True,
        ).layout.form
    )


def test_match_substring():
    assert ak.str.match_substring(string_repeats, "FOO").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.match_substring(string_repeats, "FOO").layout.form
        == ak.str.match_substring(
            ak.to_backend(string_repeats, "typetracer"), "FOO"
        ).layout.form
    )

    assert ak.str.match_substring(string_repeats, "FOO", ignore_case=True).tolist() == [
        [True, True, False],
        [True, False, True],
        [],
    ]
    assert (
        ak.str.match_substring(string_repeats, "FOO", ignore_case=True).layout.form
        == ak.str.match_substring(
            ak.to_backend(string_repeats, "typetracer"), "FOO", ignore_case=True
        ).layout.form
    )

    # Bytestrings
    assert ak.str.match_substring(bytestring_repeats, b"FOO").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.match_substring(bytestring_repeats, b"FOO").layout.form
        == ak.str.match_substring(
            ak.to_backend(bytestring_repeats, "typetracer"), b"FOO"
        ).layout.form
    )

    assert ak.str.match_substring(
        bytestring_repeats, b"FOO", ignore_case=True
    ).tolist() == [
        [True, True, False],
        [True, False, True],
        [],
    ]
    assert (
        ak.str.match_substring(bytestring_repeats, b"FOO", ignore_case=True).layout.form
        == ak.str.match_substring(
            ak.to_backend(bytestring_repeats, "typetracer"), b"FOO", ignore_case=True
        ).layout.form
    )


def test_match_substring_regex():
    assert ak.str.match_substring_regex(string_repeats, r"FOO\d+").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.match_substring_regex(string_repeats, r"FOO\d+").layout.form
        == ak.str.match_substring_regex(
            ak.to_backend(string_repeats, "typetracer"), r"FOO\d+"
        ).layout.form
    )

    assert ak.str.match_substring_regex(
        string_repeats, r"FOO\d+", ignore_case=True
    ).tolist() == [
        [True, False, False],
        [False, False, True],
        [],
    ]
    assert (
        ak.str.match_substring_regex(
            string_repeats, r"FOO\d+", ignore_case=True
        ).layout.form
        == ak.str.match_substring_regex(
            ak.to_backend(string_repeats, "typetracer"), r"FOO\d+", ignore_case=True
        ).layout.form
    )

    # Bytestrings
    assert ak.str.match_substring_regex(bytestring_repeats, rb"FOO\d+").tolist() == [
        [False, False, False],
        [False, False, False],
        [],
    ]
    assert (
        ak.str.match_substring_regex(bytestring_repeats, rb"FOO\d+").layout.form
        == ak.str.match_substring_regex(
            ak.to_backend(bytestring_repeats, "typetracer"), rb"FOO\d+"
        ).layout.form
    )

    assert ak.str.match_substring_regex(
        bytestring_repeats, rb"FOO\d+", ignore_case=True
    ).tolist() == [
        [True, False, False],
        [False, False, True],
        [],
    ]
    assert (
        ak.str.match_substring_regex(
            bytestring_repeats, rb"FOO\d+", ignore_case=True
        ).layout.form
        == ak.str.match_substring_regex(
            ak.to_backend(bytestring_repeats, "typetracer"),
            rb"FOO\d+",
            ignore_case=True,
        ).layout.form
    )


def test_is_in():
    assert ak.str.is_in(string_repeats, ["123foo", "foo"]).tolist() == [
        [False, True, False],
        [True, False, False],
        [],
    ]
    assert (
        ak.str.is_in(string_repeats, ["123foo", "foo"]).layout.form
        == ak.str.is_in(
            ak.to_backend(string_repeats, "typetracer"), ["123foo", "foo"]
        ).layout.form
    )

    assert ak.str.is_in(
        [
            ["foo123bar123baz", "foo", "bar"],
            ["123foo", "456bar", "foo123456bar"],
            [None],
        ],
        ["123foo", "foo", None],
    ).tolist() == [
        [False, True, False],
        [True, False, False],
        [True],
    ]
    assert (
        ak.str.is_in(
            [
                ["foo123bar123baz", "foo", "bar"],
                ["123foo", "456bar", "foo123456bar"],
                [None],
            ],
            ["123foo", "foo", None],
        ).layout.form
        == ak.str.is_in(
            ak.to_backend(
                [
                    ["foo123bar123baz", "foo", "bar"],
                    ["123foo", "456bar", "foo123456bar"],
                    [None],
                ],
                "typetracer",
            ),
            ["123foo", "foo", None],
        ).layout.form
    )

    assert ak.str.is_in(
        [
            ["foo123bar123baz", "foo", "bar"],
            ["123foo", "456bar", "foo123456bar"],
            [None],
        ],
        ["123foo", "foo", None],
        skip_nones=True,
    ).tolist() == [
        [False, True, False],
        [True, False, False],
        [False],
    ]
    assert (
        ak.str.is_in(
            [
                ["foo123bar123baz", "foo", "bar"],
                ["123foo", "456bar", "foo123456bar"],
                [None],
            ],
            ["123foo", "foo", None],
            skip_nones=True,
        ).layout.form
        == ak.str.is_in(
            ak.to_backend(
                [
                    ["foo123bar123baz", "foo", "bar"],
                    ["123foo", "456bar", "foo123456bar"],
                    [None],
                ],
                "typetracer",
            ),
            ["123foo", "foo", None],
            skip_nones=True,
        ).layout.form
    )

    # Bytestrings
    assert ak.str.is_in(bytestring_repeats, [b"123foo", b"foo"]).tolist() == [
        [False, True, False],
        [True, False, False],
        [],
    ]
    assert (
        ak.str.is_in(bytestring_repeats, [b"123foo", b"foo"]).layout.form
        == ak.str.is_in(
            ak.to_backend(bytestring_repeats, "typetracer"), [b"123foo", b"foo"]
        ).layout.form
    )

    assert ak.str.is_in(
        [
            [b"foo123bar123baz", b"foo", b"bar"],
            [b"123foo", b"456bar", b"foo123456bar"],
            [None],
        ],
        [b"123foo", b"foo", None],
    ).tolist() == [
        [False, True, False],
        [True, False, False],
        [True],
    ]
    assert (
        ak.str.is_in(
            [
                [b"foo123bar123baz", b"foo", b"bar"],
                [b"123foo", b"456bar", b"foo123456bar"],
                [None],
            ],
            [b"123foo", b"foo", None],
        ).layout.form
        == ak.str.is_in(
            ak.to_backend(
                [
                    [b"foo123bar123baz", b"foo", b"bar"],
                    [b"123foo", b"456bar", b"foo123456bar"],
                    [None],
                ],
                "typetracer",
            ),
            [b"123foo", b"foo", None],
        ).layout.form
    )

    assert ak.str.is_in(
        [
            [b"foo123bar123baz", b"foo", b"bar"],
            [b"123foo", b"456bar", b"foo123456bar"],
            [None],
        ],
        [b"123foo", b"foo", None],
        skip_nones=True,
    ).tolist() == [
        [False, True, False],
        [True, False, False],
        [False],
    ]
    assert (
        ak.str.is_in(
            [
                [b"foo123bar123baz", b"foo", b"bar"],
                [b"123foo", b"456bar", b"foo123456bar"],
                [None],
            ],
            [b"123foo", b"foo", None],
            skip_nones=True,
        ).layout.form
        == ak.str.is_in(
            ak.to_backend(
                [
                    [b"foo123bar123baz", b"foo", b"bar"],
                    [b"123foo", b"456bar", b"foo123456bar"],
                    [None],
                ],
                "typetracer",
            ),
            [b"123foo", b"foo", None],
            skip_nones=True,
        ).layout.form
    )


def test_index_in():
    assert ak.str.index_in(string_repeats, ["123foo", "foo"]).tolist() == [
        [None, 1, None],
        [0, None, None],
        [],
    ]
    assert (
        ak.str.index_in(string_repeats, ["123foo", "foo"]).layout.form
        == ak.str.index_in(
            ak.to_backend(string_repeats, "typetracer"), ["123foo", "foo"]
        ).layout.form
    )

    assert ak.str.index_in(
        [
            ["foo123bar123baz", "foo", "bar"],
            ["123foo", "456bar", "foo123456bar"],
            [None],
        ],
        ["123foo", "foo", None],
    ).tolist() == [
        [None, 1, None],
        [0, None, None],
        [2],
    ]
    assert (
        ak.str.index_in(
            [
                ["foo123bar123baz", "foo", "bar"],
                ["123foo", "456bar", "foo123456bar"],
                [None],
            ],
            ["123foo", "foo", None],
        ).layout.form
        == ak.str.index_in(
            ak.to_backend(
                [
                    ["foo123bar123baz", "foo", "bar"],
                    ["123foo", "456bar", "foo123456bar"],
                    [None],
                ],
                "typetracer",
            ),
            ["123foo", "foo", None],
        ).layout.form
    )

    assert ak.str.index_in(
        [
            ["foo123bar123baz", "foo", "bar"],
            ["123foo", "456bar", "foo123456bar"],
            [None],
        ],
        ["123foo", "foo", None],
        skip_nones=True,
    ).tolist() == [
        [None, 1, None],
        [0, None, None],
        [None],
    ]
    assert (
        ak.str.index_in(
            [
                ["foo123bar123baz", "foo", "bar"],
                ["123foo", "456bar", "foo123456bar"],
                [None],
            ],
            ["123foo", "foo", None],
            skip_nones=True,
        ).layout.form
        == ak.str.index_in(
            ak.to_backend(
                [
                    ["foo123bar123baz", "foo", "bar"],
                    ["123foo", "456bar", "foo123456bar"],
                    [None],
                ],
                "typetracer",
            ),
            ["123foo", "foo", None],
            skip_nones=True,
        ).layout.form
    )

    # Bytestrings
    assert ak.str.index_in(string_repeats, [b"123foo", b"foo"]).tolist() == [
        [None, 1, None],
        [0, None, None],
        [],
    ]
    assert (
        ak.str.index_in(bytestring_repeats, [b"123foo", b"foo"]).layout.form
        == ak.str.index_in(
            ak.to_backend(bytestring_repeats, "typetracer"), [b"123foo", b"foo"]
        ).layout.form
    )

    assert ak.str.index_in(
        [
            [b"foo123bar123baz", b"foo", b"bar"],
            [b"123foo", b"456bar", b"foo123456bar"],
            [None],
        ],
        [b"123foo", b"foo", None],
    ).tolist() == [
        [None, 1, None],
        [0, None, None],
        [2],
    ]
    assert (
        ak.str.index_in(
            [
                [b"foo123bar123baz", b"foo", b"bar"],
                [b"123foo", b"456bar", b"foo123456bar"],
                [None],
            ],
            [b"123foo", b"foo", None],
        ).layout.form
        == ak.str.index_in(
            ak.to_backend(
                [
                    [b"foo123bar123baz", b"foo", b"bar"],
                    [b"123foo", b"456bar", b"foo123456bar"],
                    [None],
                ],
                "typetracer",
            ),
            [b"123foo", b"foo", None],
        ).layout.form
    )

    assert ak.str.index_in(
        [
            [b"foo123bar123baz", b"foo", b"bar"],
            [b"123foo", b"456bar", b"foo123456bar"],
            [None],
        ],
        [b"123foo", b"foo", None],
        skip_nones=True,
    ).tolist() == [
        [None, 1, None],
        [0, None, None],
        [None],
    ]
    assert (
        ak.str.index_in(
            [
                [b"foo123bar123baz", b"foo", b"bar"],
                [b"123foo", b"456bar", b"foo123456bar"],
                [None],
            ],
            [b"123foo", b"foo", None],
            skip_nones=True,
        ).layout.form
        == ak.str.index_in(
            ak.to_backend(
                [
                    [b"foo123bar123baz", b"foo", b"bar"],
                    [b"123foo", b"456bar", b"foo123456bar"],
                    [None],
                ],
                "typetracer",
            ),
            [b"123foo", b"foo", None],
            skip_nones=True,
        ).layout.form
    )


def test_to_categorical():
    assert (
        ak.str.to_categorical(["foo", "bar", "bar", "fee"]).layout.form
        == ak.str.to_categorical(
            ak.to_backend(["foo", "bar", "bar", "fee"], "typetracer")
        ).layout.form
    )
