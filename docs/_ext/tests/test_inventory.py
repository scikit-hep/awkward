# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import zlib

import pytest
from awkward_docs.inventory import add_awkward_aliases
from hypothesis import given
from hypothesis import strategies as st
from strategies import dotted_names

HEADER = (
    b"# Sphinx inventory version 2\n"
    b"# Project: Awkward Array\n"
    b"# Version: 2.15\n"
    b"# The remainder of this file is compressed using zlib.\n"
)


def inventory(body: str) -> bytes:
    return HEADER + zlib.compress(body.encode("utf-8"))


def split(data: bytes) -> tuple[bytes, str]:
    header = data[: len(HEADER)]
    return header, zlib.decompress(data[len(HEADER) :]).decode("utf-8")


def test_add_awkward_aliases():
    body = (
        "ak.flatten py:function 1 reference/generated/ak.flatten.html#$ -\n"
        "ak.Array.to_list py:method 1 reference/generated/ak.Array.html#$ -\n"
        "index std:doc -1 index.html Awkward Array documentation\n"
    )
    header, new_body = split(add_awkward_aliases(inventory(body)))
    assert header == HEADER
    assert new_body == body + (
        "awkward.flatten py:function 1 reference/generated/ak.flatten.html#$ -\n"
        "awkward.Array.to_list py:method 1 reference/generated/ak.Array.html#$ -\n"
    )


def test_add_awkward_aliases_without_a_final_newline():
    body = "ak.flatten py:function 1 reference/generated/ak.flatten.html#$ -"
    _, new_body = split(add_awkward_aliases(inventory(body)))
    assert (
        new_body
        == body
        + "\nawkward.flatten py:function 1 reference/generated/ak.flatten.html#$ -\n"
    )


@pytest.mark.parametrize(
    "body",
    [
        "",
        "index std:doc -1 index.html Awkward Array documentation\n",
        "akx.thing py:data 1 x.html#$ -\n",
        "awkward.Array py:class 1 reference/generated/ak.Array.html#$ -\n",
        "xak.thing py:data 1 x.html#$ -\n",
    ],
)
def test_add_awkward_aliases_keeps_inventories_without_ak_entries(body):
    data = inventory(body)
    assert add_awkward_aliases(data) == data


@pytest.mark.parametrize(
    "data",
    [b"", HEADER[:-1], b"# Sphinx inventory version 2\n# Project: Awkward Array\n"],
)
def test_add_awkward_aliases_rejects_a_truncated_inventory(data):
    with pytest.raises((ValueError, zlib.error)):
        add_awkward_aliases(data)


def test_add_awkward_aliases_only_checks_the_name():
    # "ak." elsewhere in a line does not make it an ak.* entry
    data = inventory("guide std:doc -1 ak.html see ak.flatten\n")
    assert add_awkward_aliases(data) == data


@given(st.lists(st.tuples(st.booleans(), dotted_names), max_size=20))
def test_add_awkward_aliases_appends_one_alias_per_ak_entry(entries):
    lines = [
        f"{'ak.' if is_ak else ''}{name} py:function 1 x.html#$ -"
        for is_ak, name in entries
    ]
    body = "".join(line + "\n" for line in lines)
    header, new_body = split(add_awkward_aliases(inventory(body)))
    aliases = [
        "awkward." + line[3:]
        for line in lines
        if line.split(" ", 1)[0].startswith("ak.")
    ]
    assert header == HEADER
    if aliases:
        assert new_body.splitlines() == lines + aliases
    else:
        assert new_body == body
