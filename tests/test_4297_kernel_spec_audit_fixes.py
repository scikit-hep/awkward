# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Regression tests for divergences found auditing `kernel-specification.yml`
against the CPU kernels in `awkward-cpp/src/cpu-kernels/`."""

import ctypes

import numpy as np
import pytest

import awkward as ak

def string_array(raw_bytes, offsets):
    """Build a string array over a raw byte buffer, which a list of `str`
    cannot express."""
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.asarray(offsets, dtype=np.int64)),
            ak.contents.NumpyArray(
                np.frombuffer(raw_bytes, dtype=np.uint8),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        )
    )


def test_invalid_utf8_raises_instead_of_hanging():
    # 0x80 is a continuation byte, so it has no code-point width; the prepare
    # kernel used to advance by 0 and spin forever inside C.
    array = string_array(b"\x80", [0, 1])
    with pytest.raises(ValueError, match="invalid byte in UTF8 string"):
        ak.to_numpy(array)


def test_truncated_utf8_raises_instead_of_corrupting():
    # "\xe2\x82" opens a three-byte sequence that the sublist does not finish,
    # so the decode used to run into the following string.
    array = string_array(b"\xe2\x82" + b"XY", [0, 2, 4])
    with pytest.raises(ValueError, match="truncated UTF8 sequence"):
        ak.to_numpy(array)


def test_valid_utf8_of_every_width_still_converts():
    array = ak.Array(["a", "é", "€", "\U0001d11e", ""])
    assert ak.to_numpy(array).tolist() == ["a", "é", "€", "\U0001d11e", ""]


@pytest.mark.parametrize("operation", [ak.sort, ak.argsort])
def test_string_sort_with_embedded_nul(operation):
    # `strncmp` stopped at the NUL, so these compared equal on their prefix and
    # were then ordered only by length -- i.e. not ordered at all.
    data = ["a\x00z", "a\x00a", "a\x00m"]
    result = operation(ak.Array(data)).to_list()
    if operation is ak.sort:
        assert result == sorted(data)
    else:
        assert result == [1, 2, 0]


def test_zero_slice_step_raises_instead_of_hanging():
    listoffset = ak.Array([[1, 2, 3], [4, 5]])
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        listoffset[:, ::0]

    # RegularArray already raised; check the two agree
    regular = ak.Array(np.arange(6).reshape(3, 2))
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        regular[:, ::0]


def test_min_range_leaves_output_alone_when_empty():
    from awkward_cpp.cpu_kernels import lib

    tomin = (ctypes.c_int64 * 1)(-12345)
    empty = (ctypes.c_int64 * 1)(0)
    assert not lib.awkward_ListArray64_min_range(tomin, empty, empty, 0).str
    assert tomin[0] == -12345


def test_unsigned_list_offsets_do_not_wrap_around():
    from awkward_cpp.cpu_kernels import lib

    # stops < starts must be an error, not a ~4.29e9 length
    tocarry = (ctypes.c_int64 * 4)(0, 0, 0, 0)
    error = lib.awkward_ListArrayU32_getitem_next_at_64(
        tocarry, (ctypes.c_uint32 * 1)(3), (ctypes.c_uint32 * 1)(1), 1, 0
    )
    assert error.str == b"index out of range"


def test_overlay_mask_normalizes_to_zero_or_one():
    from awkward_cpp.cpu_kernels import lib

    tomask = (ctypes.c_int8 * 4)(9, 9, 9, 9)
    theirs = (ctypes.c_int8 * 4)(2, -1, 1, 0)
    mine = (ctypes.c_int8 * 4)(0, 0, 0, 0)
    assert not lib.awkward_ByteMaskedArray_overlay_mask8(
        tomask, theirs, mine, 4, True
    ).str
    assert list(tomask) == [1, 1, 1, 1]


def test_outstartsstops_rejects_ragged_distincts():
    from awkward_cpp.cpu_kernels import lib

    outstarts = (ctypes.c_int64 * 4)(0, 0, 0, 0)
    outstops = (ctypes.c_int64 * 4)(0, 0, 0, 0)
    distincts = (ctypes.c_int64 * 7)(*range(7))
    error = lib.awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64(
        outstarts, outstops, distincts, 7, 3
    )
    assert error.str == b"lendistincts is not a multiple of outlength"


def test_drop_none_indexes_checks_every_offset():
    from awkward_cpp.cpu_kernels import lib

    # The last offset is in range only because the offsets are not monotonic;
    # the intermediate one still runs off the end of `noneindexes`.
    tooffsets = (ctypes.c_int64 * 3)(0, 0, 0)
    noneindexes = (ctypes.c_int64 * 3)(1, -1, 0)
    fromoffsets = (ctypes.c_int64 * 3)(5, 5, 5)
    error = lib.awkward_ListOffsetArray_drop_none_indexes_64(
        tooffsets, noneindexes, fromoffsets, 3, 3
    )
    assert error.str == b"offsets[i] > len(content)"


def test_local_preparenext_sorts_only_the_requested_length():
    from awkward_cpp.cpu_kernels import lib

    tocarry = (ctypes.c_int64 * 5)(0, 0, 0, 0, 0)
    fromindex = (ctypes.c_int64 * 8)(9, 8, 7, 6, 5, 0, 1, 2)
    assert not lib.awkward_ListOffsetArray_local_preparenext_64(
        tocarry, fromindex, 5
    ).str
    assert list(tocarry) == [4, 3, 2, 1, 0]


def test_local_preparenext_is_stable_across_ties():
    from awkward_cpp.cpu_kernels import lib

    # Duplicate keys, and long enough that an unstable sort would not fall back
    # to insertion sort
    keys = [i % 4 for i in range(64)]
    tocarry = (ctypes.c_int64 * 64)(*([0] * 64))
    assert not lib.awkward_ListOffsetArray_local_preparenext_64(
        tocarry, (ctypes.c_int64 * 64)(*keys), 64
    ).str
    assert list(tocarry) == sorted(range(64), key=lambda i: keys[i])


def test_missing_repeat_keeps_missing_entries_missing():
    from awkward_cpp.cpu_kernels import lib

    outindex = (ctypes.c_int64 * 6)(*([999] * 6))
    index = (ctypes.c_int64 * 3)(0, -1, 2)
    assert not lib.awkward_missing_repeat_64(outindex, index, 3, 2, 3).str
    assert list(outindex) == [0, -1, 2, 3, -1, 5]


def test_unique_offsets_ignores_zero_length():
    from awkward_cpp.cpu_kernels import lib

    tooffsets = (ctypes.c_int64 * 2)(7, 7)
    fromoffsets = (ctypes.c_int64 * 1)(0)
    starts = (ctypes.c_int64 * 1)(0)
    assert not lib.awkward_unique_offsets_int64(
        tooffsets, 0, fromoffsets, starts, 1
    ).str
    # Nothing to read, so nothing is written
    assert list(tooffsets) == [7, 7]
