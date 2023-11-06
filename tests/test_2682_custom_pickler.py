# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import multiprocessing
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

if sys.version_info < (3, 12):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

import pytest

import awkward as ak


def has_entry_point():
    return bool(importlib_metadata.entry_points(group="awkward.pickle.reduce").names)


pytestmark = pytest.mark.skipif(
    has_entry_point(),
    reason="Custom pickler is already registered!",
)


def _init_process_with_pickler(pickler_source: str, tmp_path):
    # Create custom plugin
    (tmp_path / "impl_pickler.py").write_bytes(pickler_source.encode("UTF-8"))
    dist_info = tmp_path / "impl_pickler-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "entry_points.txt").write_bytes(
        b"[awkward.pickle.reduce]\nimpl = impl_pickler:plugin\n"
    )
    sys.path.insert(0, os.fsdecode(tmp_path))


def _pickle_complex_array_and_return_form_impl():
    array = ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])[[0, 2]]
    return pickle.loads(pickle.dumps(array)).layout.form


def pickle_complex_array_and_return_form(pickler_source, tmp_path):
    """Create a new (spawned) process, and register the given pickler source
    via entrypoints"""
    with ProcessPoolExecutor(
        1,
        initializer=_init_process_with_pickler,
        initargs=(pickler_source, tmp_path),
        # Don't fork the current process with all of its state
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        pickle_future = executor.submit(_pickle_complex_array_and_return_form_impl)
        return pickle_future.result()


def test_default_pickler():
    assert _pickle_complex_array_and_return_form_impl() == ak.forms.from_dict(
        {"class": "ListOffsetArray", "offsets": "i64", "content": "int64"}
    )


def test_noop_pickler(tmp_path):
    assert pickle_complex_array_and_return_form(
        """
def plugin(obj, protocol: int):
    return NotImplemented""",
        tmp_path,
    ) == ak.forms.from_dict(
        {"class": "ListOffsetArray", "offsets": "i64", "content": "int64"}
    )


def test_non_packing_pickler(tmp_path):
    assert pickle_complex_array_and_return_form(
        """
def plugin(obj, protocol):
    import awkward as ak
    if isinstance(obj, ak.Array):
        form, length, container = ak.to_buffers(obj)
        return (
            object.__new__,
            (ak.Array,),
            (form.to_dict(), length, container, obj.behavior),
        )
    else:
        return NotImplemented""",
        tmp_path,
    ) == ak.forms.from_dict(
        {"class": "ListArray", "starts": "i64", "stops": "i64", "content": "int64"}
    )


def test_malformed_pickler(tmp_path):
    with pytest.raises(RuntimeError, match=r"malformed pickler!"):
        pickle_complex_array_and_return_form(
            """
def plugin(obj, protocol: int):
    raise RuntimeError('malformed pickler!')""",
            tmp_path,
        )
