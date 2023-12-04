from __future__ import annotations

import pytest

from awkward._pickle import use_builtin_reducer


@pytest.fixture(autouse=True)
def disable_external_pickler():
    """Fixture to disable external pickler implementation for every test"""
    with use_builtin_reducer():
        yield
