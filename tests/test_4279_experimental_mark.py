# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import inspect
import types
import warnings

import pytest

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._experimental import _find_stack_level, experimental
from awkward.errors import ExperimentalWarning


def _experimental_warnings(caught):
    return [w for w in caught if issubclass(w.category, ExperimentalWarning)]


def _make_dispatch_sentinel():
    """Array-like that intercepts __awkward_function__ dispatch and records arguments."""
    captured = []

    def __awkward_function__(func, array_likes, args, kwargs):
        captured[:] = list(array_likes)
        return "intercepted"

    sentinel = types.SimpleNamespace(
        captured=captured,
        __awkward_function__=__awkward_function__,
    )
    return sentinel


def test_experimental_warning_is_public():
    assert issubclass(ExperimentalWarning, UserWarning)
    assert ak.errors.ExperimentalWarning is ExperimentalWarning
    assert "ExperimentalWarning" in ak.errors.__all__


def test_decorator_is_not_exported():
    assert not hasattr(ak, "experimental")
    assert not hasattr(ak.errors, "experimental")


def test_bare_form_warns_and_passes_through():
    @experimental
    def f(x, *, y=None):
        return (x, y)

    with pytest.warns(ExperimentalWarning, match=r"f is experimental\."):
        assert f(1, y=2) == (1, 2)

    # pytest runs with filterwarnings = ["error", ...], so a spurious second
    # warning would fail this call.
    assert f(3) == (3, None)


def test_parenthesized_form_equivalent():
    @experimental()
    def f():
        return 42

    with pytest.warns(ExperimentalWarning, match=r"f is experimental\."):
        assert f() == 42

    assert f() == 42

    def g():
        return None

    assert experimental(g).__wrapped__ is g
    assert experimental()(g).__wrapped__ is g


def test_arguments_rejected():
    with pytest.raises(TypeError, match="accepts no arguments"):
        experimental("not callable")

    with pytest.raises(TypeError, match="accepts no arguments"):
        experimental(123)

    with pytest.raises(TypeError):
        experimental(lambda: None, "extra")

    with pytest.raises(TypeError):
        experimental(func=lambda: None)

    with pytest.raises(TypeError):
        experimental(reason="x")


def test_message_text_exact():
    @experimental
    def stupendous_reduce():
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stupendous_reduce()

    (only,) = _experimental_warnings(caught)
    expected = (
        "test_message_text_exact.<locals>.stupendous_reduce is experimental.\n"
        "    It may change or be removed in any release, without a deprecation period.\n"
        f"    Defined in: {__name__}"
    )
    assert str(only.message) == expected
    assert only.category is ExperimentalWarning


def test_warns_once_under_always_filter():
    @experimental
    def f():
        return 1

    with warnings.catch_warnings(record=True) as caught:
        # "always" bypasses __warningregistry__, so only state owned by the
        # wrapper itself can keep this to a single warning.
        warnings.simplefilter("always")
        assert f() == 1
        assert f() == 1
        assert f() == 1

    assert len(_experimental_warnings(caught)) == 1


def test_warn_once_state_is_per_function():
    @experimental
    def first():
        return None

    @experimental
    def second():
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first()
        first()
        second()
        second()

    messages = [str(w.message) for w in _experimental_warnings(caught)]
    assert len(messages) == 2
    assert any(".first is experimental." in m for m in messages)
    assert any(".second is experimental." in m for m in messages)


def test_wraps_metadata():
    def raw():
        """Docstring survives the wrapper."""
        return None

    wrapped = experimental(raw)
    assert wrapped is not raw
    assert wrapped.__wrapped__ is raw
    # awkward._dispatch.high_level_function builds the dispatch name from the
    # __qualname__ of the object it wraps, so the mark must preserve it.
    assert wrapped.__qualname__ == raw.__qualname__
    assert wrapped.__name__ == raw.__name__
    assert wrapped.__doc__ == raw.__doc__
    assert wrapped.__module__ == raw.__module__


def test_method_warns_once_across_instances():
    class Thing:
        def __init__(self, x):
            self.x = x

        @experimental
        def frobnicate(self):
            return self.x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert Thing(1).frobnicate() == 1
        assert Thing(2).frobnicate() == 2

    (only,) = _experimental_warnings(caught)
    assert str(only.message).startswith(
        "test_method_warns_once_across_instances.<locals>.Thing.frobnicate"
        " is experimental."
    )


def test_property_warns_on_first_access_once():
    class Thing:
        def __init__(self, x):
            self.x = x

        @property
        @experimental
        def value(self):
            return self.x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Class-attribute access returns the property object without calling
        # the getter: no warning yet.
        assert isinstance(Thing.value, property)
        assert len(_experimental_warnings(caught)) == 0
        assert Thing(1).value == 1
        assert len(_experimental_warnings(caught)) == 1
        assert Thing(2).value == 2

    assert len(_experimental_warnings(caught)) == 1


def test_classmethod_warns_once():
    class Thing:
        @classmethod
        @experimental
        def make(cls, x):
            return (cls, x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert Thing.make(1) == (Thing, 1)
        assert Thing().make(2) == (Thing, 2)

    assert len(_experimental_warnings(caught)) == 1


def test_staticmethod_warns_once():
    class Thing:
        @staticmethod
        @experimental
        def double(x):
            return 2 * x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert Thing.double(3) == 6
        assert Thing().double(4) == 8

    assert len(_experimental_warnings(caught)) == 1


def test_warns_once_when_function_raises():
    @experimental
    def explode():
        raise ValueError("boom")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="boom"):
            explode()
        with pytest.raises(ValueError, match="boom"):
            explode()

    # The warning precedes the body call, and the once-per-process flag
    # survives the exception.
    assert len(_experimental_warnings(caught)) == 1


def test_first_use_raises_under_error_filter_then_usable():
    @experimental
    def f():
        return 42

    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        with pytest.raises(ExperimentalWarning):
            f()
        # The once-per-process flag is set before the warning is issued, so
        # the API stays usable after a filter escalated the warning to an
        # error.
        assert f() == 42


def test_stacklevel_direct_call():
    @experimental
    def f():
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f()

    (only,) = _experimental_warnings(caught)
    assert only.filename == __file__


def test_stacklevel_through_dispatch_generator():
    @high_level_function()
    @experimental
    def op(x):
        # Dispatch
        yield ()

        # Implementation
        return x + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert op(1) == 2

    (only,) = _experimental_warnings(caught)
    # Attributed to this file, not to awkward/_dispatch.py: the stacklevel is
    # computed, not fixed.
    assert only.filename == __file__
    assert op.__qualname__.endswith(".op")


def test_stacklevel_through_dispatch_plain_function():
    @high_level_function()
    @experimental
    def op(x):
        return x + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert op(1) == 2

    (only,) = _experimental_warnings(caught)
    assert only.filename == __file__


def test_dispatch_interception_still_warns():
    sentinel = _make_dispatch_sentinel()

    @high_level_function()
    @experimental
    def op(x):
        yield (x,)
        raise AssertionError("generator body must not resume under interception")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert op(sentinel) == "intercepted"

    assert any(x is sentinel for x in sentinel.captured)
    assert len(_experimental_warnings(caught)) == 1


def test_find_stack_level_without_frame_introspection(monkeypatch):
    # currentframe() is documented to return None on implementations without
    # stack-frame support; attribution then degrades to the warn call itself.
    monkeypatch.setattr("awkward._experimental.currentframe", lambda: None)
    assert _find_stack_level() == 1


def test_find_stack_level_when_every_frame_is_inside_the_package(monkeypatch):
    # An empty package-dir prefix makes every frame count as awkward's own:
    # the walk runs off the top of the stack and must attribute the warning
    # to the outermost real frame, not one past it.
    monkeypatch.setattr("awkward._experimental._PACKAGE_DIR", "")
    assert _find_stack_level() == len(inspect.stack())
