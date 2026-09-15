# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import warnings
from functools import cached_property

import pytest

from awkward._experimental import experimental
from awkward.errors import ExperimentalWarning

# Match the part of the message that carries the meaning, not the whole
# sentence: the wording may be reworded, the placement rule may not.
PLACEMENT = r"directly above the def"


def _experimental_warnings(caught):
    return [w for w in caught if issubclass(w.category, ExperimentalWarning)]


def test_above_staticmethod_rejected():
    # staticmethod objects are callable on every supported Python (3.10+), so
    # a callable() guard accepted this and returned a plain function: the
    # staticmethod was silently dropped, and the mark only broke later, at
    # instance access, where self was passed to a function not expecting it.
    with pytest.raises(TypeError, match=PLACEMENT) as excinfo:

        class Thing:
            @experimental
            @staticmethod
            def double(x):
                return 2 * x

    assert "staticmethod object" in str(excinfo.value)


def test_above_classmethod_rejected():
    with pytest.raises(TypeError, match=PLACEMENT) as excinfo:

        class Thing:
            @experimental
            @classmethod
            def make(cls, x):
                return (cls, x)

    assert "classmethod object" in str(excinfo.value)


def test_above_property_rejected():
    with pytest.raises(TypeError, match=PLACEMENT) as excinfo:

        class Thing:
            @experimental
            @property
            def value(self):
                return 1

    assert "property object" in str(excinfo.value)


def test_misplacement_and_argument_messages_are_distinct():
    # The issue: every rejected input was described as a misuse of the
    # parenthesized form, including the ones that are placement mistakes.
    for descriptor in (
        staticmethod(len),
        classmethod(len),
        property(len),
        cached_property(len),
    ):
        with pytest.raises(TypeError, match=PLACEMENT) as excinfo:
            experimental(descriptor)
        assert "accepts no arguments" not in str(excinfo.value)

    # An argument really was passed: that message stays, and stays specific.
    for argument in ("reason", 123):
        with pytest.raises(TypeError, match="accepts no arguments") as excinfo:
            experimental(argument)
        assert "directly above the def" not in str(excinfo.value)


def test_class_and_callable_object_rejected():
    class Thing:
        def __call__(self):
            return None

    with pytest.raises(TypeError, match="plain function"):
        experimental(Thing)

    with pytest.raises(TypeError, match="plain function"):
        experimental(Thing())


def test_both_supported_forms_still_accepted():
    @experimental
    def bare():
        return 1

    @experimental()
    def parenthesized():
        return 2

    with pytest.warns(ExperimentalWarning, match=r"bare is experimental\."):
        assert bare() == 1

    with pytest.warns(ExperimentalWarning, match=r"parenthesized is experimental\."):
        assert parenthesized() == 2


def test_valid_ordering_preserves_descriptor_behavior():
    class Thing:
        def __init__(self, x=0):
            self.x = x

        @staticmethod
        @experimental
        def double(x):
            return 2 * x

        @classmethod
        @experimental
        def make(cls, x):
            return (cls, x)

        @property
        @experimental
        def value(self):
            return self.x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Instance access is the path that silently broke when the mark
        # swallowed the staticmethod: self must not reach double().
        assert Thing.double(3) == 6
        assert Thing(1).double(4) == 8
        assert Thing.make(1) == (Thing, 1)
        assert Thing(1).make(2) == (Thing, 2)
        assert isinstance(Thing.value, property)
        assert Thing(5).value == 5

    assert len(_experimental_warnings(caught)) == 3
