# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

from awkward._experimental import experimental


@pytest.mark.parametrize("descriptor", [classmethod, property, staticmethod])
def test_descriptor_placement_is_rejected(descriptor):
    def method(arg):
        return arg

    with pytest.raises(TypeError, match="directly decorate a plain function"):
        experimental(descriptor(method))


def test_callable_non_function_is_rejected():
    class Callable:
        def __call__(self):
            return None

    with pytest.raises(TypeError, match="directly decorate a plain function"):
        experimental(Callable())
