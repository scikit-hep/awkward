# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from awkward._errors import deprecate
from awkward.highlevel import Array


class CategoricalBehavior(Array):
    __name__ = "Array"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        deprecate(
            f"{type(self).__name__} is deprecated: categorical types are now considered a built-in feature "
            f"provided by Awkward Array, rather than an extension.",
            version="2.4.0",
        )
