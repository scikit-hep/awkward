# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Recognition of the ``@experimental`` decorator."""

# How awkward applies the decorator, as written in the source
_FORMS = ("experimental", "experimental()")


def is_experimental(decorator: str) -> bool:
    """Return whether a decorator, as source text, applies ``@experimental``.

    ``decorator`` is the text after ``@``, e.g. ``experimental`` or
    ``functools.cache``.
    """
    return decorator in _FORMS
