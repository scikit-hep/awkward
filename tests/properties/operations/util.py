# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Helpers shared by the operation test modules."""

import inspect
from collections.abc import Callable, Mapping
from typing import Any


def assert_kwargs_match_signature(
    func: Callable[..., Any],
    data_param_names: set[str],
    kwargs_cls: type,
    defaults: Mapping[str, Any],
    related_cls: type,
) -> None:
    """Raise unless the option declarations agree with `func`'s parameters.

    `kwargs_cls` and `related_cls` are the module's `TypedDict`
    classes and `defaults` its `DEFAULTS`. Asserts that `kwargs_cls`
    has exactly the parameters of `func` other than
    `data_param_names`, that `defaults` equals the signature's
    default values, and that `related_cls`'s keys are a subset of
    `kwargs_cls`'s.
    """
    option_params = {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if name not in data_param_names
    }
    keys = kwargs_cls.__required_keys__ | kwargs_cls.__optional_keys__
    assert keys == set(option_params)

    assert defaults == option_params

    related_keys = related_cls.__required_keys__ | related_cls.__optional_keys__
    assert related_keys <= keys
