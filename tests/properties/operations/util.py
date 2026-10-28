# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Helpers shared by the operation test modules."""

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import awkward as ak


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


def merges_unmergeable_leaves(form: ak.forms.Form, drop_nones: bool) -> bool:
    """Return `True` if flattening all levels would merge leaves that cannot merge.

    `ak.ravel` and `ak.flatten(axis=None)` refuse, with a
    `ValueError`, leaves that `ak._do.mergeable` does not accept
    together. The leaves are collected from a typetracer of the form
    with the operation's `drop_nones` (`False` for `ak.ravel`, `True`
    for `ak.flatten`), so every branch of a union counts, whether or
    not any value reaches it. A leaf of unknown type, with or without
    an option above it, merges with anything and is not compared.
    """
    layout = ak.typetracer.typetracer_from_form(form, highlevel=False)
    leaves = [
        x
        for x in ak._do.remove_structure(layout, drop_nones=drop_nones)
        if not (x.is_unknown or (x.is_option and x.content.is_unknown))
    ]
    return any(not ak._do.mergeable(leaves[0], x) for x in leaves[1:])
