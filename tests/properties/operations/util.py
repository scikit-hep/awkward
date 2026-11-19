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


def without_categorical(a: ak.Array) -> ak.Array:
    """Return `a` with the `"categorical"` label dropped from every node.

    A conversion that writes a categorical array as its projected values
    (`to_json`, `to_numpy`, and `to_arrow` with the default
    `categorical_as_dictionary=False`) returns it unlabeled, so a
    roundtrip result equals this array, not `a`. A non-option categorical
    node is replaced by its projection, because a union may hold an
    `IndexedArray` only if it is categorical; the other nodes are rebuilt
    with `copy` rather than their simplifying constructors, which could
    merge union contents.
    """
    return ak.Array(_without_categorical(a.layout), behavior=a.behavior)


def _without_categorical(layout: ak.contents.Content) -> ak.contents.Content:
    parameters = dict(layout.parameters)
    if parameters.get("__array__") == "categorical":
        if not layout.is_option:
            return layout.project()
        del parameters["__array__"]
    if layout.is_unknown or layout.is_numpy:
        return layout
    if layout.is_record or layout.is_union:
        contents = [_without_categorical(c) for c in layout.contents]
        return layout.copy(contents=contents, parameters=parameters)
    content = _without_categorical(layout.content)
    return layout.copy(content=content, parameters=parameters)
