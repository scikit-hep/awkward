# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Callable, Collection, Generator, Mapping
from functools import lru_cache, wraps
from inspect import isgenerator

from packaging.requirements import Requirement
from packaging.version import Version

if sys.version_info < (3, 12):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

from awkward._errors import OperationErrorContext
from awkward._typing import Any, NamedTuple, TypeAlias, TypeVar

T = TypeVar("T")
DispatcherType: TypeAlias = "Callable[..., Generator[Collection[Any], None, T]]"
HighLevelType: TypeAlias = "Callable[..., T]"


class DependencyGroup(NamedTuple):
    name: str
    dependencies: tuple[str, ...]


class DependencySpecification(NamedTuple):
    groups: tuple[DependencyGroup, ...]
    non_groups: tuple[str, ...]


def normalize_dependency_specification(
    dependencies: Collection[str] | Mapping[str, Collection[str]] | None,
) -> DependencySpecification:
    """Normalize a dependency specification into a hashable object"""
    if dependencies is None:
        return DependencySpecification((), ())
    elif isinstance(dependencies, Mapping):
        return DependencySpecification(
            groups=tuple(
                DependencyGroup(key, tuple(values))
                for key, values in dependencies.items()
            ),
            non_groups=(),
        )
    else:
        return DependencySpecification(groups=(), non_groups=tuple(dependencies))


def iter_missing_dependencies(dependencies: tuple[str, ...]):
    """Build an iterator over the requirement, version pairs of dependencies that are not
    satisfied by the current environment"""
    for _requirement in dependencies:
        requirement = Requirement(_requirement)
        if requirement.extras:
            raise RuntimeError(
                "High-level functions must not declare dependencies specifications with extras"
            )

        try:
            _version = importlib_metadata.version(requirement.name)
        except importlib_metadata.PackageNotFoundError:
            version = None
        else:
            # If we found the distribution but did not resolve a version
            if _version is None:
                # Let's try and import it
                mod = importlib.import_module(requirement.name)
                try:
                    _version = mod.__version__
                except AttributeError:
                    warnings.warn(
                        f"Could not identify the version of installed package {requirement.name}",
                        stacklevel=2,
                    )
                    continue
            version = Version(_version)

        if version is None or version not in requirement.specifier:
            yield requirement, version


@lru_cache
def build_runtime_dependency_validation_error(
    dependency_spec: DependencySpecification,
) -> Exception | None:
    """Build an exception object for the given dependency specification if it
    is not satisfied by the current environment"""
    missing_extras = []
    missing_dependencies: list[tuple[Requirement, Version | None]] = []
    if dependency_spec.groups:
        for extra, extra_dependencies in dependency_spec.groups:
            extra_missing_dependencies = [
                *iter_missing_dependencies(extra_dependencies)
            ]
            missing_dependencies.extend(extra_missing_dependencies)
            if extra_missing_dependencies:
                missing_extras.append(extra)
    else:
        missing_dependencies[:] = iter_missing_dependencies(dependency_spec.non_groups)

    if not missing_dependencies:
        return None

    missing_requirement_lines = [
        (
            f"    * {req} — you do not have this package"
            if ver is None
            else f"    * {req} — you have {ver} installed"
        )
        for req, ver in missing_dependencies
    ]
    missing_requirement_message = "\n".join(missing_requirement_lines)

    missing_extras_lines = [f"    * {extra}" for extra in missing_extras]
    missing_extras_string = "\n".join(missing_extras_lines)
    maybe_missing_extras_message = (
        (
            "You can fix this error by installing these packages directly, or install awkward with "
            "all of the following extras:\n\n"
            f"{missing_extras_string}"
        )
        if missing_extras
        else ""
    )
    return ImportError(
        f"This function has the following dependency requirements that are not met by your current environment:\n\n"
        f"{missing_requirement_message}\n\n"
        f"{maybe_missing_extras_message}"
    )


def validate_runtime_dependencies(
    dependency_spec: DependencySpecification,
):
    exception = build_runtime_dependency_validation_error(dependency_spec)
    if exception is None:
        return
    else:
        raise exception


def high_level_function(
    module: str = "ak",
    name: str | None = None,
    *,
    dependencies: Collection[str] | Mapping[str, Collection[str]] | None = None,
) -> Callable[[DispatcherType], HighLevelType]:
    """Decorate a high-level function such that it may be overloaded by third-party array objects"""

    dependency_spec = normalize_dependency_specification(dependencies)

    def capture_func(func: DispatcherType) -> HighLevelType:
        if name is None:
            captured_name = func.__qualname__
        else:
            captured_name = name
        return named_high_level_function(
            func, f"{module}.{captured_name}", dependency_spec
        )

    return capture_func


def named_high_level_function(
    func: DispatcherType,
    name: str,
    dependency_spec: DependencySpecification,
) -> HighLevelType:
    """Decorate a named high-level function such that it may be overloaded by third-party array objects"""

    @wraps(func)
    def dispatch(*args, **kwargs):
        # NOTE: this decorator assumes that the operation is exposed under `ak.`
        with OperationErrorContext(name, args, kwargs):
            gen_or_result = func(*args, **kwargs)
            if isgenerator(gen_or_result):
                array_likes = next(gen_or_result)
                assert isinstance(array_likes, Collection)

                # Permit a third-party array object to intercept the invocation
                for array_like in array_likes:
                    try:
                        custom_impl = array_like.__awkward_function__
                    except AttributeError:
                        continue
                    else:
                        result = custom_impl(dispatch, array_likes, args, kwargs)

                        # Future-proof the implementation by permitting the `__awkward_function__`
                        # to return `NotImplemented`. This may later be used to signal that another
                        # overload should be used.
                        if result is NotImplemented:
                            raise NotImplementedError
                        else:
                            return result

                # Failed to find a custom overload, so resume the original function
                validate_runtime_dependencies(dependency_spec)
                try:
                    next(gen_or_result)
                except StopIteration as err:
                    return err.value
                else:
                    raise AssertionError(
                        "high-level functions should only implement a single yield statement"
                    )

            return gen_or_result

    return dispatch
