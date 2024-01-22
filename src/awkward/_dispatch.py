# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Callable, Collection, Generator, Mapping
from functools import lru_cache, partial, wraps
from inspect import isgeneratorfunction

import packaging.version
from packaging.requirements import Requirement
from packaging.version import Version

if sys.version_info < (3, 12):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

from awkward._errors import with_named_operation_context
from awkward._typing import Any, NamedTuple, TypeAlias, TypeVar

T = TypeVar("T")
DispatcherType: TypeAlias = "Callable[..., Generator[Collection[Any], None, T]]"
DispatchedType: TypeAlias = "Callable[..., T]"


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


def regularize_dunder_version(version: str) -> str:
    return version.replace("/", ".")


def iter_missing_dependencies(dependencies: tuple[str, ...]):
    """Build an iterator over the requirement, version pairs of dependencies that are not
    satisfied by the current environment"""
    for _requirement in dependencies:
        requirement = Requirement(_requirement)
        if requirement.extras:
            raise RuntimeError(
                "High-level functions must not declare dependencies specifications with extras"
            )

        # Try and find version
        try:
            # Try and get the version the canonical way
            _version = importlib_metadata.version(requirement.name)
        except importlib_metadata.PackageNotFoundError:
            # Otherwise, fall back on `__version__` (e.g. for ROOT)
            mod = importlib.import_module(requirement.name)
            try:
                mod_version = mod.__version__
            except AttributeError:
                warnings.warn(
                    f"Could not identify the version of installed package {requirement.name}",
                    stacklevel=2,
                )
                # Don't treat this as an error
                continue
            # Packages like ROOT seem to be playing poorly with standards, so we'll apply a simple regularization transform
            _version = regularize_dunder_version(mod_version)

        # Try and parse version
        try:
            version = Version(_version)
        except packaging.version.InvalidVersion:
            warnings.warn(
                f"Could not parse the version of installed package {requirement.name}: {_version!r}",
                stacklevel=2,
            )
            # Don't treat this as an error
            continue

        if version not in requirement.specifier:
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

    # Install string
    missing_dependencies_string = " ".join(
        [str(req) for req, ver in missing_dependencies]
    )
    missing_requirements_direct_message = (
        f"If you use pip, you can install these packages with "
        f"`python -m pip install {missing_dependencies_string}`.\n"
        "Otherwise, if you use Conda, install the corresponding packages "
        "for the correct versions. "
    )

    missing_extras_lines = [f"    * {extra}" for extra in missing_extras]
    missing_extras_list_string = "\n".join(missing_extras_lines)
    missing_extras_string = ",".join(missing_extras)
    maybe_missing_extras_message = (
        (
            f"{missing_requirements_direct_message}\n\n"
            "These dependencies can also be conveniently installed using the following extras:\n\n"
            f"{missing_extras_list_string}\n\n"
            f"If you're using `pip`, then you can install these extras with `pip install awkward[{missing_extras_string}]`"
        )
        if missing_extras
        else missing_requirements_direct_message
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


def on_dispatch_trivial():
    return


def with_type_dispatch(
    func: DispatcherType, on_dispatch_internal: Callable[[], None] = on_dispatch_trivial
) -> DispatchedType:
    if isgeneratorfunction(func):

        @wraps(func)
        def dispatch(*args, **kwargs):
            gen_or_result = func(*args, **kwargs)
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
            on_dispatch_internal()

            try:
                next(gen_or_result)
            except StopIteration as err:
                return err.value
            else:
                raise AssertionError(
                    "high-level functions should only implement a single yield statement"
                )
    else:

        @wraps(func)
        def dispatch(*args, **kwargs):
            on_dispatch_internal()
            return func(*args, **kwargs)

    return dispatch


def high_level_function(
    module: str = "ak",
    name: str | None = None,
    *,
    dependencies: Collection[str] | Mapping[str, Collection[str]] | None = None,
) -> Callable[[DispatcherType], DispatchedType]:
    """Decorate a high-level function such that it may be overloaded by third-party array objects"""

    # Callback for dispatches that use internal implementation
    on_dispatch_internal = partial(
        validate_runtime_dependencies, normalize_dependency_specification(dependencies)
    )

    def capture_func(func: DispatcherType) -> DispatchedType:
        if name is None:
            captured_name = func.__qualname__
        else:
            captured_name = name

        return with_named_operation_context(
            with_type_dispatch(func, on_dispatch_internal=on_dispatch_internal),
            f"{module}.{captured_name}",
        )

    return capture_func
