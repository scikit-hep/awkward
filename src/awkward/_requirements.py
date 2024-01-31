# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import contextlib
import importlib
import sys
import types
import warnings
from collections.abc import Collection, Mapping
from functools import lru_cache, update_wrapper, wraps

import packaging.version
from packaging.requirements import Requirement
from packaging.version import Version

from awkward._util import UNSET

if sys.version_info < (3, 12):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

from awkward._typing import NamedTuple, Protocol


class UnmetRequirementError(Exception):
    def __init__(
        self,
        msg: str,
        name: str,
        distribution: str,
        dependency: Dependency,
        version: Version | None,
    ):
        super().__init__(msg)

        self._name = name
        self._distribution = distribution
        self._dependency = dependency
        self._version = version

    @property
    def name(self) -> str:
        return self._name

    @property
    def distribution(self) -> str:
        return self._distribution

    @property
    def dependency(self) -> Dependency:
        return self._dependency

    @property
    def version(self) -> Version | None:
        return self._version


class Dependency(NamedTuple):
    requirement: Requirement
    group: str | None
    module_name: str
    conda_forge_spec: str | None
    is_on_pypi: bool


class DependencySpecification(NamedTuple):
    unvalidated: dict[str, Dependency]
    validated: dict[str, Dependency]

    def on_validated(self, name: str):
        self.validated[name] = self.unvalidated.pop(name)


_dependencies_stack: list[DependencySpecification] = []


@contextlib.contextmanager
def dependency_specification_context(spec: DependencySpecification):
    global _dependencies_stack

    # Optimisation: do not push specification if we have no unvalidated requirements
    if spec.unvalidated:
        _dependencies_stack.append(spec)
        try:
            yield
        finally:
            _dependencies_stack.pop()
    else:
        yield


def _format_requirement_error(dependency: Dependency, version: Version | None) -> str:
    maybe_missing_extras_message = (
        (
            f"\n\nThis dependency can also be conveniently installed using the {dependency.group!r} extra. "
            f"If you're using `pip`, then you can install this extra with\n\n"
            f"    pip install awkward[{dependency.group}]"
        )
        if dependency.group
        else ""
    )
    version_string = (
        f"{dependency.requirement} is not installed."
        if version is None
        else f"you have {version} installed."
    )
    maybe_pypi_string = (
        f"If you use pip, you can install this package with\n\n"
        f"    pip install {dependency.requirement}\n\n"
        if dependency.is_on_pypi
        else ""
    )
    maybe_conda_forge_string = (
        ""
        if dependency.conda_forge_spec is None
        else (
            "If you use conda, you can install this package with\n\n"
            f"    conda install {dependency.conda_forge_spec}\n\n"
            f"You should check that the installed package satisfies {dependency.requirement}."
        )
    )
    return (
        f"This function requires {dependency.requirement} which is not met by your current environment: "
        f"{version_string}\n\n"
        f"{maybe_pypi_string}"
        f"{maybe_conda_forge_string}"
        f"{maybe_missing_extras_message}"
    )


def _find_dependency_for_module(name: str) -> Dependency:
    # Take only the FOO component of FOO.BAR.BAZ
    module_name, _, _ = name.partition(".")

    for spec in reversed(_dependencies_stack):
        for distribution, dependency in spec.unvalidated.items():
            if dependency.module_name == module_name:
                return dependency
        for distribution, dependency in spec.validated.items():
            if dependency.module_name == module_name:
                return dependency

    raise LookupError


@lru_cache
def _load_package_distributions():
    return importlib_metadata.packages_distributions()


def _regularize_dunder_version(version: str) -> str:
    return version.replace("/", ".")


class ImplementsRequires(Protocol):
    def add_requirement(
        self,
        requirement: Requirement,
        group: str | None,
        module_name: str,
        conda_forge_spec: str | None,
        is_on_pypi: bool,
    ):
        ...

    def get_specification(self) -> DependencySpecification:
        ...


class _WithHasRequirements(ImplementsRequires):
    _specification: DependencySpecification

    def __init__(self, func):
        update_wrapper(self, func)
        self._func = func
        self._dependencies = {}

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def add_requirement(
        self,
        requirement: Requirement,
        group: str | None,
        module_name: str,
        conda_forge_spec: str | None,
        is_on_pypi: bool,
    ):
        if group is not None:
            assert is_on_pypi

        self._dependencies[requirement.name] = Dependency(
            requirement,
            group,
            module_name=module_name,
            conda_forge_spec=conda_forge_spec,
            is_on_pypi=is_on_pypi,
        )

    def get_specification(self) -> DependencySpecification:
        try:
            return self._specification
        except AttributeError:
            self._specification = DependencySpecification(self._dependencies, {})
            return self._specification


def with_has_requirements(func):
    return _WithHasRequirements(func)


def requires(
    spec: str,
    *,
    group: str = UNSET,
    module_name: str = UNSET,
    conda_forge_spec: str | None = UNSET,
    is_on_pypi: bool = True,
):
    if group is UNSET:
        group = None

    requirement = Requirement(spec)
    if module_name is UNSET:
        module_name = requirement.name
    if conda_forge_spec is UNSET:
        conda_forge_spec = str(requirement)

    def decorator(func):
        if not hasattr(func, "add_requirement"):
            raise RuntimeError()

        func.add_requirement(
            requirement,
            group,
            module_name=module_name,
            conda_forge_spec=conda_forge_spec,
            is_on_pypi=is_on_pypi,
        )

        return func

    return decorator


def _determine_installed_requirement_version(
    distribution: str, module: types.ModuleType
) -> Version | None:
    # Try and find version
    try:
        # Try and get the version the canonical way
        _version = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        try:
            # Otherwise, fall back on `__version__` (e.g. for ROOT)
            mod_version = module.__version__
        except AttributeError:
            warnings.warn(
                f"Could not identify the version of installed package {distribution}",
                stacklevel=2,
            )
            # Don't treat this as an error
            return None
        # Packages like ROOT seem to be playing poorly with standards, so we'll apply a simple regularization transform
        _version = _regularize_dunder_version(mod_version)

    # Try and parse version
    try:
        version = Version(_version)
    except packaging.version.InvalidVersion:
        warnings.warn(
            f"Could not parse the version of installed package {distribution}: {_version!r}",
            stacklevel=2,
        )
        # Don't treat this as an error
        return None

    return version


def import_required_module(name: str) -> types.ModuleType:
    try:
        module = importlib.import_module(name)
    except ImportError as err:
        dependency = _find_dependency_for_module(name)

        raise UnmetRequirementError(
            _format_requirement_error(dependency, None),
            name,
            dependency.requirement.name,
            dependency,
            None,
        )

    # Find the distributions that provide `name`
    # For now, assume only one! This may struggle with namespace packages
    # Take only the FOO component of FOO.BAR.BAZ
    module_name, _, _ = name.partition(".")
    (distribution,) = _load_package_distributions()[module_name]

    for spec in _dependencies_stack:
        if distribution in spec.validated:
            continue

        try:
            dependency = spec.unvalidated[distribution]
        except KeyError:
            continue

        # Dependencies _themselves_ can't specify extras
        if dependency.requirement.extras:
            raise RuntimeError(
                "High-level functions must not declare dependencies specifications with extras"
            )
        # Try and find version
        version = _determine_installed_requirement_version(distribution, module)

        if version is None:
            continue

        # We need the right version
        if version not in dependency.requirement.specifier:
            raise UnmetRequirementError(
                _format_requirement_error(dependency, version),
                name,
                distribution,
                dependency,
                version,
            )

        spec.on_validated(distribution)
    return module


def with_dependency_specification_context(func, specification: DependencySpecification):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with dependency_specification_context(specification):
            return func(*args, **kwargs)

    return wrapper


def build_requirement_context_factory(
    *spec: Collection[str] | Mapping[str, Collection[str]],
):
    # specification = normalize_dependency_specification(spec)

    def context_factory():
        return contextlib.nullcontext()
        return dependency_specification_context(specification)

    return context_factory
