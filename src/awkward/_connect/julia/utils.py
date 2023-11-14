# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path

from julia.api import JuliaError

from awkward import __version__

# FIXME: move this out?
__awkward_array_jl_version__ = "1.0.0-DEV"

juliainfo = None
julia_initialized = False
julia_kwargs_at_initialization = None
julia_activated_env = None


@lru_cache
def _load_juliainfo():
    """Execute julia.core.JuliaInfo.load(), and store as juliainfo."""
    global juliainfo

    if juliainfo is None:
        from julia.core import JuliaInfo

        try:
            juliainfo = JuliaInfo.load(julia="julia")
        except FileNotFoundError:
            env_path = os.environ["PATH"]
            raise FileNotFoundError(
                f"Julia is not installed in your PATH. Please install Julia and add it to your PATH.\n\nCurrent PATH: {env_path}",
            ) from None

    return juliainfo


def _julia_version() -> tuple[int, ...]:
    """Check if Julia version is greater than specified version."""
    juliainfo = _load_juliainfo()
    return (
        juliainfo.version_major,
        juliainfo.version_minor,
        juliainfo.version_patch,
    )


def _check_for_conflicting_libraries():  # pragma: no cover
    """Check whether there are conflicting modules, and display warnings."""


def _set_julia_project_env(julia_project, is_shared):
    if is_shared:
        os.environ["JULIA_PROJECT"] = "@" + str(julia_project)
    else:
        os.environ["JULIA_PROJECT"] = str(julia_project)


def _process_julia_project(julia_project):
    if julia_project is None:
        is_shared = True
        processed_julia_project = f"awkward-{__version__}"
    elif julia_project[0] == "@":
        is_shared = True
        processed_julia_project = julia_project[1:]
    else:
        is_shared = False
        processed_julia_project = Path(julia_project)
    return processed_julia_project, is_shared


def _import_error():
    return """
    Required dependencies are not installed or built.  Run the following command in your terminal:
        python3 -m awkward install
    """


def _get_io_arg(quiet):
    io = "devnull" if quiet else "stderr"
    return f"io={io}"


def init_julia(julia_project=None, quiet=False, julia_kwargs=None, return_aux=False):
    """Initialize julia binary, turning off compiled modules if needed."""
    global julia_initialized
    global julia_kwargs_at_initialization
    global julia_activated_env

    if not julia_initialized:
        _check_for_conflicting_libraries()

    if julia_kwargs is None:
        julia_kwargs = {"optimize": 3}

    from julia.core import JuliaInfo, UnsupportedPythonError

    if _julia_version() < (1, 9, 0):
        raise NotImplementedError(
            "AwkwardArray requires Julia 1.9.0 or greater. "
            "Please update your Julia installation."
        )

    processed_julia_project, is_shared = _process_julia_project(julia_project)
    _set_julia_project_env(processed_julia_project, is_shared)

    try:
        info = JuliaInfo.load(julia="julia")
    except FileNotFoundError:
        env_path = os.environ["PATH"]
        raise FileNotFoundError(
            f"Julia is not installed in your PATH. Please install Julia and add it to your PATH.\n\nCurrent PATH: {env_path}",
        ) from None

    if not info.is_pycall_built():
        raise ImportError(_import_error())

    from julia.core import Julia

    try:
        Julia(**julia_kwargs)
    except UnsupportedPythonError:
        # Static python binary, so we turn off pre-compiled modules.
        julia_kwargs = {**julia_kwargs, "compiled_modules": False}
        Julia(**julia_kwargs)
        warnings.warn(
            "Your system's Python library is static (e.g., conda), so precompilation will be turned off. For a dynamic library, try using `pyenv` and installing with `--enable-shared`: https://github.com/pyenv/pyenv/blob/master/plugins/python-build/README.md#building-with---enable-shared.",
            stacklevel=2,
        )

    using_compiled_modules = ("compiled_modules" not in julia_kwargs) or julia_kwargs[
        "compiled_modules"
    ]

    from julia import Main as julia_main

    if julia_activated_env is None:
        julia_activated_env = processed_julia_project

    if julia_initialized and julia_kwargs_at_initialization is not None:
        # Check if the kwargs are the same as the previous initialization
        init_set = set(julia_kwargs_at_initialization.items())
        new_set = set(julia_kwargs.items())
        set_diff = new_set - init_set
        # Remove the `compiled_modules` key, since it is not a user-specified kwarg:
        set_diff = {k: v for k, v in set_diff if k != "compiled_modules"}
        if len(set_diff) > 0:
            warnings.warn(
                "Julia has already started. The new Julia options "
                + str(set_diff)
                + " will be ignored.",
                stacklevel=2,
            )

    if julia_initialized and julia_activated_env != processed_julia_project:
        julia_main.eval("using Pkg")

        io_arg = _get_io_arg(quiet)
        # Can't pass IO to Julia call as it evaluates to PyObject, so just directly
        # use Main.eval:
        julia_main.eval(
            f'Pkg.activate("{_escape_filename(processed_julia_project)}",'
            f"shared = Bool({int(is_shared)}), "
            f"{io_arg})"
        )

        julia_activated_env = processed_julia_project

    if not julia_initialized:
        julia_kwargs_at_initialization = julia_kwargs

    julia_initialized = True
    if return_aux:
        return julia_main, {"compiled_modules": using_compiled_modules}
    return julia_main


def _add_awkward_to_julia_project(julia_main, io_arg):
    julia_main.eval("using Pkg")
    julia_main.eval("Pkg.Registry.update()")
    julia_main.awkward_spec = julia_main.PackageSpec(
        name="AwkwardArray",
        url="https://github.com/jpivarski/AwkwardArray.jl",
        rev="v" + __awkward_array_jl_version__,
    )
    julia_main.eval(f"Pkg.add([awkward_spec], {io_arg})")


def _escape_filename(filename):
    """Turn a path into a string with correctly escaped backslashes."""
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr


def _backend_version_assertion(julia_main):
    try:
        backend_version = julia_main.eval("string(pkgversion(AwkwardArray))")
        expected_backend_version = __awkward_array_jl_version__
        if backend_version != expected_backend_version:  # pragma: no cover
            warnings.warn(
                f"AwkwardArray backend (AwkwardArray.jl) version {backend_version} "
                f"does not match expected version {expected_backend_version}. "
                "Things may break. "
                "Please update your AwkwardArray installation with "
                "`python3 -m AwkwardArray install`.",
                stacklevel=2,
            )
    except JuliaError:  # pragma: no cover
        warnings.warn(
            "You seem to have an outdated version of AwkwardArray.jl. "
            "Things may break. "
            "Please update your AwkwardArray installation with "
            "`python3 -m AwkwardArray install`.",
            stacklevel=2,
        )


def _update_julia_project(julia_main, is_shared, io_arg):
    try:
        if is_shared:
            _add_awkward_to_julia_project(julia_main, io_arg)
        julia_main.eval("using Pkg")
        julia_main.eval(f"Pkg.resolve({io_arg})")
    except (JuliaError, RuntimeError) as e:
        raise ImportError(_import_error()) from e


def _load_backend(julia_main):
    try:
        # Load namespace, so that various internal operators work:
        julia_main.eval("using AwkwardArray")
    except (JuliaError, RuntimeError) as e:
        raise ImportError(_import_error()) from e

    _backend_version_assertion(julia_main)

    # Load Julia package AwkwardArray.jl
    from julia import AwkwardArray

    return AwkwardArray
