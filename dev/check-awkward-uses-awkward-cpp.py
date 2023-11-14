# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pathlib

import packaging.requirements
import packaging.utils
import toml

THIS_FILE = pathlib.Path(__file__)


def main():
    # Get the dependencies of `awkward`
    awkward_path = THIS_FILE.parents[1]
    with open(awkward_path / "pyproject.toml") as f:
        awkward_metadata = toml.load(f)

    # Get the dependencies of `awkward-cpp`
    awkward_cpp_path = THIS_FILE.parents[1] / "awkward-cpp"
    with open(awkward_cpp_path / "pyproject.toml") as f:
        awkward_cpp_metadata = toml.load(f)

    # Find the awkward-cpp requirement in awkward's dependencies
    awkward_requirements = [
        packaging.requirements.Requirement(r)
        for r in awkward_metadata["project"]["dependencies"]
    ]
    try:
        awkward_cpp_requirement = next(
            r
            for r in awkward_requirements
            if packaging.utils.canonicalize_name(r.name) == "awkward-cpp"
        )
    except StopIteration:
        raise RuntimeError(
            "could not find awkward-cpp requirement in awkward dependencies"
        ) from None

    # Check whether awkward-cpp version is currently compatible
    awkward_cpp_version = awkward_cpp_metadata["project"]["version"]
    if awkward_cpp_version not in awkward_cpp_requirement.specifier:
        raise RuntimeError(
            "awkward-cpp package version is not compatible with the requirement specified in awkward"
        )


if __name__ == "__main__":
    main()
