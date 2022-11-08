import hashlib
import json
import os
import pathlib
import tempfile
import urllib.error
import urllib.request

import build
import build.env
import toml

THIS_FILE = pathlib.Path(__file__)


def main():
    # We need SOURCE_DATE_EPOCH to best set such that the sdist is identical for builds of the
    # same Git SHA
    if "SOURCE_DATE_EPOCH" not in os.environ:
        raise RuntimeError(
            "SOURCE_DATE_EPOCH must be set to ensure reproducible builds"
        )

    awkward_cpp_path = THIS_FILE.parents[1] / "awkward-cpp"

    with open(awkward_cpp_path / "pyproject.toml") as f:
        metadata = toml.load(f)

    version = metadata["project"]["version"]

    # First, are we working on an unreleased version?
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/awkward-cpp/{version}/json"
        ) as response:
            data = json.load(response)
    except urllib.error.HTTPError as err:
        assert err.status == 404, err.status
        return

    # Get SHA256 hashes of published sdist(s)
    sdist_shas = {
        entry["digests"]["sha256"]
        for entry in data["urls"]
        if entry["packagetype"] == "sdist"
    }

    # Build sdist
    builder = build.ProjectBuilder(str(awkward_cpp_path))
    with build.env.IsolatedEnvBuilder() as env, tempfile.TemporaryDirectory() as path:
        builder.python_executable = env.executable
        builder.scripts_dir = env.scripts_dir
        # First install the build dependencies
        env.install(builder.build_system_requires)
        # Then get the extra required dependencies from the backend
        env.install(builder.get_requires_for_build("sdist"))
        # Build an sdist
        output_path = pathlib.Path(builder.build("sdist", path, {}))
        # Compute sdist SHA256
        sha = hashlib.sha256(output_path.read_bytes())

    if sha not in sdist_shas:
        raise RuntimeError(
            "SHA256 of current sdist differs from release published with same version"
        )


if __name__ == "__main__":
    main()
