# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.error
import urllib.parse
import urllib.request

import toml

THIS_FILE = pathlib.Path(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sdist", type=pathlib.Path)
    parser.add_argument("--endpoint", default="https://pypi.org/pypi/")
    args = parser.parse_args()

    if not args.endpoint.endswith("/"):
        raise ValueError(
            "provided endpoint URL will lose the trailing component when "
            "joined with the relative version URL"
        )

    awkward_cpp_path = THIS_FILE.parents[1] / "awkward-cpp"

    with open(awkward_cpp_path / "pyproject.toml") as f:
        metadata = toml.load(f)
    project = metadata["project"]

    # Load version information from PyPI
    version_url = urllib.parse.urljoin(
        args.endpoint, "{name}/{version}/json".format_map(project)
    )
    try:
        with urllib.request.urlopen(version_url) as response:
            data = json.load(response)
    except urllib.error.HTTPError as err:
        raise SystemExit(err.status) from err

    # Get SHA256 hashes of published sdist(s)
    sdist_shas = {
        entry["digests"]["sha256"]
        for entry in data["urls"]
        if entry["packagetype"] == "sdist"
    }

    assert len(sdist_shas) == 1, "Should find exactly one sdist for a given release"

    # Check hashes match
    if sdist_shas.pop() != hashlib.sha256(args.sdist.read_bytes()).hexdigest():
        raise ValueError(
            "hash of awkward-cpp {version} on PyPI does not match given sdist hash".format_map(
                project
            )
        )


if __name__ == "__main__":
    main()
