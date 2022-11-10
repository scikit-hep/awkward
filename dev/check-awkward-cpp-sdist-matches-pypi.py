import argparse
import hashlib
import json
import pathlib
import urllib.error
import urllib.request

import toml

THIS_FILE = pathlib.Path(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sdist", type=pathlib.Path)
    args = parser.parse_args()

    awkward_cpp_path = THIS_FILE.parents[1] / "awkward-cpp"

    with open(awkward_cpp_path / "pyproject.toml") as f:
        metadata = toml.load(f)

    version = metadata["project"]["version"]

    # Load version information from PyPI
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/awkward-cpp/{version}/json"
        ) as response:
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
            f"hash of awkward-cpp {version} on PyPI does not match given sdist hash"
        )


if __name__ == "__main__":
    main()
