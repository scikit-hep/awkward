# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import email.parser
import os
import pathlib
import runpy
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = pytest.importorskip("tomli")

# Packaging-only dependencies are installed by `nox -s versioning`.
pytest.importorskip("build")
vcs = pytest.importorskip("hatch_vcs.version_source")

ROOT = pathlib.Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def version(root):
    with (root / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)["tool"]["hatch"]["version"]
    return vcs.VCSVersionSource(str(root), config).get_version_data()["version"]


@pytest.fixture
def repository(tmp_path, monkeypatch):
    # Use the real build configuration, but a tiny package: no compiled kernels
    # or runtime dependencies are needed to exercise versioning and packaging.
    for name in tuple(os.environ):
        if name.startswith(("SETUPTOOLS_SCM_", "GIT_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    root = tmp_path / "repository"
    root.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "juliapkg.json"):
        shutil.copyfile(ROOT / name, root / name)
    package = root / "src" / "awkward"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from ._version import __version__\n")
    (root / ".gitignore").write_text("dist/\nsrc/awkward/_version.py\n")
    git(root, "init")
    git(root, "config", "user.name", "Versioning test")
    git(root, "config", "user.email", "versioning@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    return root


def build(root, output, *options):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output),
            *options,
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


@pytest.mark.parametrize("tag", ["v2.14.0", "v2.15.0rc1"])
@pytest.mark.parametrize("annotated", [False, True])
def test_release_tags(repository, tag, annotated):
    args = ("-a", "-m", "release") if annotated else ()
    git(repository, "tag", *args, tag)
    assert version(repository) == tag[1:]


def test_development_and_cpp_tags(repository):
    git(repository, "tag", "v2.14.0")
    git(repository, "commit", "--allow-empty", "-m", "development")
    expected = version(repository)
    assert expected.startswith("2.14.1.dev1+g")
    git(repository, "tag", "-a", "awkward-cpp-999", "-m", "C++ release")
    git(repository, "tag", "999.0.0")
    assert version(repository) == expected
    git(repository, "commit", "--allow-empty", "-m", "more development")
    assert version(repository).startswith("2.14.1.dev2+g")


def test_prerelease_development(repository):
    git(repository, "tag", "v2.15.0rc1")
    git(repository, "commit", "--allow-empty", "-m", "development")
    assert version(repository).startswith("2.15.0rc2.dev1+g")


def test_fetching_history_restores_development_version(repository, tmp_path):
    git(repository, "tag", "v2.14.0")
    git(repository, "commit", "--allow-empty", "-m", "development")
    clone = tmp_path / "shallow"
    git(tmp_path, "clone", "--depth", "1", repository.as_uri(), str(clone))
    assert git(clone, "rev-parse", "--is-shallow-repository") == "true"
    with pytest.raises(ValueError, match="shallow"):
        version(clone)
    git(clone, "fetch", "--unshallow", "--tags")
    assert version(clone) == version(repository)


@pytest.mark.parametrize("tag", ["v2.14.0", "v2.15.0rc1", None])
def test_sdist_wheel_version_without_git(repository, tmp_path, tag):
    git(repository, "tag", tag or "v2.14.0")
    if tag is None:
        git(repository, "commit", "--allow-empty", "-m", "development")
    expected = version(repository)
    output = tmp_path / "dist"
    build(repository, output, "--sdist")
    archive = next(output.glob("*.tar.gz"))
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as source:
        # This archive was produced locally from the fixture's source files.
        source.extractall(extracted, filter="data")
    unpacked = next(extracted.iterdir())
    assert not (unpacked / ".git").exists()
    metadata = email.parser.Parser().parsestr((unpacked / "PKG-INFO").read_text())
    assert metadata["Version"] == expected
    # Move the original checkout on; the sdist must keep its recorded version.
    git(repository, "commit", "--allow-empty", "-m", "later development")
    build(unpacked, output, "--wheel")
    wheel = next(output.glob("*.whl"))
    with zipfile.ZipFile(wheel) as source:
        metadata_name = next(
            n for n in source.namelist() if n.endswith(".dist-info/METADATA")
        )
        metadata = email.parser.Parser().parsestr(source.read(metadata_name).decode())
        assert metadata["Version"] == expected
        version_file = tmp_path / "_version.py"
        version_file.write_bytes(source.read("awkward/_version.py"))
    assert runpy.run_path(str(version_file))["__version__"] == expected
