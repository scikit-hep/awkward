import argparse
import os
import pathlib
import shutil

import nox

ALL_PYTHONS = ["3.7", "3.8", "3.9", "3.10", "3.11"]

nox.options.sessions = ["lint", "tests"]


def remove_if_found(*paths):
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.unlink(path)


@nox.session(python=ALL_PYTHONS)
def tests(session):
    """
    Run the unit and regular tests.
    """
    session.install("-r", "requirements-test.txt", "./awkward-cpp", "./awkward")
    session.run("pytest", *session.posargs if session.posargs else ["tests"])


@nox.session
def lint(session):
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def pylint(session):
    """
    Run the pylint process.
    """

    session.install("pylint==2.12.2")
    session.run("pylint", "src", *session.posargs)


@nox.session(python=ALL_PYTHONS)
def coverage(session):
    """
    Run the unit and regular tests.
    """
    session.install("-r", "requirements-test.txt", "./awkward-cpp", "./awkward")
    session.run(
        "pytest", "tests", "--cov=awkward", "--cov-report=xml", *session.posargs
    )


@nox.session
def docs(session):
    """
    Build the docs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args(session.posargs)

    session.install("-r", "requirements-dev.txt")

    # Generate C++ documentation
    with session.chdir("awkward-cpp/docs"):
        session.run("doxygen")

    # Copy generated C++ docs to Sphinx
    shutil.copytree("awkward-cpp/docs/html", "docs/_static/doxygen", dirs_exist_ok=True)

    # Generate kernel documentation
    session.run("python", "dev/generate-kernel-docs.py")

    # Build Sphinx docs
    with session.chdir("docs"):
        session.install("-r", "requirements.txt")
        session.run("sphinx-build", "-M", "html", ".", "_build")

        if args.serve:
            session.log("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8000", "-d", "_build/html")


@nox.session
def clean(session):
    """
    Clean generated artifacts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--headers", action="store_true")
    parser.add_argument("--signatures", action="store_true")
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--docs", action="store_true")
    args = parser.parse_args(session.posargs)

    clean_all = not session.posargs

    if args.headers or clean_all:
        remove_if_found(pathlib.Path("awkward-cpp", "header-only"))
    if args.signatures or clean_all:
        remove_if_found(
            pathlib.Path("awkward-cpp", "include", "awkward", "kernels.h"),
            pathlib.Path("awkward-cpp", "src", "awkward_cpp", "_kernel_signatures.py"),
            pathlib.Path("src", "awkward", "_connect", "cuda", "_kernel_signatures.py"),
        )
    if args.tests or clean_all:
        remove_if_found(
            pathlib.Path("awkward-cpp", "tests-spec"),
            pathlib.Path("awkward-cpp", "tests-spec-explicit"),
            pathlib.Path("awkward-cpp", "tests-cpu-kernels"),
            pathlib.Path("tests-cuda-kernels"),
        )
    if args.docs or clean_all:
        remove_if_found(pathlib.Path("docs", "reference", "generated", "kernels.rst"))


@nox.session
def prepare(session):
    """
    Prepare for package building.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--headers", action="store_true")
    parser.add_argument("--signatures", action="store_true")
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--docs", action="store_true")
    args = parser.parse_args(session.posargs)

    session.install("-r", "requirements-dev.txt")

    prepare_all = not session.posargs

    if args.headers or prepare_all:
        session.run("python", "dev/copy-cpp-headers.py")
    if args.signatures or prepare_all:
        session.run("python", "dev/generate-kernel-signatures.py")
    if args.tests or prepare_all:
        session.run("python", "dev/generate-tests.py")
    if args.docs or prepare_all:
        session.run("python", "dev/generate-kernel-docs.py")


@nox.session
def check_cpp_constraint(session):
    """
    Check that the awkward-cpp version is compatible with awkward.
    """
    session.install("-r", "requirements-dev.txt")
    session.run("python", "dev/check-awkward-uses-awkward-cpp.py")


@nox.session
def check_cpp_sdist_released(session):
    """
    Check that the awkward-cpp sdist is already on PyPI
    """
    session.install("-r", "requirements-dev.txt")
    session.run(
        "python", "dev/check-awkward-cpp-sdist-matches-pypi.py", *session.posargs
    )


@nox.session
def diagnostics(session):
    """
    Check that the CPU kernels are defined correctly.
    """
    session.install("-r", "requirements-dev.txt")
    session.run("python", "dev/kernel-diagnostics.py", *session.posargs)
