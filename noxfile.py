import shutil

import nox

ALL_PYTHONS = ["3.7", "3.8", "3.9", "3.10", "3.11"]

nox.options.sessions = ["lint", "tests"]


@nox.session(python=ALL_PYTHONS)
def tests(session):
    """
    Run the unit and regular tests.
    """
    session.install("-r", "requirements-test.txt")
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
    session.install("-r", "requirements-test.txt")
    session.run(
        "pytest", "tests", "--cov=awkward", "--cov-report=xml", *session.posargs
    )


@nox.session
def docs(session):
    """
    Build the docs. Pass "serve" to serve.
    """
    # Generate C++ documentation
    with session.chdir("awkward-cpp/docs"):
        session.run("doxygen")

    # Copy generated C++ docs to Sphinx
    shutil.copytree("awkward-cpp/docs/html", "docs/_static/doxygen", dirs_exist_ok=True)

    # Generate kernel documentation
    session.install("pyyaml")
    session.run("python", "dev/generate-kernel-docs.py")

    # Build Sphinx docs
    with session.chdir("docs"):
        session.install("-r", "requirements.txt")
        session.run("sphinx-build", "-M", "html", ".", "_build")

        if session.posargs:
            if "--serve" in session.posargs:
                session.log(
                    "Launching docs at http://localhost:8000/ - use Ctrl-C to quit"
                )
                session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
            else:
                session.error("Unsupported argument to docs")


@nox.session
def clean(session):
    """
    Clean generated artifacts.
    """
    session.run("python", "dev/clean-cpp-headers.py")
    session.run("python", "dev/clean-kernel-signatures.py")
    session.run("python", "dev/clean-kernel-docs.py")
    session.run("python", "dev/clean-tests.py")


@nox.session
def prepare(session):
    """
    Prepare for package building.
    """
    session.install("-r", "requirements-dev.txt")
    if not session.posargs:
        flags = {"headers", "signatures", "tests", "docs"}
    else:
        flags = set(session.posargs)

    if "headers" in flags:
        session.run("python", "dev/copy-cpp-headers.py")
    if "signatures" in flags:
        session.run("python", "dev/generate-kernel-signatures.py")
    if "tests" in flags:
        session.run("python", "dev/generate-tests.py")
    if "docs" in flags:
        session.run("python", "dev/generate-kernel-docs.py")


@nox.session
def check_version(session):
    """
    Check that the awkward-cpp version is compatible with awkward
    and that we are not modifying a released awkward-cpp version
    """
    session.install("-r", "requirements-dev.txt")
    session.run("python", "dev/check-awkward-cpp-unchanged.py")
    session.run("python", "dev/check-awkward-uses-awkward-cpp.py")


@nox.session
def diagnostics(session):
    """
    Check that the CPU kernels are defined correctly
    """
    session.install("-r", "requirements-dev.txt")
    session.run("python", "dev/kernel-diagnostics.py", *session.posargs)
