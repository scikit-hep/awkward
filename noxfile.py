import nox

ALL_PYTHONS = ["3.7", "3.8", "3.9", "3.10", "3.11"]

nox.options.sessions = ["lint", "tests"]


@nox.session(python=ALL_PYTHONS)
def tests(session):
    """
    Run the unit and regular tests.
    """
    session.install(".[test]", "numba", "pandas", "pyarrow", "jax", "numexpr", "uproot")
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

    session.install(".")
    session.install("pylint==2.12.2")
    session.run("pylint", "src", *session.posargs)


@nox.session(python=ALL_PYTHONS)
def coverage(session):
    """
    Run the unit and regular tests.
    """
    session.install(".[test]", "pytest-cov")
    session.run(
        "pytest", "tests", "--cov=awkward", "--cov-report=xml", *session.posargs
    )


@nox.session
def docs(session):
    """
    Build the docs. Pass "serve" to serve.
    """

    session.chdir("docs-sphinx")
    session.install("-r", "requirements.txt")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            session.log("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
        else:
            session.error("Unsupported argument to docs")
