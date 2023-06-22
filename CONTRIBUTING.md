# Contributing to Awkward Array

[![Needs C++ Release](https://github.com/scikit-hep/awkward/actions/workflows/needs-cpp-release.yml/badge.svg)](https://github.com/scikit-hep/awkward/actions/workflows/needs-cpp-release.yml)

Thank you for your interest in contributing! We're eager to see your ideas and look forward to working with you.

This document describes the technical procedures we follow in this project. It should also be stressed that as members of the Scikit-HEP community, we are all obliged to maintaining a welcoming, harassment-free environment. See the [Code of Conduct](https://scikit-hep.org/code-of-conduct) for details.

### Where to start

The front page for the Awkward Array project is its [GitHub README](https://github.com/scikit-hep/awkward#readme). This leads directly to tutorials and reference documentation that you may have already seen. It also includes instructions for [compiling for development](https://github.com/scikit-hep/awkward#installation-for-developers).

### Reporting issues

The first thing you should do if you want to fix something is to [submit an issue through GitHub](https://github.com/scikit-hep/awkward/issues). That way, we can all see it and maybe one of us or a member of the community knows of a solution that could save you the time spent fixing it. If you "assign yourself" to the issue (top of right side-bar), you can signal your intent to fix it in the issue report.

### Contributing a pull request

Feel free to [open pull requests in GitHub](https://github.com/scikit-hep/awkward/pulls) from your [forked repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo) when you start working on the problem. We recommend opening the pull request early so that we can see your progress and communicate about it. (Note that you can `git commit --allow-empty` to make an empty commit and start a pull request before you even have new code.)

Please [make the pull request a draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/) to indicate that it is in an incomplete state and shouldn't be merged until you click "ready for review."

### Getting your pull request reviewed

Currently, we have three regular reviewers of pull requests:

  * Angus Hollands ([agoose77](https://github.com/agoose77))
  * Ioana Ifrim ([ioanaif](https://github.com/ioanaif))
  * Jim Pivarski ([jpivarski](https://github.com/jpivarski))

You can request a review from one of us or just comment in GitHub that you want a review and we'll see it. Only one review is required to be allowed to merge a pull request. We'll work with you to get it into shape.

If you're waiting for a response and haven't heard in a few days, it's possible that we forgot/got distracted/thought someone else was reviewing it/thought we were waiting on you, rather than you waiting on us—just write another comment to remind us.

### Becoming a regular committer

If you want to contribute frequently, we'll grant you write access to the `scikit-hep/awkward` repo itself. This is more convenient than pull requests from forked repos.

### Git practices

Unless you ask us not to, we might commit directly to your pull request as a way of communicating what needs to be changed. That said, most of the commits on a pull request are from a single author: corrections and suggestions are exceptions.

Therefore, we prefer git branches to be named with your GitHub userid, such as `jpivarski/write-contributing-md`.

The titles of pull requests (and therefore the merge commit messages) should follow [these conventions](https://www.conventionalcommits.org/en/v1.0.0/#summary). Mostly, this means prefixing the title with one of these words and a colon:

  * feat: new feature
  * fix: bug-fix
  * perf: code change that improves performance
  * refactor: code change that neither fixes a bug nor adds a feature
  * style: changes that do not affect the meaning of the code
  * test: adding missing tests or correcting existing tests
  * build: changes that affect the build system or external dependencies
  * docs: documentation only changes
  * ci: changes to our CI configuration files and scripts
  * chore: other changes that don't modify src or test files
  * revert: reverts a previous commit

Almost all pull requests are merged with the "squash and merge" feature, so details about commit history within a pull request are hidden from the `main` branch's history. Feel free, therefore, to commit with any frequency you're comfortable with.

It is unnecessary to manually edit (rebase) commit history within a pull request.

### Building and testing locally

The [installation for developers](README.md#installation-for-developers) procedure is described in brief on the front page, and in more detail here.

Awkward Array is shipped as two packages: `awkward` and `awkward-cpp`. The `awkward-cpp` package contains the compiled C++ components required for performance, and `awkward` is only Python code. If you do not need to modify any C++ (the usual case), then `awkward-cpp` can simply be installed using `pip` or `conda`.

Subsequent steps require the generation of code and datafiles (kernel specification, header-only includes). This can be done with the `prepare` [nox](https://nox.thea.codes/) session:

```bash
nox -s prepare
```

<details>

The `prepare` session accepts flags to specify exact generation targets, e.g.
```bash
nox -s prepare -- --tests --docs
```
This can reduce the time taken to perform the preparation step in the event that only the package-building step is needed.

`nox` also lets us re-use the virtualenvs that it creates for each session with the `-R` flag, eliminating the dependency reinstall time:
```bash
nox -R -s prepare
```

</details>

#### Installing the `awkward-cpp` package

The C++ components can be installed by building the `awkward-cpp` package:

```bash
python -m pip install ./awkward-cpp
```

<details>

If you are working on the C++ components of Awkward Array, it might be more convenient to skip the build isolation step, which involves creating an isolated build environment. First, you must install the build requirements:

```bash
python -m pip install "scikit-build-core[pyproject,color]" pybind11 ninja cmake
```

Then the installation can be performed without build isolation:

```bash
python -m pip install --no-build-isolation --check-build-dependencies ./awkward-cpp
```

 </details>

#### Installing the `awkward` package

With `awkward-cpp` installed, an editable installation of the pure-python `awkward` package can be performed with

```bash
python -m pip install -e .
```

#### Testing the installed packages

Finally, let's run the integration test suite to ensure that everything's working as expected:

```bash
python -m pytest -n auto tests
```

For more fine-grained testing, we also have tests of the low-level kernels, which can be invoked with

```bash
python -m pytest -n auto awkward-cpp/tests-spec
python -m pytest -n auto awkward-cpp/tests-cpu-kernels
```

This assumes that the `nox -s prepare` session ran the `--tests` target.

Furthermore, if you have an Nvidia GPU and CuPy installed, you can run the CUDA tests with

```bash
python -m pytest tests-cuda-kernels
python -m pytest tests-cuda
```

### Building wheels

Sometimes it's convenient to build a wheel for the `awkward-cpp` package, so that subsequent re-installs do not require the package to be rebuilt. The `build` package can be used to do this, though care must be taken to specify the *current* Python interpreter in [pipx](https://pypa.github.io/pipx/):

```bash
pipx run --python=$(which python) build --wheel awkward-cpp
```

The built wheel will then be available in `awkward-cpp/dist`.

### Automatic formatting and linting

The Awkward Array project uses [pre-commit](https://pre-commit.com) to handle formatters and linters. This automatically checks (and may push corrections to) your pull request's git branch.

To respond more quickly to pre-commit's feedback, it can help to install it and run it locally. Once it is installed, run

```bash
pre-commit run -a
```

to test all of your files. If you leave off the `-a`, it will run only on currently stashed changes.

### Automated tests

As stated above, we use [pytest](https://docs.pytest.org/) to verify the correctness of the code, and GitHub will reject a pull request if either pre-commit or pytest fails (red "X"). All tests must pass for a pull request to be accepted.

Note that if a pull request doesn't modify code, only the documentation tests will run. That's okay: documentation-only pull requests only need the documentation tests to pass.

### Testing practices

Unless you're refactoring code, such that your changes are fully tested by the existing test suite, new code should be accompanied by new tests. Our testing suite is organized by GitHub issue or pull request number: that is, test file names are

```
tests/test_XXXX-yyyy.py
```

where `XXXX` is either the number of the issue your pull request fixes or the number of the pull request and `yyyy` is descriptive text, often the same as the git branch. This makes it easier to run your test in isolation:

```bash
python -m pytest tests/test_XXXX-yyyy.py
```

and it makes it easier to figure out why a particular test was added. The easiest way to make a new testing file is to copy an existing one and replace its `test_zzzz` functions with your own. The previous tests should also give you a sense of the way we test things and the kinds of things that are constrained in tests.

### Building documentation locally

Documentation is automatically built by each pull request. You usually won't need to build the documentation locally, but if you do, this section describes how.

We use [Sphinx](https://pypi.org/project/Sphinx/) to generate documentation. You may need to install some additional packages:

  * [Doxygen](https://www.doxygen.nl/download.html)
  * [pycparser](https://pypi.org/project/pycparser/)
  * [black](https://pypi.org/project/black/)
  * [sphinx](https://pypi.org/project/sphinx/)
  * [sphinx-rtd-theme](https://pypi.org/project/sphinx-rtd-theme/)

To build documentation locally, first prepare the generated data files with

```bash
nox -s prepare
```

<details>

Only the `--headers` and `--docs` flags are actually required at the time of writing. These can be passed with:

```bash
nox -s prepare -- --docs --headers
```

 </details>

Then, use `nox` to run the various documentation build steps

```bash
nox -s docs
```

This command executes multiple custom Python scripts (some require a working internet connection), in addition to using Sphinx and Doxygen to generate the required browser viewable documentation.

To view the built documentation, open

```bash
docs/_build/html/index.html
```

from the root directory of the project in your preferred web browser, e.g.

```bash
python -m http.server 8080 --directory docs/_build/html/
```

Before re-building documentation, you might want to delete the files that were generated to create viewable documentation. A simple command to remove all of them is

```bash
rm -rf docs/reference/generated docs/_build docs/_static/doxygen
```

There is also a cache in the `docs/_build/.jupyter_cache` directory for Jupyter Book, which can be removed.

### The main branch

The Awkward Array `main` branch must be kept in an unbroken state. There are two reasons for this: so that developers can work independently on known-to-be-working states and so that users can test the latest changes (usually to see if the bug they've discovered is fixed by a potential correction).

The `main` branch is also never far from the latest released version. We usually deploy patch releases (`z` in a version number like `x.y.z`) within days of a bug-fix.

Committing directly to `main` is not allowed except for

   * updating the `pyproject.toml` file to increase the version number, which should be independent of pull requests
   * updating documentation or non-code files
   * unprecedented emergencies

and only by the the [reviewing team](CONTRIBUTING.md#getting-your-pull-request-reviewed).

### The main-v1 branch

The `main-v1` branch was split from `main` just before Awkward 1.x code was removed, so it exists to make 1.10.x bug-fix releases. These commits must be drawn from `main-v1`, not `main`, and pull requests must target `main-v1` (not the GitHub default). A single commit cannot be applied to both `main` and `main-v1` because they have diverged too much. If a bug-fix needs to be applied to both (unlikely), it will have to be reimplemented on both.

### Releases

Currently, only one person can deploy releases:

  * Jim Pivarski ([jpivarski](https://github.com/jpivarski))

There are two kinds of releases: (1) `awkward-cpp` updates, which only occur when the C++ is updated (rare) and involves compilation on many platforms (takes hours), and (2) `awkward` updates, which can happen with any bug-fix. The [releases listed in GitHub](https://github.com/scikit-hep/awkward/releases) are `awkward` releases, not `awkward-cpp`.

If you need your merged pull request to be deployed in a release, just ask!

#### `awkward-cpp` releases
To make an `awkward-cpp` release:
1. A commit to `main` should increase the version number in `awkward-cpp/pyproject.toml`
2. The [Deploy C++](https://github.com/scikit-hep/awkward/actions/workflows/deploy-cpp.yml) GitHub Actions workflow should be manually triggered.
3. A `git` tag `awkward-cpp-{version}` should be created for the new version epoch.

#### `awkward` releases
To make an `awkward` release:
1. A commit to `main` should increase the version number in `pyproject.toml`
2. A new GitHub release must be published.
3. A `docs/switcher.json` entry must be added for new minor/major versions.

Pushes that modify `docs/switcher.json` on `main` will automatically be synchronised with AWS.
