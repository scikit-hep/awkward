# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Sphinx extension for the Awkward Array documentation.

The modules other than ``_sphinx`` do not import Sphinx or sphinx-autoapi, so
that their unit tests run without either. ``_sphinx`` connects them to Sphinx
and sphinx-autoapi and is tested by the documentation build.
"""


def setup(app):
    from awkward_docs import _sphinx

    return _sphinx.setup(app)


def autoapi_prepare_jinja_env(jinja_env):
    """Register the filters and globals that the autoapi templates use."""
    from awkward_docs import _sphinx

    _sphinx.prepare_jinja_env(jinja_env)
