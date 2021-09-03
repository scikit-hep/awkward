# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import


def headtail(oldtail):
    if len(oldtail) == 0:
        return (), ()
    else:
        return oldtail[0], oldtail[1:]
