# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numbers
import re

import numpy

def field2index(lookup, numfields, key):
    if isinstance(key, (int, numbers.Integral, numpy.integer)):
        attempt = key
    else:
        attempt = None if lookup is None else lookup.get(key)

    if attempt is None:
        m = field2index._pattern.match(key)
        if m is not None:
            attempt = m.group(0)

    if attempt is None or attempt >= numfields:
        raise ValueError("key {0} not found in Record".format(repr(key)))
    else:
        return attempt

field2index._pattern = re.compile(r"^[1-9][0-9]*$")
