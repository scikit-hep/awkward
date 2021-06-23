# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Form(object):
    @property
    def has_identities(self):
        return self._has_identities

    @property
    def parameters(self):
        return self._parameters

    @property
    def form_key(self):
        return self._form_key

    def parameter(self, key):
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    def __str__(self):
        return json.dumps(self.tolist(verbose=False), indent="    ")

    def tolist(self, verbose=True, toplevel=False):
        if toplevel:
            return self._tolist_part(verbose=verbose, toplevel=toplevel)
        out = self._tolist_part(verbose=verbose)
        if verbose or self._has_identities:
            out["has_identities"] = self._has_identities
        if verbose or self._parameters is not None and len(self._parameters) > 0:
            out["parameters"] = {} if self._parameters is None else self._parameters
        if verbose or self._form_key is not None:
            out["form_key"] = self._form_key
        return out

    def to_json(self):
        return json.dumps(self.tolist())

    def _repr_args(self):
        out = []
        if self._has_identities is not False:
            out.append("has_identities=" + repr(self._has_identities))

        if self._parameters is not None and len(self._parameters) > 0:
            out.append("parameters=" + repr(self._parameters))

        if self._form_key is not None:
            out.append("form_key=" + repr(self._form_key))
        return out
