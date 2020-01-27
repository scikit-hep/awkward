# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import distutils.version

import numpy

# Don't import 'pandas' until an Awkward Array is used in Pandas (when its 'dtype' is first accessed).
checked_version = False
def get_pandas():
    import pandas
    global checked_version
    if not checked_version:
        if distutils.version.LooseVersion(pandas.__version__) < distutils.version.LooseVersion("0.24.0"):
            raise ImportError("cannot use Awkward Array with Pandas version {0} (at least 0.24.0 is required)".format(pandas.__version__))
        checked_version = True
    return pandas

AwkwardDtype = None

def get_dtype():
    import awkward1.highlevel
    pandas = get_pandas()
    global AwkwardDtype
    if AwkwardDtype is None:
        class AwkwardDtype(pandas.api.extensions.ExtensionDtype):
            name = "awkward1"
            type = awkward1.highlevel.Array
            kind = "O"
            base = numpy.dtype("O")
            @classmethod
            def construct_from_string(cls, string):
                if string == cls.name:
                    return cls()
                else:
                    raise TypeError("cannot construct a {0} from {1}".format(cls, string))
    return AwkwardDtype

class PandasNotImportedYet(object):
    pass

class PandasMixin(PandasNotImportedYet):
    @property
    def dtype(self):
        if isinstance(self, PandasNotImportedYet):
            pandas = get_pandas()
            PandasMixin.__bases__ = (pandas.api.extensions.ExtensionArray,)
        return get_dtype()()

    @property
    def nbytes(self):
        return self._layout.nbytes

    def isna(self):
        import awkward1.operations.structure
        return numpy.array(awkward1.operations.structure.isna(self))

    @classmethod
    def _concat_same_type(cls, to_concat):
        import awkward1.operations.structure
        return awkward1.operations.structure.concatenate(to_concat)
