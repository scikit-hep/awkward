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

def extra(defaults, args, kwargs):
    out = []
    for i in range(len(defaults)):
        name, default = defaults[i]
        if i < len(args):
            out.append(args[i])
        elif name in kwargs:
            out.append(kwargs[name])
        else:
            out.append(default)
    return out

class PandasMixin(PandasNotImportedYet):
    # REQUIRED by Pandas:

    @classmethod
    def _from_sequence(scalars, *args, **kwargs):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._from_sequence.html
        # scalars:
        # dtype:
        # copy:
        raise NotImplementedError

    @classmethod
    def _from_factorized(values, original):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._from_factorized.html
        # values:
        # original:
        raise NotImplementedError

    # __getitem__(self)
    # __len__(self)

    @property
    def dtype(self):
        if isinstance(self, PandasNotImportedYet):
            pandas = get_pandas()
            PandasMixin.__bases__ = (pandas.api.extensions.ExtensionArray,)
        return get_dtype()()

    @property
    def nbytes(self):
        return self._layout.nbytes

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return (len(self),)

    def isna(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.isna.html
        import awkward1.operations.structure
        return numpy.array(awkward1.operations.structure.isna(self))

    def take(self, indices, *args, **kwargs):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.take.html
        # indices:
        # allow_fill:
        # fill_value:
        raise NotImplementedError

    def copy(self):
        import awkward1.highlevel
        return awkward1.highlevel.Array(self._layout.deep_copy(copyarrays=True, copyindexes=True, copyidentities=True))

    @classmethod
    def _concat_same_type(cls, to_concat):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._concat_same_type.html
        import awkward1.operations.structure
        return awkward1.operations.structure.concatenate(to_concat)

    # RECOMMENDED for performance:

    # def fillna(self, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.fillna.html
    #     # value:
    #     # method:
    #     # limit:
    #     raise NotImplementedError
    #
    # def dropna(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.dropna.html
    #     raise NotImplementedError
    #
    # def unique(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.unique.html
    #     raise NotImplementedError
    #
    # def factorize(self, na_sentinel):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.factorize.html
    #     # na_sentinel:
    #     raise NotImplementedError
    #
    # def _values_for_factorize(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._values_for_factorize.html
    #     raise NotImplementedError
    #
    # def argsort(self, ascending, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.argsort.html
    #     # ascending: bool, default True
    #     # kind: {"quicksort", "mergesort", "heapsort"}, optional
    #     raise NotImplementedError
    #
    # def _values_for_argsort(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._values_for_argsort.html
    #     raise NotImplementedError
    #
    # def searchsorted(self, value, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.isna.html
    #     # value:
    #     # side:
    #     # sorter:
    #     raise NotImplementedError
    #
    # def _reduce(self, name, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._reduce.html
    #     # name:
    #     # skipna
    #     raise NotImplementedError
