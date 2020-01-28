# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import distutils.version

import numpy

import awkward1.layout
import awkward1._util
import awkward1.operations.convert
import awkward1.operations.structure

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
        @pandas.api.extensions.register_extension_dtype
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

            @classmethod
            def construct_array_type(cls):
                return awkward1.highlevel.Array

    return AwkwardDtype

class PandasNotImportedYet(object):
    pass

def extra(args, kwargs, defaults):
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
    def _from_sequence(cls, scalars, *args, **kwargs):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._from_sequence.html
        dtype, copy = extra(args, kwargs, [
            ("dtype", None),
            ("copy", False)])
        return awkward1._util.wrap(awkward1.operations.convert.fromiter(scalars), None, None)

    @classmethod
    def _from_factorized(cls, values, original):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._from_factorized.html
        raise NotImplementedError("_from_factorized")

    # __getitem__(self)
    # __len__(self)

    @property
    def dtype(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.dtype.html
        if isinstance(self, PandasNotImportedYet):
            pandas = get_pandas()
            PandasMixin.__bases__ = (pandas.api.extensions.ExtensionArray,)
        return get_dtype()()

    @property
    def nbytes(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.nbytes.html
        return self._layout.nbytes

    @property
    def ndim(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.nbytes.html
        return 1

    @property
    def shape(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.shape.html
        return (len(self),)

    def isna(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.isna.html
        return numpy.array(awkward1.operations.structure.isna(self))

    def take(self, indices, *args, **kwargs):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.take.html
        allow_fill, fill_value = extra(args, kwargs, [
            ("allow_fill", False),
            ("fill_value", None)])

        if allow_fill:
            indices = numpy.asarray(indices, dtype=numpy.int64)
            if fill_value is None:
                index = awkward1.layout.Index64(indices)
                layout = awkward1.layout.IndexedOptionArray64(index, self.layout, parameters=self.layout.parameters)
                return awkward1._util.wrap(layout, self._classes, self._functions)

            else:
                tags = (indices >= 0).view(numpy.int8)
                index = indices.copy()
                index[~tags] = 0
                content0 = awkward1.operations.convert.fromiter([fill_value], highlevel=False)
                content1 = self.layout
                tags = awkward1.layout.Index8(tags)
                index = awkward1.layout.Index64(index)
                layout = awkward1.layout.UnionArray8_64(tags, index, [content0, content1])
                return awkward1._util.wrap(layout, self._classes, self._functions)

        else:
            return self[indices]

    def copy(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.copy.html
        return awkward1._util.wrap(self._layout.deep_copy(copyarrays=True, copyindexes=True, copyidentities=True), self._classes, self._functions)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._concat_same_type.html
        return awkward1.operations.structure.concatenate(to_concat)

    # RECOMMENDED for performance:

    # def fillna(self, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.fillna.html
    #     value, method, limit = extra(args, kwargs, [
    #         ("value", None),
    #         ("method", None),
    #         ("limit", None)])
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
    #     raise NotImplementedError
    #
    # def _values_for_factorize(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._values_for_factorize.html
    #     raise NotImplementedError
    #
    # def argsort(self, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.argsort.html
    #     ascending, kind= extra(args, kwargs, [
    #         ("ascending", True),
    #         ("kind", "quicksort")])   # "quicksort", "mergesort", "heapsort"
    #     raise NotImplementedError
    #
    # def _values_for_argsort(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._values_for_argsort.html
    #     raise NotImplementedError
    #
    # def searchsorted(self, value, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.searchsorted.html
    #     side, sorter = extra(args, kwargs, [
    #         ("side", "left"),
    #         ("sorter", None)])
    #     raise NotImplementedError
    #
    # def _reduce(self, name, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._reduce.html
    #     skipna = extra(args, kwargs, [
    #         ("skipna", True)])
    #     raise NotImplementedError
