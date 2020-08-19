# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import distutils.version
import warnings

import awkward1.layout
import awkward1._util
import awkward1.operations.convert
import awkward1.operations.structure
import awkward1.nplike


np = awkward1.nplike.NumpyMetadata.instance()


def register():
    global AwkwardDtype
    try:
        pandas = get_pandas()
    except ImportError:
        pass
    else:
        if AwkwardDtype is None:
            get_dtype()
        if issubclass(PandasMixin, PandasNotImportedYet):
            PandasMixin.__bases__ = (pandas.api.extensions.ExtensionArray,)


def vote():
    global AwkwardDtype

    if AwkwardDtype is None:
        raise RuntimeError(
            "Use of Awkward Arrays in Pandas Series and DataFrames is deprecated."
            "\nFor now, you can\n\n    ak.pandas.register()\n\n"
            "to get this feature, but it will be removed in Awkward1 0.3.0. "
            "(see https://github.com/scikit-hep/awkward-1.0/issues/350)"
        )


checked_version = False


def get_pandas():
    global checked_version
    try:
        import pandas
    except ImportError:
        raise ImportError(
            """install the 'pandas' package with:

    pip install pandas --upgrade

or

    conda install pandas"""
        )
    else:
        if not checked_version and distutils.version.LooseVersion(
            pandas.__version__
        ) < distutils.version.LooseVersion("0.24"):
            raise ImportError(
                "awkward1 can only work with pandas 0.24 or later "
                "(you have version {0})".format(pandas.__version__)
            )
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
            base = np.dtype("O")

            @classmethod
            def construct_from_string(cls, string):
                if string == cls.name:
                    return cls()
                else:
                    raise TypeError(
                        "cannot construct a {0} from {1}".format(cls, string)
                        + awkward1._util.exception_suffix(__file__)
                    )

            @classmethod
            def construct_array_type(cls):
                return awkward1.highlevel.Array

    return AwkwardDtype


class PandasNotImportedYet(object):
    pass


class NoFields(object):
    def __str__(self):
        return "(no fields)"

    def __eq__(self, other):
        return other is NoFields or isinstance(other, NoFields)

    def __hash__(self):
        return hash(NoFields)


class PandasMixin(PandasNotImportedYet):
    @property
    def _typ(self):
        vote()
        return "dataframe"

    @property
    def columns(self):
        if self.layout.numfields >= 0:
            return self.layout.keys()
        else:
            return [NoFields()]

    def _ixs(self, i, axis):
        vote()
        if self.layout.numfields >= 0:
            return get_pandas().Series(self[str(i)])
        else:
            return get_pandas().Series(self)

    # REQUIRED by Pandas:

    @classmethod
    def _from_sequence(cls, scalars, *args, **kwargs):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._from_sequence.html
        vote()
        dtype, copy = awkward1._util.extra(
            args, kwargs, [("dtype", None), ("copy", False)]
        )
        return awkward1.operations.convert.from_iter(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._from_factorized.html
        vote()
        raise NotImplementedError(
            "_from_factorized"
            + awkward1._util.exception_suffix(__file__)
        )

    # __getitem__(self)
    # __len__(self)

    @property
    def dtype(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.dtype.html
        if awkward1._util.called_by_module(
            "pandas"
        ) and not awkward1._util.called_by_module("dask"):
            vote()
            if isinstance(self.layout, awkward1.partition.PartitionedArray):
                raise ValueError(
                    "partitioned arrays cannot be Pandas columns; "
                    "try ak.repartition(array, None)"
                    + awkward1._util.exception_suffix(__file__)
                )
            else:
                return AwkwardDtype()

        else:
            return np.dtype(np.object)

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
        vote()
        nplike = awkward1.nplike.of(self)
        return nplike.array(awkward1.operations.structure.is_none(self))

    def take(self, indices, *args, **kwargs):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.take.html
        allow_fill, fill_value = awkward1._util.extra(
            args, kwargs, [("allow_fill", False), ("fill_value", None)]
        )
        vote()

        nplike = awkward1.nplike.of(self)
        if allow_fill:
            content1 = self.layout
            if isinstance(content1, awkward1.partition.PartitionedArray):
                content1 = content1.toContent()

            indices = nplike.asarray(indices, dtype=np.int64)
            if fill_value is None:
                index = awkward1.layout.Index64(indices)
                layout = awkward1.layout.IndexedOptionArray64(
                    index, content1, parameters=self.layout.parameters
                )
                return awkward1._util.wrap(layout, awkward1._util.behaviorof(self))

            else:
                tags = (indices >= 0).view(np.int8)
                index = indices.copy()
                index[~tags] = 0
                content0 = awkward1.operations.convert.from_iter(
                    [fill_value], highlevel=False
                )
                tags = awkward1.layout.Index8(tags)
                index = awkward1.layout.Index64(index)
                layout = awkward1.layout.UnionArray8_64(
                    tags, index, [content0, content1]
                )
                return awkward1._util.wrap(layout, awkward1._util.behaviorof(self))

        else:
            return self[indices]

    def copy(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.copy.html
        return awkward1._util.wrap(
            self._layout.deep_copy(
                copyarrays=True, copyindexes=True, copyidentities=True
            ),
            awkward1._util.behaviorof(self),
        )

    @classmethod
    def _concat_same_type(cls, to_concat):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._concat_same_type.html
        vote()
        return awkward1.operations.structure.concatenate(to_concat)

    # RECOMMENDED for performance:

    # def fillna(self, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.fillna.html
    #     value, method, limit = awkward1._util.extra(args, kwargs, [
    #         ("value", None),
    #         ("method", None),
    #         ("limit", None)])
    #     vote()
    #     raise NotImplementedError
    #
    # def dropna(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.dropna.html
    #     vote()
    #     raise NotImplementedError
    #
    # def unique(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.unique.html
    #     vote()
    #     raise NotImplementedError
    #
    # def factorize(self, na_sentinel):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.factorize.html
    #     vote()
    #     raise NotImplementedError
    #
    # def _values_for_factorize(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._values_for_factorize.html
    #     vote()
    #     raise NotImplementedError
    #
    # def argsort(self, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.argsort.html
    #     ascending, kind = awkward1._util.extra(args, kwargs, [
    #         ("ascending", True),
    #         ("kind", "quicksort")])   # "quicksort", "mergesort", "heapsort"
    #     vote()
    #     raise NotImplementedError
    #
    # def _values_for_argsort(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._values_for_argsort.html
    #     vote()
    #     raise NotImplementedError
    #
    # def searchsorted(self, value, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.searchsorted.html
    #     side, sorter = awkward1._util.extra(args, kwargs, [
    #         ("side", "left"),
    #         ("sorter", None)])
    #     vote()
    #     raise NotImplementedError
    #
    # def _reduce(self, name, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._reduce.html
    #     skipna, = awkward1._util.extra(args, kwargs, [
    #         ("skipna", True)])
    #     vote()
    #     raise NotImplementedError


def df(array, how="inner", levelname=lambda i: "sub" * i + "entry", anonymous="values"):
    warnings.warn(
        "ak.pandas.df is deprecated, will be removed in 0.3.0. Use\n\n"
        "    ak.to_pandas(array)\n\ninstead.",
        DeprecationWarning,
    )
    return awkward1.operations.convert.to_pandas(
        array, how=how, levelname=levelname, anonymous=anonymous
    )

def dfs(array, levelname=lambda i: "sub" * i + "entry", anonymous="values"):
    warnings.warn(
        "ak.pandas.dfs is deprecated, will be removed in 0.3.0. Use\n\n"
        "    ak.to_pandas(array, how=None)\n\ninstead.",
        DeprecationWarning,
    )
    return awkward1.operations.convert.to_pandas(
        array, how=None, levelname=levelname, anonymous=anonymous
    )
