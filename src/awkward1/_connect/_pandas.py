# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import distutils.version

import numpy

import awkward1.layout
import awkward1._util
import awkward1.operations.convert
import awkward1.operations.structure


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
            base = numpy.dtype("O")

            @classmethod
            def construct_from_string(cls, string):
                if string == cls.name:
                    return cls()
                else:
                    raise TypeError(
                        "cannot construct a {0} from {1}".format(cls, string)
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
        register()
        return "dataframe"

    @property
    def columns(self):
        if self.layout.numfields >= 0:
            return self.layout.keys()
        else:
            return [NoFields()]

    def _ixs(self, i, axis):
        register()
        if self.layout.numfields >= 0:
            return get_pandas().Series(self[str(i)])
        else:
            return get_pandas().Series(self)

    # REQUIRED by Pandas:

    @classmethod
    def _from_sequence(cls, scalars, *args, **kwargs):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._from_sequence.html
        register()
        dtype, copy = awkward1._util.extra(
            args, kwargs, [("dtype", None), ("copy", False)]
        )
        return awkward1.operations.convert.from_iter(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._from_factorized.html
        register()
        raise NotImplementedError("_from_factorized")

    # __getitem__(self)
    # __len__(self)

    @property
    def dtype(self):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.dtype.html
        if awkward1._util.called_by_module(
            "pandas"
        ) and not awkward1._util.called_by_module("dask"):
            register()
            if isinstance(self.layout, awkward1.partition.PartitionedArray):
                raise ValueError(
                    "partitioned arrays cannot be Pandas columns; "
                    "try ak.repartition(array, None)"
                )
            else:
                return AwkwardDtype()

        else:
            return numpy.dtype(numpy.object)

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
        register()
        return numpy.array(awkward1.operations.structure.is_none(self))

    def take(self, indices, *args, **kwargs):
        # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.take.html
        allow_fill, fill_value = awkward1._util.extra(
            args, kwargs, [("allow_fill", False), ("fill_value", None)]
        )
        register()

        if allow_fill:
            content1 = self.layout
            if isinstance(content1, awkward1.partition.PartitionedArray):
                content1 = content1.toContent()

            indices = numpy.asarray(indices, dtype=numpy.int64)
            if fill_value is None:
                index = awkward1.layout.Index64(indices)
                layout = awkward1.layout.IndexedOptionArray64(
                    index, content1, parameters=self.layout.parameters
                )
                return awkward1._util.wrap(layout, awkward1._util.behaviorof(self))

            else:
                tags = (indices >= 0).view(numpy.int8)
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
        register()
        return awkward1.operations.structure.concatenate(to_concat)

    # RECOMMENDED for performance:

    # def fillna(self, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.fillna.html
    #     value, method, limit = awkward1._util.extra(args, kwargs, [
    #         ("value", None),
    #         ("method", None),
    #         ("limit", None)])
    #     register()
    #     raise NotImplementedError
    #
    # def dropna(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.dropna.html
    #     register()
    #     raise NotImplementedError
    #
    # def unique(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.unique.html
    #     register()
    #     raise NotImplementedError
    #
    # def factorize(self, na_sentinel):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.factorize.html
    #     register()
    #     raise NotImplementedError
    #
    # def _values_for_factorize(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._values_for_factorize.html
    #     register()
    #     raise NotImplementedError
    #
    # def argsort(self, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.argsort.html
    #     ascending, kind = awkward1._util.extra(args, kwargs, [
    #         ("ascending", True),
    #         ("kind", "quicksort")])   # "quicksort", "mergesort", "heapsort"
    #     register()
    #     raise NotImplementedError
    #
    # def _values_for_argsort(self):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._values_for_argsort.html
    #     register()
    #     raise NotImplementedError
    #
    # def searchsorted(self, value, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray.searchsorted.html
    #     side, sorter = awkward1._util.extra(args, kwargs, [
    #         ("side", "left"),
    #         ("sorter", None)])
    #     register()
    #     raise NotImplementedError
    #
    # def _reduce(self, name, *args, **kwargs):
    #     # https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.api.extensions.ExtensionArray._reduce.html
    #     skipna, = awkward1._util.extra(args, kwargs, [
    #         ("skipna", True)])
    #     register()
    #     raise NotImplementedError


def df(array, how="inner", levelname=lambda i: "sub" * i + "entry", anonymous="values"):
    register()
    pandas = get_pandas()
    out = None
    for df in dfs(array, levelname=levelname, anonymous=anonymous):
        if out is None:
            out = df
        else:
            out = pandas.merge(out, df, how=how, left_index=True, right_index=True)
    return out


def dfs(array, levelname=lambda i: "sub" * i + "entry", anonymous="values"):
    register()
    pandas = get_pandas()

    def recurse(layout, row_arrays, col_names):
        if layout.purelist_depth > 1:
            offsets, flattened = layout.offsets_and_flatten(axis=1)
            offsets = numpy.asarray(offsets)
            starts, stops = offsets[:-1], offsets[1:]
            counts = stops - starts
            if awkward1._util.win:
                counts = counts.astype(numpy.int32)
            if len(row_arrays) == 0:
                newrows = [
                    numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
                ]
            else:
                newrows = [numpy.repeat(x, counts) for x in row_arrays]
            newrows.append(
                numpy.arange(offsets[-1], dtype=counts.dtype)
                - numpy.repeat(starts, counts)
            )
            return recurse(flattened, newrows, col_names)

        elif isinstance(layout, awkward1.layout.RecordArray):
            return sum(
                [
                    recurse(layout.field(n), row_arrays, col_names + (n,))
                    for n in layout.keys()
                ],
                [],
            )

        else:
            try:
                return [
                    (
                        awkward1.operations.convert.to_numpy(layout),
                        row_arrays,
                        col_names,
                    )
                ]
            except Exception:
                return [(layout, row_arrays, col_names)]

    layout = awkward1.operations.convert.to_layout(
        array, allow_record=True, allow_other=False
    )
    if isinstance(layout, awkward1.partition.PartitionedArray):
        layout = layout.toContent()

    if isinstance(layout, awkward1.layout.Record):
        layout2 = layout.array[layout.at : layout.at + 1]
    else:
        layout2 = layout

    tables = []
    last_row_arrays = None
    for column, row_arrays, col_names in recurse(layout2, [], ()):
        if isinstance(layout, awkward1.layout.Record):
            row_arrays = row_arrays[1:]  # Record --> one-element RecordArray
        if len(col_names) == 0:
            columns = [anonymous]
        else:
            columns = pandas.MultiIndex.from_tuples([col_names])

        if (
            last_row_arrays is not None
            and len(last_row_arrays) == len(row_arrays)
            and all(
                numpy.array_equal(x, y) for x, y in zip(last_row_arrays, row_arrays)
            )
        ):
            oldcolumns = tables[-1].columns
            if isinstance(oldcolumns, pandas.MultiIndex):
                numold = len(oldcolumns.levels)
            else:
                numold = max(len(x) for x in oldcolumns)
            numnew = len(columns.levels)
            maxnum = max(numold, numnew)
            if numold != maxnum:
                oldcolumns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxnum - numold) for x in oldcolumns]
                )
                tables[-1].columns = oldcolumns
            if numnew != maxnum:
                columns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxnum - numnew) for x in columns]
                )

            newframe = pandas.DataFrame(
                data=column, index=tables[-1].index, columns=columns
            )
            tables[-1] = pandas.concat([tables[-1], newframe], axis=1)

        else:
            if len(row_arrays) == 0:
                index = pandas.RangeIndex(len(column), name=levelname(0))
            else:
                index = pandas.MultiIndex.from_arrays(
                    row_arrays, names=[levelname(i) for i in range(len(row_arrays))]
                )
            tables.append(pandas.DataFrame(data=column, index=index, columns=columns))

        last_row_arrays = row_arrays

    return tables
