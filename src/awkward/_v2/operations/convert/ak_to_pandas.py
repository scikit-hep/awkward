# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_pandas(
    array, how="inner", levelname=lambda i: "sub" * i + "entry", anonymous="values"
):
    raise NotImplementedError


#     """
#     Args:
#         array: Data to convert into one or more Pandas DataFrames.
#         how (None or str): Passed to
#             [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html)
#             to combine DataFrames for each multiplicity into one DataFrame. If
#             None, a list of Pandas DataFrames is returned.
#         levelname (int -> str): Computes a name for each level of the row index
#             from the number of levels deep.
#         anonymous (str): Column name to use if the `array` does not contain
#             records; otherwise, column names are derived from record fields.

#     Converts Awkward data structures into Pandas
#     [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
#     rows and columns. The resulting DataFrame(s) contains no Awkward structures.

#     #ak.Array structures can't be losslessly converted into a single DataFrame;
#     different fields in a record structure might have different nested list
#     lengths, but a DataFrame can have only one index.

#     If `how` is None, this function always returns a list of DataFrames (even
#     if it contains only one DataFrame); otherwise `how` is passed to
#     [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html)
#     to merge them into a single DataFrame with the associated loss of data.

#     In the following example, nested lists are converted into MultiIndex rows.
#     The index level names `"entry"`, `"subentry"` and `"subsubentry"` can be
#     controlled with the `levelname` parameter. The column name `"values"` is
#     assigned because this array has no fields; it can be controlled with the
#     `anonymous` parameter.

#         >>> ak.to_pandas(ak.Array([[[1.1, 2.2], [], [3.3]],
#         ...                        [],
#         ...                        [[4.4], [5.5, 6.6]],
#         ...                        [[7.7]],
#         ...                        [[8.8]]]))
#                                     values
#         entry subentry subsubentry
#         0     0        0               1.1
#                        1               2.2
#               2        0               3.3
#         2     0        0               4.4
#               1        0               5.5
#                        1               6.6
#         3     0        0               7.7
#         4     0        0               8.8

#     In this example, nested records are converted into MultiIndex columns.
#     (MultiIndex rows and columns can be mixed; these examples are deliberately
#     simple.)

#         >>> ak.to_pandas(ak.Array([
#         ...     {"I": {"a": _, "b": {"i": _}}, "II": {"x": {"y": {"z": _}}}}
#         ...     for _ in range(0, 50, 10)]))
#                 I      II
#                 a   b   x
#                     i   y
#                         z
#         entry
#         0       0   0   0
#         1      10  10  10
#         2      20  20  20
#         3      30  30  30
#         4      40  40  40

#     The following two examples show how fields of different length lists are
#     merged. With `how="inner"` (default), only subentries that exist for all
#     fields are preserved; with `how="outer"`, all subentries are preserved at
#     the expense of requiring missing values.

#         >>> ak.to_pandas(ak.Array([{"x": [], "y": [4.4, 3.3, 2.2, 1.1]},
#         ...                        {"x": [1], "y": [3.3, 2.2, 1.1]},
#         ...                        {"x": [1, 2], "y": [2.2, 1.1]},
#         ...                        {"x": [1, 2, 3], "y": [1.1]},
#         ...                        {"x": [1, 2, 3, 4], "y": []}]),
#         ...                        how="inner")
#                         x    y
#         entry subentry
#         1     0         1  3.3
#         2     0         1  2.2
#               1         2  1.1
#         3     0         1  1.1

#     The same with `how="outer"`:

#         >>> ak.to_pandas(ak.Array([{"x": [], "y": [4.4, 3.3, 2.2, 1.1]},
#         ...                        {"x": [1], "y": [3.3, 2.2, 1.1]},
#         ...                        {"x": [1, 2], "y": [2.2, 1.1]},
#         ...                        {"x": [1, 2, 3], "y": [1.1]},
#         ...                        {"x": [1, 2, 3, 4], "y": []}]),
#         ...                        how="outer")
#                           x    y
#         entry subentry
#         0     0         NaN  4.4
#               1         NaN  3.3
#               2         NaN  2.2
#               3         NaN  1.1
#         1     0         1.0  3.3
#               1         NaN  2.2
#               2         NaN  1.1
#         2     0         1.0  2.2
#               1         2.0  1.1
#         3     0         1.0  1.1
#               1         2.0  NaN
#               2         3.0  NaN
#         4     0         1.0  NaN
#               1         2.0  NaN
#               2         3.0  NaN
#               3         4.0  NaN
#     """
#     try:
#         import pandas
#     except ImportError:
#         raise ImportError(
#             """install the 'pandas' package with:

#     pip install pandas --upgrade

# or

#     conda install pandas"""
#         )

#     if how is not None:
#         out = None
#         for df in to_pandas(array, how=None, levelname=levelname, anonymous=anonymous):
#             if out is None:
#                 out = df
#             else:
#                 out = pandas.merge(out, df, how=how, left_index=True, right_index=True)
#         return out

#     def recurse(layout, row_arrays, col_names):
#         if isinstance(layout, ak._v2._util.virtualtypes):
#             return recurse(layout.array, row_arrays, col_names)

#         elif isinstance(layout, ak._v2._util.indexedtypes):
#             return recurse(layout.project(), row_arrays, col_names)

#         elif layout.parameter("__array__") in ("string", "bytestring"):
#             return [(to_numpy(layout), row_arrays, col_names)]

#         elif layout.purelist_depth > 1:
#             offsets, flattened = layout.offsets_and_flatten(axis=1)
#             offsets = numpy.asarray(offsets)
#             starts, stops = offsets[:-1], offsets[1:]
#             counts = stops - starts
#             if ak._v2._util.win or ak._v2._util.bits32:
#                 counts = counts.astype(np.int32)
#             if len(row_arrays) == 0:
#                 newrows = [
#                     numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
#                 ]
#             else:
#                 newrows = [numpy.repeat(x, counts) for x in row_arrays]
#             newrows.append(
#                 numpy.arange(offsets[-1], dtype=counts.dtype)
#                 - numpy.repeat(starts, counts)
#             )
#             return recurse(flattened, newrows, col_names)

#         elif isinstance(layout, ak._v2._util.uniontypes):
#             layout = ak._v2._util.union_to_record(layout, anonymous)
#             if isinstance(layout, ak._v2._util.uniontypes):
#                 return [(to_numpy(layout), row_arrays, col_names)]
#             else:
#                 return sum(
#                     [
#                         recurse(layout.field(n), row_arrays, col_names + (n,))
#                         for n in layout.keys()
#                     ],
#                     [],
#                 )

#         elif isinstance(layout, ak._v2.contents.RecordArray):
#             return sum(
#                 [
#                     recurse(layout.field(n), row_arrays, col_names + (n,))
#                     for n in layout.keys()
#                 ],
#                 [],
#             )

#         else:
#             return [(to_numpy(layout), row_arrays, col_names)]

#     layout = to_layout(array, allow_record=True, allow_other=False)
#     if isinstance(layout, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#         layout = layout.toContent()

#     if isinstance(layout, ak._v2.record.Record):
#         layout2 = layout.array[layout.at : layout.at + 1]
#     else:
#         layout2 = layout

#     tables = []
#     last_row_arrays = None
#     for column, row_arrays, col_names in recurse(layout2, [], ()):
#         if isinstance(layout, ak._v2.record.Record):
#             row_arrays = row_arrays[1:]  # Record --> one-element RecordArray
#         if len(col_names) == 0:
#             columns = [anonymous]
#         else:
#             columns = pandas.MultiIndex.from_tuples([col_names])

#         if (
#             last_row_arrays is not None
#             and len(last_row_arrays) == len(row_arrays)
#             and all(
#                 numpy.array_equal(x, y) for x, y in zip(last_row_arrays, row_arrays)
#             )
#         ):
#             oldcolumns = tables[-1].columns
#             if isinstance(oldcolumns, pandas.MultiIndex):
#                 numold = len(oldcolumns.levels)
#             else:
#                 numold = max(len(x) for x in oldcolumns)
#             numnew = len(columns.levels)
#             maxnum = max(numold, numnew)
#             if numold != maxnum:
#                 oldcolumns = pandas.MultiIndex.from_tuples(
#                     [x + ("",) * (maxnum - numold) for x in oldcolumns]
#                 )
#                 tables[-1].columns = oldcolumns
#             if numnew != maxnum:
#                 columns = pandas.MultiIndex.from_tuples(
#                     [x + ("",) * (maxnum - numnew) for x in columns]
#                 )

#             newframe = pandas.DataFrame(
#                 data=column, index=tables[-1].index, columns=columns
#             )
#             tables[-1] = pandas.concat([tables[-1], newframe], axis=1)

#         else:
#             if len(row_arrays) == 0:
#                 index = pandas.RangeIndex(len(column), name=levelname(0))
#             else:
#                 index = pandas.MultiIndex.from_arrays(
#                     row_arrays, names=[levelname(i) for i in range(len(row_arrays))]
#                 )
#             tables.append(pandas.DataFrame(data=column, index=index, columns=columns))

#         last_row_arrays = row_arrays

#     for table in tables:
#         if isinstance(table.columns, pandas.MultiIndex) and table.columns.nlevels == 1:
#             table.columns = table.columns.get_level_values(0)

#     return tables
