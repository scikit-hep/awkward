# How-to questions survey

## What are the most-asked about topics?

| Frequency | Rough categories |
|:---------:|:-----------------|
| 20 | Pandas |
| 10 | persistence, including HDF5 (4) and Parquet (2) |
| 8  | Lorentz vectors and TVector3 |
| 8  | reducing |
| 7  | jagged arrays |
| 6  | ak.fillna and ak.rpad (regularizing) |
| 6  | ak.concatenate and stack |
| 5  | strings |
| 5  | getitem |
| 4  | Numba |
| 4  | Histogramming (flattening before) |
| 3  | lazy arrays |
| 3  | ak.cross |
| 2  | Arrow |
| 2  | ak.with_field |
| 2  | ak.where and np.choose |
| 2  | ak.num (including axis=1) |
| 1  | SQL-like join |
| 1  | reducer-like |
| 1  | np.digitize |
| 1  | non-ufuncs |
| 1  | nested records |
| 1  | named axis |
| 1  | masking |
| 1  | Functional interface |
| 1  | Content vs flatten |
| 1  | Changing leaf (numeric) types |
| 1  | Caching |
| 1  | Broadcasting |
| 1  | ak.zip |
| 1  | ak.sort |
| 1  | ak.isnan |

## Categories of categories

   * Data into other systems: Pandas (20), NumPy/regularizing (6), Histogramming (4), Arrow (2)
   * Persistence (10), laziness (3)
   * Physics objects/vectors (8)
   * Restructuring: reducing (8), concatenate (6), getitem (5), cross (3), num (2), with_field (2), where (2), zip/sort (2), digitize (1)
   * Types: Jagged arrays (7), strings (5)
   * Fast code: Numba (4)
   * Lazy arrays (3)
   * SQL-like join (1)
   * All the rest

## References

   * ak.all: https://github.com/scikit-hep/awkward-0.x/issues/166
   * ak.all: https://github.com/scikit-hep/awkward-0.x/issues/53
   * ak.argmax: https://github.com/scikit-hep/awkward-0.x/issues/158
   * ak.argmax: https://github.com/scikit-hep/awkward-0.x/issues/176
   * ak.concatenate axis=1: https://github.com/scikit-hep/awkward-0.x/issues/205
   * ak.concatenate: https://github.com/scikit-hep/awkward-0.x/issues/127
   * ak.concatenate: https://github.com/scikit-hep/awkward-0.x/issues/149
   * ak.concatenate: https://github.com/scikit-hep/awkward-0.x/issues/195
   * ak.concatenate: https://stackoverflow.com/questions/58440970/how-to-combine-two-uproot-jagged-arrays-into-one
   * ak.count: https://github.com/scikit-hep/awkward-0.x/issues/185
   * ak.cross: https://github.com/scikit-hep/awkward-0.x/issues/180
   * ak.cross: https://github.com/scikit-hep/awkward-0.x/issues/212
   * ak.cross: https://stackoverflow.com/questions/60003131/propagating-a-selection-on-a-subset-of-awkward-array-back-up
   * ak.fillna: https://github.com/scikit-hep/awkward-0.x/issues/201
   * ak.isnan: https://github.com/scikit-hep/awkward-0.x/issues/227
   * ak.max: https://github.com/scikit-hep/awkward-0.x/issues/100
   * ak.max: https://github.com/scikit-hep/awkward-0.x/issues/69
   * ak.num: https://github.com/scikit-hep/awkward-0.x/issues/230
   * ak.num with axis=1: https://stackoverflow.com/questions/58459548/can-jaggedarray-counts-innermost-layer-and-return-another-jaggeredarray
   * ak.rpad, ak.fillna: https://stackoverflow.com/questions/58308027/efficiently-sorting-and-filtering-a-jaggedarray-by-another-one
   * ak.rpad and ak.fillna: https://stackoverflow.com/questions/58923496/zero-padding-variable-length-of-array-in-uproot
   * ak.rpad: https://github.com/scikit-hep/awkward-0.x/issues/190
   * ak.sort: https://github.com/scikit-hep/awkward-0.x/issues/81
   * ak.where: https://github.com/scikit-hep/awkward-0.x/issues/84
   * ak.with_field: https://github.com/scikit-hep/awkward-0.x/issues/116
   * ak.with_field: https://stackoverflow.com/questions/60569460/appending-columns-to-awkward-table
   * Arrow: https://github.com/scikit-hep/awkward-0.x/issues/203
   * Arrow: https://stackoverflow.com/questions/59198577/arrow-listarray-from-pandas-has-very-different-structure-from-arrow-array-genera
   * Broadcasting: https://github.com/scikit-hep/uproot3/issues/250
   * Caching: https://github.com/scikit-hep/uproot3/issues/409
   * Changing leaf types: https://github.com/scikit-hep/awkward-0.x/issues/39
   * Content vs flatten: https://github.com/scikit-hep/awkward-0.x/issues/82
   * Flatten before plotting: https://github.com/scikit-hep/uproot3/issues/349
   * From Pandas: https://github.com/scikit-hep/awkward-0.x/issues/215
   * Functional interface: https://github.com/scikit-hep/awkward-0.x/issues/93
   * HDF5, ak.rpad: https://stackoverflow.com/questions/59001209/read-a-tree-from-a-root-file-and-then-make-zero-padded-h5-file
   * HDF5: https://github.com/scikit-hep/awkward-0.x/issues/17
   * HDF5: https://github.com/scikit-hep/awkward-0.x/issues/210
   * HDF5: https://github.com/scikit-hep/awkward-0.x/issues/67
   * Histograms: https://stackoverflow.com/questions/58503117/may-i-see-a-short-example-of-cutting-on-data-to-prepare-it-for-histogramming-in
   * Histograms: https://stackoverflow.com/questions/58503117/may-i-see-a-short-example-of-cutting-on-data-to-prepare-it-for-histogramming-in
   * Histograms: https://stackoverflow.com/questions/59166618/getting-a-histogram-of-a-jaggedarray
   * Indexing: https://github.com/scikit-hep/awkward-0.x/issues/31
   * Jagged array: https://stackoverflow.com/questions/60103825/retrieve-data-in-pandas
   * Jagged arrays and Pandas: https://github.com/scikit-hep/uproot3/issues/88
   * Jagged arrays and Pandas: https://github.com/scikit-hep/uproot3/issues/97
   * Jagged arrays, how does it work? https://github.com/scikit-hep/uproot3/issues/119
   * Jagged arrays: https://github.com/scikit-hep/uproot3/issues/43
   * Jagged arrays: https://github.com/scikit-hep/uproot3/issues/99
   * Jagged arrays: https://stackoverflow.com/questions/58912123/python-collection-of-different-sized-arrays-jagged-arrays-dask
   * Jagged indexing: https://github.com/scikit-hep/awkward-0.x/issues/109
   * Jagged indexing: https://github.com/scikit-hep/awkward-0.x/issues/163
   * Jagged indexing: https://github.com/scikit-hep/uproot3/issues/103
   * Lazy arrays: https://github.com/scikit-hep/uproot3/issues/159
   * Lazy arrays: https://stackoverflow.com/questions/58747852/how-to-use-lazyarrays-in-uproot
   * Lorentz vectors: https://github.com/scikit-hep/awkward-0.x/issues/128
   * Lorentz vectors: https://github.com/scikit-hep/awkward-0.x/issues/56
   * Lorentz vectors: https://github.com/scikit-hep/awkward-0.x/issues/85
   * Lorentz vectors: https://github.com/scikit-hep/awkward-0.x/issues/9
   * Lorentz vectors: https://github.com/scikit-hep/uproot3/issues/57
   * Lorentz vectors: https://github.com/scikit-hep/uproot3/issues/96
   * Masking: https://github.com/scikit-hep/awkward-0.x/issues/134
   * Named axis: https://stackoverflow.com/questions/59150837/awkward-array-fancy-indexing-with-boolean-mask-along-named-axis
   * Nested records: https://github.com/scikit-hep/awkward-0.x/issues/95
   * Non-ufuncs: https://github.com/scikit-hep/awkward-0.x/issues/13
   * np.choose: https://stackoverflow.com/questions/58750767/awkward-arrays-choosing-an-item-from-each-row-according-to-a-list-of-indices
   * np.digitize: https://github.com/scikit-hep/awkward-0.x/issues/200
   * np.stack: https://stackoverflow.com/questions/60336285/is-there-a-way-to-stack-jaggedarrays-without-fromiter
   * Numba: https://github.com/scikit-hep/awkward-0.x/issues/192
   * Numba: https://github.com/scikit-hep/awkward-0.x/issues/226
   * Numba: https://github.com/scikit-hep/awkward-0.x/issues/63
   * Numba: https://github.com/scikit-hep/awkward-0.x/issues/73
   * NumPy advanced indexing: https://github.com/scikit-hep/awkward-0.x/issues/58
   * Pandas: https://github.com/scikit-hep/awkward-0.x/issues/197
   * Pandas: https://github.com/scikit-hep/awkward-0.x/issues/30
   * Pandas: https://github.com/scikit-hep/uproot3/issues/102
   * Pandas: https://github.com/scikit-hep/uproot3/issues/156
   * Pandas: https://github.com/scikit-hep/uproot3/issues/177
   * Pandas: https://github.com/scikit-hep/uproot3/issues/179
   * Pandas: https://github.com/scikit-hep/uproot3/issues/227
   * Pandas: https://github.com/scikit-hep/uproot3/issues/256
   * Pandas: https://github.com/scikit-hep/uproot3/issues/263
   * Pandas: https://github.com/scikit-hep/uproot3/issues/284
   * Pandas: https://github.com/scikit-hep/uproot3/issues/322
   * Pandas: https://github.com/scikit-hep/uproot3/issues/396
   * Pandas: https://github.com/scikit-hep/uproot3/issues/397
   * Pandas: https://github.com/scikit-hep/uproot3/issues/61
   * Pandas: https://github.com/scikit-hep/uproot3/issues/86
   * Pandas: https://stackoverflow.com/questions/58986029/reindex-panda-multiindex
   * Pandas: https://stackoverflow.com/questions/60058505/get-data-from-pandas-multiindex
   * Pandas with strings: https://stackoverflow.com/questions/58937233/strings-in-pandas-dataframe-from-uproot
   * Pandas with TVector3: https://stackoverflow.com/questions/59930539/how-can-i-load-a-ttree-with-tvector3-branches-into-a-pandas-dataframe-using-upro
   * Parquet: https://stackoverflow.com/questions/59191959/awkwardarray-possible-to-append-an-array-to-an-exisitng-parquet-file
   * Parquet: https://stackoverflow.com/questions/59264202/awkward-array-how-to-get-numpy-array-after-storing-as-parquet-not-bitmasked
   * Persistence and Lorentz vectors: https://github.com/scikit-hep/awkward-0.x/issues/169
   * Persistence as compact: https://stackoverflow.com/questions/60381016/how-to-copy-a-jagged-array-in-awkward-array
   * Persistence: https://github.com/scikit-hep/awkward-0.x/issues/189
   * Persistence lazy arrays: https://github.com/scikit-hep/awkward-0.x/issues/220
   * Reducer-like processing: https://github.com/scikit-hep/awkward-0.x/issues/232
   * Reducers: https://github.com/scikit-hep/awkward-0.x/issues/38
   * SQL-like join: https://github.com/scikit-hep/uproot3/issues/314
   * Strings: https://github.com/scikit-hep/uproot3/issues/31
   * Strings: https://github.com/scikit-hep/uproot3/issues/32
   * Strings: https://github.com/scikit-hep/uproot3/issues/63
   * Strings: https://stackoverflow.com/questions/58922824/arrays-of-strings-from-uproot
   * To NumPy: https://github.com/scikit-hep/uproot3/issues/208
   * Zip: https://github.com/scikit-hep/uproot3/issues/252

## Bugs in Awkward structures

   * ChunkedArray and jagged indexing: https://github.com/scikit-hep/awkward-0.x/issues/186
   * ChunkedArray and ObjectArray: https://github.com/scikit-hep/awkward-0.x/issues/229
   * ChunkedArray: https://github.com/scikit-hep/awkward-0.x/issues/161
   * ChunkedArray: https://github.com/scikit-hep/awkward-0.x/issues/181
   * ChunkedArray: https://github.com/scikit-hep/awkward-0.x/issues/234
   * ChunkedArray: https://github.com/scikit-hep/uproot3/issues/296
   * ChunkedArray: https://github.com/scikit-hep/uproot3/issues/317
   * ChunkedArray: https://github.com/scikit-hep/uproot3/issues/323
   * ChunkedArray: https://github.com/scikit-hep/uproot3/issues/377
   * ChunkedArray: https://github.com/scikit-hep/uproot3/issues/398
   * ChunkedArray: https://github.com/scikit-hep/uproot3/issues/412
   * ChunkedArray: https://github.com/scikit-hep/uproot3/issues/458
   * ChunkedArray: https://stackoverflow.com/questions/58921328/using-chunked-array-akwkard-lib-for-fancy-indexing-or-masking
   * ChunkedArray: https://stackoverflow.com/questions/59277507/content-starts-and-stops-of-chunkedarray-built-from-lazyarray
   * JaggedArray: https://github.com/scikit-hep/awkward-0.x/issues/239
   * Lorentz vectors: https://github.com/scikit-hep/awkward-0.x/issues/238
   * Min/Max identities: https://github.com/scikit-hep/awkward-0.x/issues/237
   * ObjectArray: https://github.com/scikit-hep/awkward-0.x/issues/103
   * ObjectArray: https://github.com/scikit-hep/uproot3/issues/452
   * ObjectArray: https://stackoverflow.com/questions/60250877/combine-awkward-array-jaggedarray-contents-and-offsets-into-nested-jaggedarray
