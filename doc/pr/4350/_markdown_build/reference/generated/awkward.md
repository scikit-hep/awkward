# awkward

## Submodules

| `behaviors`   |    |
|---------------|----|
| `builder`     |    |
| `contents`    |    |
| `cppyy`       |    |
| `errors`      |    |
| `forms`       |    |
| `forth`       |    |
| `highlevel`   |    |
| `index`       |    |
| `jax`         |    |
| `numba`       |    |
| `operations`  |    |
| `prettyprint` |    |
| `record`      |    |
| `types`       |    |
| `typetracer`  |    |

## Attributes

| [`behavior`](sphinx-llm:4b60b7e758c24ef49b600a5157f03bae)   |    |
|-------------------------------------------------------------|----|
| [`__all__`](sphinx-llm:1626512509c548d5af55c586e28a654f)    |    |

## Classes

| `Array`        |    |
|----------------|----|
| `ArrayBuilder` |    |
| `Record`       |    |

## Functions

| `mixin_class`(registry[, name])                                |                                                                                                   |
|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `mixin_class_method`(ufunc[, rhs, transpose])                  |                                                                                                   |
| `all`(array[, axis, keepdims, mask_identity, highlevel, ...])  | Returns whether all elements are True over one or all levels of nesting.                          |
| `almost_equal`(left, right, \*[, rtol, atol, ...])             | Returns True if two arrays are equal within the given tolerances and options.                     |
| `angle`(val[, deg, highlevel, behavior, attrs])                | Returns the counterclockwise angle of each complex element in radians or degrees.                 |
| `any`(array[, axis, keepdims, mask_identity, highlevel, ...])  | Returns whether any elements are True over one or all levels of nesting.                          |
| `argcartesian`(arrays[, axis, nested, parameters, ...])        | Computes the Cartesian product of arrays, returning integer indexes.                              |
| `argcombinations`(array, n, \*[, replacement, axis, ...])      | Computes combinations of n items from an array, returning integer indexes.                        |
| `argmax`(array[, axis, keepdims, mask_identity, ...])          | Returns the index of the maximum value over one or all levels of nesting.                         |
| `nanargmax`(array[, axis, keepdims, mask_identity, ...])       | Returns the index of the maximum value, treating NaN values as missing.                           |
| `argmin`(array[, axis, keepdims, mask_identity, ...])          | Returns the index of the minimum value over one or all levels of nesting.                         |
| `nanargmin`(array[, axis, keepdims, mask_identity, ...])       | Returns the index of the minimum value, treating NaN values as missing.                           |
| `argsort`(array[, axis, ascending, stable, highlevel, ...])    | Sorts an array along an axis, returning integer indexes.                                          |
| `array_equal`(a1, a2[, equal_nan, dtype_exact, ...])           | Returns True if two arrays have the same shape and elements.                                      |
| `backend`(\*arrays)                                            | Returns the name of the backend used by the given arrays.                                         |
| `broadcast_arrays`(\*arrays[, depth_limit, ...])               | Broadcasts arrays together so they can be combined element-by-element.                            |
| `broadcast_fields`(\*arrays[, highlevel, behavior, attrs])     | Returns a list of arrays whose record types contain the same fields.                              |
| `cartesian`(arrays[, axis, nested, parameters, ...])           | Computes the Cartesian product (cross product) of data from a set of arrays.                      |
| `categories`(array[, highlevel, behavior, attrs])              | Returns the categories of a categorical array.                                                    |
| `combinations`(array, n, \*[, replacement, axis, fields, ...]) | Computes combinations of n items from an array, without replacement.                              |
| `concatenate`(arrays[, axis, mergebool, highlevel, ...])       | Returns an array with the given arrays concatenated along an axis.                                |
| `copy`(array)                                                  | Returns a deep copy of the array (no memory shared with original).                                |
| `corr`(x, y[, weight, axis, keepdims, mask_identity, ...])     | Computes the correlation of x and y over one or all levels of nesting.                            |
| `count`(array[, axis, keepdims, mask_identity, ...])           | Counts an array's elements over one or all levels of nesting.                                     |
| `count_nonzero`(array[, axis, keepdims, mask_identity, ...])   | Counts an array's nonzero elements over one or all levels of nesting.                             |
| `covar`(x, y[, weight, axis, keepdims, mask_identity, ...])    | Computes the covariance of x and y over one or all levels of nesting.                             |
| `drop_none`(array[, axis, highlevel, behavior, attrs])         | Removes missing values (None) from a given array.                                                 |
| `enforce_type`(array, type, \*[, highlevel, behavior, attrs])  | Returns an array whose structure is modified to match the given type.                             |
| `fields`(array)                                                | Returns a list of field names or tuple slot numbers for the outermost record.                     |
| `fill_none`(array, value[, axis, highlevel, behavior, attrs])  | Replaces missing values (None) with a given value.                                                |
| `firsts`(array[, axis, highlevel, behavior, attrs])            | Returns the first element of each list, or None for each empty list.                              |
| `flatten`(array[, axis, highlevel, behavior, attrs])           | Returns an array with one or all levels of nesting removed.                                       |
| `from_arrow`(array, \*[, generate_bitmasks, highlevel, ...])   | Converts an Apache Arrow array into an Awkward Array.                                             |
| `from_arrow_schema`(schema)                                    | Converts an Apache Arrow schema into an Awkward Form.                                             |
| `from_avro_file`(file[, limit_entries, debug_forth, ...])      | Reads an Avro file as an Awkward Array.                                                           |
| `from_buffers`(form, length, container[, buffer_key, ...])     | Reconstitutes an Awkward Array from a Form, length, and memory buffers.                           |
| `from_categorical`(array, \*[, highlevel, behavior, attrs])    | Replaces categorical data with equivalent non-categorical data.                                   |
| `from_cupy`(array, \*[, regulararray, highlevel, ...])         | Converts a CuPy array into an Awkward Array.                                                      |
| `from_dlpack`(array, \*[, prefer_cpu, regulararray, ...])      | Converts a DLPack-aware array into an Awkward Array.                                              |
| `from_feather`(path, \*[, columns, use_threads, ...])          | Reads a Feather file as an Awkward Array (through pyarrow).                                       |
| `from_iter`(iterable, \*[, allow_record, highlevel, ...])      | Converts Python data into an Awkward Array.                                                       |
| `from_jax`(array, \*[, regulararray, highlevel, ...])          | Converts a JAX Array into an Awkward Array.                                                       |
| `from_json`(source, \*[, line_delimited, schema, ...])         | Reads JSON from a string, bytes, file, or URL into an Awkward Array.                              |
| `from_numpy`(array, \*[, regulararray, recordarray, ...])      | Converts a NumPy array into an Awkward Array.                                                     |
| `from_parquet`(path, \*[, columns, row_groups, ...])           | Reads data from a local or remote Parquet file or collection of files.                            |
| `from_raggedtensor`(array)                                     | Converts a TensorFlow RaggedTensor into an Awkward Array.                                         |
| `from_rdataframe`(rdf, columns, \*[, keep_order, ...])         | Converts ROOT RDataFrame columns into an Awkward Array.                                           |
| `from_regular`(array[, axis, highlevel, behavior, attrs])      | Converts one or all regular axes into irregular ones.                                             |
| `from_safetensors`(source, \*[, storage_options, ...])         | Reads a safetensors file as an Awkward Array.                                                     |
| `from_tensorflow`(array)                                       | Converts a TensorFlow Tensor into an Awkward Array.                                               |
| `from_torch`(array)                                            | Converts a PyTorch Tensor into an Awkward Array.                                                  |
| `full_like`(array, fill_value, \*[, dtype, ...])               | Returns an array with the same structure as the input, filled with a given value.                 |
| `imag`(val[, highlevel, behavior, attrs])                      | Returns the imaginary components of the given array elements.                                     |
| `is_categorical`(array)                                        | Returns True if the array is categorical.                                                         |
| `is_none`(array[, axis, highlevel, behavior, attrs])           | Returns an array with True where an element is None at a given axis depth, False otherwise.       |
| `is_tuple`(array)                                              | Returns True if a record, or the outermost record of an array, is a tuple.                        |
| `is_valid`(array, \*[, exception])                             | Returns True if the array has no structural errors and False otherwise.                           |
| `isclose`(a, b[, rtol, atol, equal_nan, highlevel, ...])       | Returns a boolean array of element-wise approximate-equality between two arrays.                  |
| `linear_fit`(x, y[, weight, axis, keepdims, ...])              | Computes the linear fit of y against x over one or all levels of nesting.                         |
| `local_index`(array[, axis, highlevel, behavior, attrs])       | Returns the within-list index of each element at a given axis depth.                              |
| `mask`(array, mask, \*[, valid_when, highlevel, ...])          | Returns an array with elements replaced by None where a mask condition fails.                     |
| `materialize`(array[, highlevel, behavior, attrs])             | Materializes any virtual buffers in the array.                                                    |
| `max`(array[, axis, keepdims, initial, mask_identity, ...])    | Returns the maximum value over one or all levels of nesting.                                      |
| `nanmax`(array[, axis, keepdims, initial, ...])                | Returns the maximum value, treating NaN values as missing.                                        |
| `mean`(x[, weight, axis, keepdims, mask_identity, ...])        | Computes the mean over one or all levels of nesting.                                              |
| `nanmean`(x[, weight, axis, keepdims, mask_identity, ...])     | Computes the mean, treating NaN values as missing.                                                |
| `merge_option_of_records`(array[, axis, highlevel, ...])       | Simplifies options of records into records of options.                                            |
| `merge_union_of_records`(array[, axis, highlevel, ...])        | Simplifies unions of records into records of options.                                             |
| `metadata_from_parquet`(path, \*[, storage_options, ...])      | Reads metadata from a Parquet file or dataset without reading the data.                           |
| `min`(array[, axis, keepdims, initial, mask_identity, ...])    | Returns the minimum value over one or all levels of nesting.                                      |
| `nanmin`(array[, axis, keepdims, initial, ...])                | Returns the minimum value, treating NaN values as missing.                                        |
| `moment`(x, n[, weight, axis, keepdims, mask_identity, ...])   | Computes the <br/><br/>```<br/>`<br/>```<br/><br/>n\`th moment over one or all levels of nesting. |
| `nan_to_none`(array, \*[, highlevel, behavior, attrs])         | Converts NaN ("not a number") into None, i.e. missing values with option-type.                    |
| `nan_to_num`(array[, copy, nan, posinf, neginf, ...])          | Replaces NaN and infinite values with finite numbers in floating-point arrays.                    |
| `num`(array[, axis, highlevel, behavior, attrs])               | Returns the number of elements at a given axis depth.                                             |
| `ones_like`(array, \*[, dtype, including_unknown, ...])        | Returns an array with the same structure as the input, filled with ones.                          |
| `pad_none`(array, target[, axis, clip, highlevel, ...])        | Increases the lengths of lists to a target length by adding None values.                          |
| `parameters`(array)                                            | Returns the parameters dict of the outermost layout node.                                         |
| `nanprod`(array[, axis, keepdims, mask_identity, ...])         | Multiplies an array's elements, treating NaN values as missing.                                   |
| `prod`(array[, axis, keepdims, mask_identity, ...])            | Multiplies an array's elements over one or all levels of nesting.                                 |
| `ptp`(array[, axis, keepdims, mask_identity, highlevel, ...])  | Returns the range of values over one or all levels of nesting.                                    |
| `ravel`(array, \*[, highlevel, behavior, attrs])               | Returns an array with all levels of nesting removed.                                              |
| `real`(array[, highlevel, behavior, attrs])                    | Returns the real components of the given array elements.                                          |
| `round`(array[, decimals, out, highlevel, behavior, attrs])    | Rounds each array element to the given number of decimals.                                        |
| `run_lengths`(array, \*[, highlevel, behavior, attrs])         | Returns the lengths of runs of identical values at the deepest level.                             |
| `singletons`(array[, axis, highlevel, behavior, attrs])        | Wraps each value in a length-1 list, or an empty list for each missing value.                     |
| `softmax`(x[, axis, keepdims, mask_identity, highlevel, ...])  | Computes the softmax over the innermost level of nesting.                                         |
| `sort`(array[, axis, ascending, stable, highlevel, ...])       | Returns an array with elements sorted along an axis.                                              |
| `nanstd`(x[, weight, ddof, axis, keepdims, ...])               | Computes the standard deviation, treating NaN values as missing.                                  |
| `std`(x[, weight, ddof, axis, keepdims, mask_identity, ...])   | Computes the standard deviation over one or all levels of nesting.                                |
| `strings_astype`(array, to, \*[, highlevel, behavior, attrs])  | Converts all strings in the array to a new type, leaving the structure untouched.                 |
| `nansum`(array[, axis, keepdims, mask_identity, ...])          | Sums an array's elements, treating NaN values as missing.                                         |
| `sum`(array[, axis, keepdims, mask_identity, highlevel, ...])  | Sums an array's elements over one or all levels of nesting.                                       |
| `to_arrow`(array, \*[, list_to32, string_to32, ...])           | Converts an Awkward Array into an Apache Arrow array.                                             |
| `to_arrow_table`(array, \*[, list_to32, string_to32, ...])     | Converts an Awkward Array into an Apache Arrow table.                                             |
| `to_backend`(array, backend, \*[, highlevel, behavior, attrs]) | Returns an array on a different backend (kernel set).                                             |
| `to_buffers`(array[, container, buffer_key, form_key, ...])    | Decomposes an Awkward Array into a Form, length, and memory buffers.                              |
| `to_cudf`(array)                                               | Converts an Awkward Array into a cuDF Series.                                                     |
| `to_cupy`(array)                                               | Converts an Awkward Array into a CuPy array, if possible.                                         |
| `to_dataframe`(array, \*[, how, levelname, anonymous])         | Converts an Awkward Array into a pandas DataFrame.                                                |
| `to_feather`(array, destination, \*[, list_to32, ...])         | Writes an Awkward Array to a Feather file (through pyarrow).                                      |
| `to_jax`(array)                                                | Converts an Awkward Array into a JAX Array, if possible.                                          |
| `to_json`(array[, file, line_delimited, ...])                  | Converts an Awkward Array into JSON text, as a string or to a file.                               |
| `to_layout`(array, \*[, allow_record, allow_unknown, ...])     | Converts data into a low-level layout object.                                                     |
| `to_list`(array)                                               | Converts an Awkward Array into Python objects.                                                    |
| `to_numpy`(array, \*[, allow_missing])                         | Converts an Awkward Array into a NumPy array, if possible.                                        |
| `to_packed`(array, \*[, highlevel, behavior, attrs])           | Packs an array's inner structure and materializes its virtual buffers.                            |
| `to_parquet`(array, destination, \*[, list_to32, ...])         | Writes an Awkward Array to a Parquet file (through pyarrow).                                      |
| `to_parquet_dataset`(directory[, filenames, storage_options])  | Creates a \_common_metadata and a \_metadata in a directory of Parquet files.                     |
| `to_parquet_row_groups`(iterator, destination, \*[, ...])      | Writes a sequence of Awkward Arrays to a Parquet file as row groups.                              |
| `to_raggedtensor`(array)                                       | Converts an Awkward Array into a TensorFlow RaggedTensor, if possible.                            |
| `to_rdataframe`(arrays, \*[, flatlist_as_rvec])                | Converts an Awkward Array into ROOT RDataFrame columns.                                           |
| `to_regular`(array[, axis, highlevel, behavior, attrs])        | Converts one or all variable-length axes into regular ones, if possible.                          |
| `to_safetensors`(array, destination, \*[, ...])                | Writes an Awkward Array to a safetensors file.                                                    |
| `to_tensorflow`(array)                                         | Converts an Awkward Array into a TensorFlow Tensor, if possible.                                  |
| `to_torch`(array)                                              | Converts an Awkward Array into a PyTorch Tensor, if possible.                                     |
| `transform`(transformation, array, \*more_arrays[, ...])       | Applies a transformation function to every node of one or more arrays.                            |
| `type`(array, \*[, behavior])                                  | Returns the high-level type of an array as a Type object.                                         |
| `unflatten`(array, counts[, axis, highlevel, behavior, ...])   | Returns an array with an additional level of nesting.                                             |
| `unzip`(array, \*[, how, highlevel, behavior, attrs])          | Splits records or tuples into a tuple or dict of arrays, one per field.                           |
| `validity_error`(array, \*[, exception])                       | Returns an error message if the array has a structural error, or empty if valid.                  |
| `values_astype`(array, to, \*[, including_unknown, ...])       | Converts all numbers in the array to a new type, leaving the structure untouched.                 |
| `nanvar`(x[, weight, ddof, axis, keepdims, ...])               | Computes the variance, treating NaN values as missing.                                            |
| `var`(x[, weight, ddof, axis, keepdims, mask_identity, ...])   | Computes the variance over one or all levels of nesting.                                          |
| `where`(condition, \*args[, mergebool, highlevel, ...])        | Selects elements from x or y by a condition, or finds where it is True.                           |
| `with_field`(array, what[, where, highlevel, behavior, ...])   | Returns an array or record with a new field added, or an existing field replaced.                 |
| `with_name`(array, name, \*[, highlevel, behavior, attrs])     | Returns an array or record with the \_\_record_\_ parameter set to the given name.                |
| `with_named_axis`(array, named_axis, \*[, highlevel, ...])     | Returns an array with named axes attached.                                                        |
| `with_parameter`(array, parameter, value, \*[, ...])           | Returns an array with the given parameter set on the outermost layout node.                       |
| `without_field`(array, where, \*[, highlevel, behavior, ...])  | Returns an array or record with the named field removed.                                          |
| `without_named_axis`(array, \*[, highlevel, behavior, attrs])  | Returns an array with named axes removed.                                                         |
| `without_parameters`(array, \*[, highlevel, behavior, attrs])  | Returns an array with all parameters removed from every layout node.                              |
| `zeros_like`(array, \*[, dtype, including_unknown, ...])       | Returns an array with the same structure as the input, filled with zeros.                         |
| `zip`(arrays[, depth_limit, parameters, with_name, ...])       | Combines arrays into records or tuples, broadcasting them together.                               |
| `zip_no_broadcast`(arrays, \*[, parameters, with_name, ...])   | Combines arrays into a collection of records or tuples without broadcasting.                      |

## Package Contents

### awkward.behavior *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)*

### awkward.\_\_all_\_
