// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     supported_types = (cupy.int32, cupy.float16, cupy.float32, cupy.float64, cupy.uint32, cupy.uint64)
//     if fromptr.dtype not in supported_types:
//        # Promote to int32 (or float64 if original was float)
//        working_dtype = cupy.int32 if cupy.issubdtype(fromptr.dtype, cupy.integer) else cupy.float32
//        data_to_reduce = fromptr[:lenparents].astype(working_dtype)
//     else:
//        data_to_reduce = fromptr[:lenparents]
//     if toptr.dtype not in supported_types:
//        temp_sum = cupy.zeros(outlength, dtype=data_to_reduce.dtype)
//        cupy.add.at(temp_sum, parents[:lenparents], data_to_reduce)
//        # Cast back to the original requested output type
//        toptr[:outlength] = temp_sum.astype(toptr.dtype)
//     else:
//        toptr[:outlength] = 0
//        cupy.add.at(toptr[:outlength], parents[:lenparents], data_to_reduce)
// END PYTHON
