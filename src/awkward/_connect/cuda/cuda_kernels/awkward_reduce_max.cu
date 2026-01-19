// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code) = args
//     # 1. Check for int64 and promote to float64
//     if fromptr.dtype == cupy.int64:
//         data_to_reduce = fromptr[:lenparents].astype(cupy.float64)
//         # We must also ensure the identity is promoted
//         working_identity = cupy.array(identity, dtype=cupy.float64)
//         # Create a temporary float64 buffer for the atomic reduction
//         temp_out = cupy.full((outlength,), working_identity, dtype=cupy.float64)
//         cupy.maximum.at(temp_out, parents[:lenparents], data_to_reduce)
//         # 2. Cast back to the original int64 toptr
//         toptr[:outlength] = temp_out.astype(cupy.int64)
//     else:
//         # Standard logic for supported types (int32, float32, etc.)
//         toptr[:outlength] = identity
//         cupy.maximum.at(toptr[:outlength], parents[:lenparents], fromptr[:lenparents])
// END PYTHON
