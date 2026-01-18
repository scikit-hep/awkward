// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code) = args
//     toptr[:outlength] = identity
//     cupy.maximum.at(
//        toptr[:outlength],
//        parents[:lenparents],
//        fromptr[:lenparents]
//    )
// END PYTHON
