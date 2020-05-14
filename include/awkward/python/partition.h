// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_PARTITION_H_
#define AWKWARDPY_PARTITION_H_

#include <string>

#include <pybind11/pybind11.h>

#include "awkward/partition/PartitionedArray.h"
#include "awkward/partition/IrregularlyPartitionedArray.h"

namespace py = pybind11;
namespace ak = awkward;

/// @brief Makes an abstract PartitionedArray class in Python that mirrors the
/// one in C++.
py::class_<ak::PartitionedArray, std::shared_ptr<ak::PartitionedArray>>
  make_PartitionedArray(const py::handle& m, const std::string& name);

/// @brief Makes an IrregularlyPartitionedArray in Python that mirrors the one
/// in C++.
py::class_<ak::IrregularlyPartitionedArray,
           std::shared_ptr<ak::IrregularlyPartitionedArray>,
           ak::PartitionedArray>
  make_IrregularlyPartitionedArray(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_PARTITION_H_
