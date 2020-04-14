// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/python/content.h"

#include "awkward/python/partition.h"

// template <typename T>
// ak::Iterator
// iter(const T& self) {
//   return ak::Iterator(self.shallow_copy());
// }

template <typename T>
py::class_<T, std::shared_ptr<T>, ak::PartitionedArray>
partitionedarray_methods(py::class_<T, std::shared_ptr<T>,
                         ak::PartitionedArray>& x) {
  return x.def("__repr__", &repr<T>)
          .def("__len__", &len<T>)
          .def("partitions", [](const T& self) -> py::object {
            py::list out;
            std::vector<ak::ContentPtr> partitions = self.partitions();
            for (auto item : partitions) {
              out.append(box(item));
            }
            return out;
          })
          .def_property_readonly("numpartitions", &T::numpartitions)
          .def("partition", &T::partition)
          .def("start", &T::start)
          .def("stop", &T::stop)
          .def("partitionid_index_at", [](const T& self, int64_t at)
                                       -> py::object {
            int64_t partitionid;
            int64_t index;
            self.partitionid_index_at(at, partitionid, index);
            py::tuple out(2);
            out[0] = py::cast(partitionid);
            out[1] = py::cast(index);
            return out;
          })
          .def("tojson",
               &tojson_string<T>,
               py::arg("pretty") = false,
               py::arg("maxdecimals") = py::none())
          .def("tojson",
               &tojson_file<T>,
               py::arg("destination"),
               py::arg("pretty") = false,
               py::arg("maxdecimals") = py::none(),
               py::arg("buffersize") = 65536)
          .def("getitem_at", &T::getitem_at)
          .def("getitem_range", &T::getitem_range)

  ;
}

////////// PartitionedArray

py::class_<ak::PartitionedArray, std::shared_ptr<ak::PartitionedArray>>
make_PartitionedArray(const py::handle& m, const std::string& name) {
  return py::class_<ak::PartitionedArray,
                    std::shared_ptr<ak::PartitionedArray>>(m, name.c_str());
}

////////// IrregularlyPartitionedArray

py::class_<ak::IrregularlyPartitionedArray,
           std::shared_ptr<ak::IrregularlyPartitionedArray>,
           ak::PartitionedArray>
make_IrregularlyPartitionedArray(const py::handle& m, const std::string& name) {
  return partitionedarray_methods(
          py::class_<ak::IrregularlyPartitionedArray,
          std::shared_ptr<ak::IrregularlyPartitionedArray>,
          ak::PartitionedArray>(m, name.c_str())
      .def(py::init([](const std::vector<ak::ContentPtr>& partitions,
                       const std::vector<int64_t>& stops)
                    -> ak::IrregularlyPartitionedArray {
        return ak::IrregularlyPartitionedArray(partitions, stops);
      }), py::arg("partitions"), py::arg("stops"))
  );
}
