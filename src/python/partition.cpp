// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/partition.cpp", line)

#include "awkward/python/content.h"

#include "awkward/python/partition.h"

// template <typename T>
// ak::Iterator
// iter(const T& self) {
//   return ak::Iterator(self.shallow_copy());
// }

template <typename T>
std::string
tojson_string(const T& self,
              bool pretty,
              const py::object& maxdecimals) {
  return self.tojson(pretty,
                     check_maxdecimals(maxdecimals));
}

template <typename T>
void
tojson_file(const T& self,
            const std::string& destination,
            bool pretty,
            py::object maxdecimals,
            int64_t buffersize) {
#ifdef _MSC_VER
  FILE* file;
  if (fopen_s(&file, destination.c_str(), "wb") != 0) {
#else
  FILE* file = fopen(destination.c_str(), "wb");
  if (file == nullptr) {
#endif
    throw std::invalid_argument(
      std::string("file \"") + destination
      + std::string("\" could not be opened for writing") + FILENAME(__LINE__));
  }
  try {
    self.tojson(file,
                pretty,
                check_maxdecimals(maxdecimals),
                buffersize);
  }
  catch (...) {
    fclose(file);
    throw;
  }
  fclose(file);
}

template <typename T>
py::class_<T, std::shared_ptr<T>, ak::PartitionedArray>
partitionedarray_methods(py::class_<T, std::shared_ptr<T>,
                         ak::PartitionedArray>& x) {
  return x.def("__repr__", &repr<T>)
          .def("__len__", &len<T>)
          .def_property_readonly("partitions", [](const T& self) -> py::object {
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
          .def("repartition", [](const T& self,
                                 const std::vector<int64_t>& stops)
                              -> ak::PartitionedArrayPtr {
            return self.repartition(stops);
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
          .def("getitem_at", [](const T& self, int64_t at) -> py::object {
            return box(self.getitem_at(at));
          })
          .def("getitem_range", [](const T& self,
                                   py::object start,
                                   py::object stop,
                                   py::object step) -> ak::PartitionedArrayPtr {
            int64_t intstart = ak::Slice::none();
            int64_t intstop = ak::Slice::none();
            int64_t intstep = ak::Slice::none();
            if (!start.is(py::none())) {
              intstart = start.cast<int64_t>();
            }
            if (!stop.is(py::none())) {
              intstop = stop.cast<int64_t>();
            }
            if (!step.is(py::none())) {
              intstep = step.cast<int64_t>();
            }
            return self.getitem_range(intstart, intstop, intstep);
          })
          .def("copy_to",
               [](const T& self, const std::string& ptr_lib) -> ak::PartitionedArrayPtr {
               if (ptr_lib == "cpu") {
                 return self.copy_to(ak::kernel::lib::cpu);
               }
               else if (ptr_lib == "cuda") {
                 return self.copy_to(ak::kernel::lib::cuda);
               }
               else {
                 throw std::invalid_argument(
                   std::string("specify 'cpu' or 'cuda'") + FILENAME(__LINE__));
               }
          })

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
      .def(py::init([](const std::vector<ak::ContentPtr>& partitions)
                    -> ak::IrregularlyPartitionedArray {
        int64_t total_length = 0;
        std::vector<int64_t> stops;
        for (auto p : partitions) {
          total_length += p.get()->length();
          stops.push_back(total_length);
        }
        return ak::IrregularlyPartitionedArray(partitions, stops);
      }))
      .def_property_readonly("stops", &ak::IrregularlyPartitionedArray::stops)

  );
}
