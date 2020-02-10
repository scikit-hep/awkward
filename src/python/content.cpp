// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/numpy.h>

#include "awkward/python/identities.h"
#include "awkward/python/boxing.h"
#include "awkward/python/util.h"

#include "awkward/python/content.h"

/////////////////////////////////////////////////////////////// Iterator

py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>> make_Iterator(py::handle m, std::string name) {
  auto next = [](ak::Iterator& iterator) -> py::object {
    if (iterator.isdone()) {
      throw py::stop_iteration();
    }
    return box(iterator.next());
  };

  return (py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>>(m, name.c_str())
      .def(py::init([](py::object content) -> ak::Iterator {
        return ak::Iterator(unbox_content(content));
      }))
      .def("__repr__", &ak::Iterator::tostring)
      .def("__next__", next)
      .def("next", next)
      .def("__iter__", [](py::object self) -> py::object { return self; })
  );
}

/////////////////////////////////////////////////////////////// Content

py::class_<ak::Content, std::shared_ptr<ak::Content>> make_Content(py::handle m, std::string name) {
  return py::class_<ak::Content, std::shared_ptr<ak::Content>>(m, name.c_str());
}

template <typename T>
py::tuple identity(const T& self) {
  if (self.identities().get() == nullptr) {
    throw std::invalid_argument(self.classname() + std::string(" instance has no associated identities (use 'setidentities' to assign one to the array it is in)"));
  }
  ak::Identities::FieldLoc fieldloc = self.identities().get()->fieldloc();
  if (self.isscalar()) {
    py::tuple out((size_t)(self.identities().get()->width()) + fieldloc.size());
    size_t j = 0;
    for (int64_t i = 0;  i < self.identities().get()->width();  i++) {
      out[j] = py::cast(self.identities().get()->value(0, i));
      j++;
      for (auto pair : fieldloc) {
        if (pair.first == i) {
          out[j] = py::cast(pair.second);
          j++;
        }
      }
    }
    return out;
  }
  else {
    py::tuple out((size_t)(self.identities().get()->width() - 1) + fieldloc.size());
    size_t j = 0;
    for (int64_t i = 0;  i < self.identities().get()->width();  i++) {
      if (i < self.identities().get()->width() - 1) {
        out[j] = py::cast(self.identities().get()->value(0, i));
        j++;
      }
      for (auto pair : fieldloc) {
        if (pair.first == i) {
          out[j] = py::cast(pair.second);
          j++;
        }
      }
    }
    return out;
  }
}

template <typename T>
ak::Iterator iter(T& self) {
  return ak::Iterator(self.shallow_copy());
}

template <typename T>
py::dict getparameters(T& self) {
  return parameters2dict(self.parameters());
}

template <typename T>
py::object parameter(T& self, std::string& key) {
  std::string cppvalue = self.parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
py::object purelist_parameter(T& self, std::string& key) {
  std::string cppvalue = self.purelist_parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
void setparameters(T& self, py::object parameters) {
  self.setparameters(dict2parameters(parameters));
}

template <typename T>
void setparameter(T& self, std::string& key, py::object value) {
  py::object valuestr = py::module::import("json").attr("dumps")(value);
  self.setparameter(key, valuestr.cast<std::string>());
}

int64_t check_maxdecimals(py::object maxdecimals) {
  if (maxdecimals.is(py::none())) {
    return -1;
  }
  try {
    return maxdecimals.cast<int64_t>();
  }
  catch (py::cast_error err) {
    throw std::invalid_argument("maxdecimals must be None or an integer");
  }
}

template <typename T>
std::string tojson_string(T& self, bool pretty, py::object maxdecimals) {
  return self.tojson(pretty, check_maxdecimals(maxdecimals));
}

template <typename T>
void tojson_file(T& self, std::string destination, bool pretty, py::object maxdecimals, int64_t buffersize) {
#ifdef _MSC_VER
  FILE* file;
  if (fopen_s(&file, destination.c_str(), "wb") != 0) {
#else
  FILE* file = fopen(destination.c_str(), "wb");
  if (file == nullptr) {
#endif
    throw std::invalid_argument(std::string("file \"") + destination + std::string("\" could not be opened for writing"));
  }
  try {
    self.tojson(file, pretty, check_maxdecimals(maxdecimals), buffersize);
  }
  catch (...) {
    fclose(file);
    throw;
  }
  fclose(file);
}

template <typename T>
py::class_<T, std::shared_ptr<T>, ak::Content> content_methods(py::class_<T, std::shared_ptr<T>, ak::Content>& x) {
  return x.def("__repr__", &repr<T>)
          .def_property("identities", [](T& self) -> py::object { return box(self.identities()); }, [](T& self, py::object identities) -> void { self.setidentities(unbox_identities_none(identities)); })
          .def("setidentities", [](T& self, py::object identities) -> void {
           self.setidentities(unbox_identities_none(identities));
          })
          .def("setidentities", [](T& self) -> void {
            self.setidentities();
          })
          .def_property("parameters", &getparameters<T>, &setparameters<T>)
          .def("setparameter", &setparameter<T>)
          .def("parameter", &parameter<T>)
          .def("purelist_parameter", &purelist_parameter<T>)
          .def_property_readonly("type", [](T& self) -> py::object {
            return box(self.type());
          })
          .def("astype", [](T& self, std::shared_ptr<ak::Type>& type) -> py::object {
            return box(self.astype(type));
          })
          .def("__len__", &len<T>)
          .def("__getitem__", &getitem<T>)
          .def("__iter__", &iter<T>)
          .def("tojson", &tojson_string<T>, py::arg("pretty") = false, py::arg("maxdecimals") = py::none())
          .def("tojson", &tojson_file<T>, py::arg("destination"), py::arg("pretty") = false, py::arg("maxdecimals") = py::none(), py::arg("buffersize") = 65536)
          .def_property_readonly("nbytes", &T::nbytes)
          .def("deep_copy", &T::deep_copy, py::arg("copyarrays") = true, py::arg("copyindexes") = true, py::arg("copyidentities") = true)
          .def_property_readonly("identity", &identity<T>)
          .def_property_readonly("numfields", &T::numfields)
          .def("fieldindex", &T::fieldindex)
          .def("key", &T::key)
          .def("haskey", &T::haskey)
          .def("keys", &T::keys)
          .def_property_readonly("purelist_isregular", &T::purelist_isregular)
          .def_property_readonly("purelist_depth", &T::purelist_depth)
          .def("getitem_nothing", &T::getitem_nothing)

          // operations
          .def("count", [](T& self, int64_t axis) -> py::object {
            return box(self.count(axis));
          }, py::arg("axis") = 0)
          .def("flatten", [](T& self, int64_t axis) -> py::object {
            return box(self.flatten(axis));
          }, py::arg("axis") = 0)
          .def("mergeable", [](T& self, py::object other, bool mergebool) -> bool {
            return self.mergeable(unbox_content(other), mergebool);
          }, py::arg("other"), py::arg("mergebool") = false)
          .def("merge", [](T& self, py::object other) -> py::object {
            return box(self.merge(unbox_content(other)));
          })
          .def("merge_as_union", [](T& self, py::object other) -> py::object {
            return box(self.merge_as_union(unbox_content(other)));
          })

  ;
}

/////////////////////////////////////////////////////////////// EmptyArray

py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content> make_EmptyArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content>(m, name.c_str())
      .def(py::init([](py::object identities, py::object parameters) -> ak::EmptyArray {
        return ak::EmptyArray(unbox_identities_none(identities), dict2parameters(parameters));
      }), py::arg("identities") = py::none(), py::arg("parameters") = py::none())
  );
}

/////////////////////////////////////////////////////////////// IndexedArray

template <typename T, bool ISOPTION>
py::class_<ak::IndexedArrayOf<T, ISOPTION>, std::shared_ptr<ak::IndexedArrayOf<T, ISOPTION>>, ak::Content> make_IndexedArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::IndexedArrayOf<T, ISOPTION>, std::shared_ptr<ak::IndexedArrayOf<T, ISOPTION>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& index, py::object content, py::object identities, py::object parameters) -> ak::IndexedArrayOf<T, ISOPTION> {
        return ak::IndexedArrayOf<T, ISOPTION>(unbox_identities_none(identities), dict2parameters(parameters), index, std::shared_ptr<ak::Content>(unbox_content(content)));
      }), py::arg("index"), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("index", &ak::IndexedArrayOf<T, ISOPTION>::index)
      .def_property_readonly("content", &ak::IndexedArrayOf<T, ISOPTION>::content)
      .def_property_readonly("isoption", &ak::IndexedArrayOf<T, ISOPTION>::isoption)
      .def("project", [](ak::IndexedArrayOf<T, ISOPTION>& self, py::object mask) {
        if (mask.is(py::none())) {
          return box(self.project());
        }
        else {
          return box(self.project(mask.cast<ak::Index8>()));
        }
      }, py::arg("mask") = py::none())
      .def("bytemask", &ak::IndexedArrayOf<T, ISOPTION>::bytemask)
      .def("simplify", [](ak::IndexedArrayOf<T, ISOPTION>& self) {
        return box(self.simplify());
      })
  );
}

template py::class_<ak::IndexedArray32, std::shared_ptr<ak::IndexedArray32>, ak::Content> make_IndexedArrayOf(py::handle m, std::string name);
template py::class_<ak::IndexedArrayU32, std::shared_ptr<ak::IndexedArrayU32>, ak::Content> make_IndexedArrayOf(py::handle m, std::string name);
template py::class_<ak::IndexedArray64, std::shared_ptr<ak::IndexedArray64>, ak::Content> make_IndexedArrayOf(py::handle m, std::string name);
template py::class_<ak::IndexedOptionArray32, std::shared_ptr<ak::IndexedOptionArray32>, ak::Content> make_IndexedArrayOf(py::handle m, std::string name);
template py::class_<ak::IndexedOptionArray64, std::shared_ptr<ak::IndexedOptionArray64>, ak::Content> make_IndexedArrayOf(py::handle m, std::string name);

/////////////////////////////////////////////////////////////// ListArray

template <typename T>
py::class_<ak::ListArrayOf<T>, std::shared_ptr<ak::ListArrayOf<T>>, ak::Content> make_ListArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::ListArrayOf<T>, std::shared_ptr<ak::ListArrayOf<T>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& starts, ak::IndexOf<T>& stops, py::object content, py::object identities, py::object parameters) -> ak::ListArrayOf<T> {
        return ak::ListArrayOf<T>(unbox_identities_none(identities), dict2parameters(parameters), starts, stops, unbox_content(content));
      }), py::arg("starts"), py::arg("stops"), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("starts", &ak::ListArrayOf<T>::starts)
      .def_property_readonly("stops", &ak::ListArrayOf<T>::stops)
      .def_property_readonly("content", &ak::ListArrayOf<T>::content)
      .def("compact_offsets64", &ak::ListArrayOf<T>::compact_offsets64)
      .def("broadcast_tooffsets64", &ak::ListArrayOf<T>::broadcast_tooffsets64)
      .def("toRegularArray", &ak::ListArrayOf<T>::toRegularArray)
  );
}

template py::class_<ak::ListArray32, std::shared_ptr<ak::ListArray32>, ak::Content> make_ListArrayOf(py::handle m, std::string name);
template py::class_<ak::ListArrayU32, std::shared_ptr<ak::ListArrayU32>, ak::Content> make_ListArrayOf(py::handle m, std::string name);
template py::class_<ak::ListArray64, std::shared_ptr<ak::ListArray64>, ak::Content> make_ListArrayOf(py::handle m, std::string name);

/////////////////////////////////////////////////////////////// ListOffsetArray

template <typename T>
py::class_<ak::ListOffsetArrayOf<T>, std::shared_ptr<ak::ListOffsetArrayOf<T>>, ak::Content> make_ListOffsetArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::ListOffsetArrayOf<T>, std::shared_ptr<ak::ListOffsetArrayOf<T>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& offsets, py::object content, py::object identities, py::object parameters) -> ak::ListOffsetArrayOf<T> {
        return ak::ListOffsetArrayOf<T>(unbox_identities_none(identities), dict2parameters(parameters), offsets, std::shared_ptr<ak::Content>(unbox_content(content)));
      }), py::arg("offsets"), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("starts", &ak::ListOffsetArrayOf<T>::starts)
      .def_property_readonly("stops", &ak::ListOffsetArrayOf<T>::stops)
      .def_property_readonly("offsets", &ak::ListOffsetArrayOf<T>::offsets)
      .def_property_readonly("content", &ak::ListOffsetArrayOf<T>::content)
      .def("compact_offsets64", &ak::ListOffsetArrayOf<T>::compact_offsets64)
      .def("broadcast_tooffsets64", &ak::ListOffsetArrayOf<T>::broadcast_tooffsets64)
      .def("toRegularArray", &ak::ListOffsetArrayOf<T>::toRegularArray)
  );
}

template py::class_<ak::ListOffsetArray32, std::shared_ptr<ak::ListOffsetArray32>, ak::Content> make_ListOffsetArrayOf(py::handle m, std::string name);
template py::class_<ak::ListOffsetArrayU32, std::shared_ptr<ak::ListOffsetArrayU32>, ak::Content> make_ListOffsetArrayOf(py::handle m, std::string name);
template py::class_<ak::ListOffsetArray64, std::shared_ptr<ak::ListOffsetArray64>, ak::Content> make_ListOffsetArrayOf(py::handle m, std::string name);

/////////////////////////////////////////////////////////////// NumpyArray

py::class_<ak::NumpyArray, std::shared_ptr<ak::NumpyArray>, ak::Content> make_NumpyArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::NumpyArray, std::shared_ptr<ak::NumpyArray>, ak::Content>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](ak::NumpyArray& self) -> py::buffer_info {
        return py::buffer_info(
          self.byteptr(),
          self.itemsize(),
          self.format(),
          self.ndim(),
          self.shape(),
          self.strides());
      })

      .def(py::init([](py::array array, py::object identities, py::object parameters) -> ak::NumpyArray {
        py::buffer_info info = array.request();
        if (info.ndim == 0) {
          throw std::invalid_argument("NumpyArray must not be scalar; try array.reshape(1)");
        }
        if (info.shape.size() != info.ndim  ||  info.strides.size() != info.ndim) {
          throw std::invalid_argument("NumpyArray len(shape) != ndim or len(strides) != ndim");
        }
        return ak::NumpyArray(unbox_identities_none(identities), dict2parameters(parameters), std::shared_ptr<void>(
          reinterpret_cast<void*>(info.ptr), pyobject_deleter<void>(array.ptr())),
          info.shape,
          info.strides,
          0,
          info.itemsize,
          info.format);
      }), py::arg("array"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("shape", &ak::NumpyArray::shape)
      .def_property_readonly("strides", &ak::NumpyArray::strides)
      .def_property_readonly("itemsize", &ak::NumpyArray::itemsize)
      .def_property_readonly("format", &ak::NumpyArray::format)
      .def_property_readonly("ndim", &ak::NumpyArray::ndim)
      .def_property_readonly("isscalar", &ak::NumpyArray::isscalar)
      .def_property_readonly("isempty", &ak::NumpyArray::isempty)
      .def("toRegularArray", &ak::NumpyArray::toRegularArray)

      .def_property_readonly("iscontiguous", &ak::NumpyArray::iscontiguous)
      .def("contiguous", &ak::NumpyArray::contiguous)
      .def("become_contiguous", &ak::NumpyArray::become_contiguous)
  );
}

/////////////////////////////////////////////////////////////// RecordArray

py::class_<ak::Record, std::shared_ptr<ak::Record>> make_Record(py::handle m, std::string name) {
  return py::class_<ak::Record, std::shared_ptr<ak::Record>>(m, name.c_str())
      .def(py::init([](std::shared_ptr<ak::RecordArray> recordarray, int64_t at) -> ak::Record {
        return ak::Record(recordarray, at);
      }), py::arg("recordarray"), py::arg("at"))
      .def("__repr__", &repr<ak::Record>)
      .def_property_readonly("identities", [](ak::Record& self) -> py::object { return box(self.identities()); })
      .def("__getitem__", &getitem<ak::Record>)
      .def_property_readonly("type", [](ak::Record& self) -> py::object {
        return box(self.type());
      })
      .def_property("parameters", &getparameters<ak::Record>, &setparameters<ak::Record>)
      .def("setparameter", &setparameter<ak::Record>)
      .def("parameter", &parameter<ak::Record>)
      .def("purelist_parameter", &purelist_parameter<ak::Record>)
      .def_property_readonly("type", [](ak::Record& self) -> py::object {
        return box(self.type());
      })
      .def("astype", [](ak::Record& self, std::shared_ptr<ak::Type>& type) -> py::object {
        return box(self.astype(type));
      })
      .def("tojson", &tojson_string<ak::Record>, py::arg("pretty") = false, py::arg("maxdecimals") = py::none())
      .def("tojson", &tojson_file<ak::Record>, py::arg("destination"), py::arg("pretty") = false, py::arg("maxdecimals") = py::none(), py::arg("buffersize") = 65536)

      .def_property_readonly("array", [](ak::Record& self) -> py::object { return box(self.array()); })
      .def_property_readonly("at", &ak::Record::at)
      .def_property_readonly("istuple", &ak::Record::istuple)
      .def_property_readonly("numfields", &ak::Record::numfields)
      .def("fieldindex", &ak::Record::fieldindex)
      .def("key", &ak::Record::key)
      .def("haskey", &ak::Record::haskey)
      .def("keys", &ak::Record::keys)
      .def("field", [](ak::Record& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("field", [](ak::Record& self, std::string key) -> py::object {
        return box(self.field(key));
      })
      .def("fields", [](ak::Record& self) -> py::object {
        py::list out;
        for (auto item : self.fields()) {
          out.append(box(item));
        }
        return out;
      })
      .def("fielditems", [](ak::Record& self) -> py::object {
        py::list out;
        for (auto item : self.fielditems()) {
          py::str key(item.first);
          py::object val(box(item.second));
          py::tuple pair(2);
          pair[0] = key;
          pair[1] = val;
          out.append(pair);
        }
        return out;
      })
      .def_property_readonly("astuple", [](ak::Record& self) -> py::object {
        return box(self.astuple());
      })
     .def_property_readonly("identity", &identity<ak::Record>)

  ;
}

ak::RecordArray iterable_to_RecordArray(py::iterable contents, py::object keys, py::object identities, py::object parameters) {
  std::vector<std::shared_ptr<ak::Content>> out;
  for (auto x : contents) {
    out.push_back(unbox_content(x));
  }
  if (out.empty()) {
    throw std::invalid_argument("construct RecordArrays without fields using RecordArray(length) where length is an integer");
  }
  if (keys.is(py::none())) {
    return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, std::shared_ptr<ak::util::RecordLookup>(nullptr));
  }
  else {
    std::shared_ptr<ak::util::RecordLookup> recordlookup = std::make_shared<ak::util::RecordLookup>();
    for (auto x : keys.cast<py::iterable>()) {
      recordlookup.get()->push_back(x.cast<std::string>());
    }
    if (out.size() != recordlookup.get()->size()) {
      throw std::invalid_argument("if provided, 'keys' must have the same length as 'types'");
    }
    return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, recordlookup);
  }
}

py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content> make_RecordArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content>(m, name.c_str())
      .def(py::init([](py::dict contents, py::object identities, py::object parameters) -> ak::RecordArray {
        std::shared_ptr<ak::util::RecordLookup> recordlookup = std::make_shared<ak::util::RecordLookup>();
        std::vector<std::shared_ptr<ak::Content>> out;
        for (auto x : contents) {
          std::string key = x.first.cast<std::string>();
          recordlookup.get()->push_back(key);
          out.push_back(unbox_content(x.second));
        }
        if (out.empty()) {
          throw std::invalid_argument("construct RecordArrays without fields using RecordArray(length) where length is an integer");
        }
        return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, recordlookup);
      }), py::arg("contents"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())
      .def(py::init(&iterable_to_RecordArray), py::arg("contents"), py::arg("keys") = py::none(), py::arg("identities") = py::none(), py::arg("parameters") = py::none())
      .def(py::init([](int64_t length, bool istuple, py::object identities, py::object parameters) -> ak::RecordArray {
        return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), length, istuple);
      }), py::arg("length"), py::arg("istuple") = false, py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("istuple", &ak::RecordArray::istuple)
      .def_property_readonly("contents", &ak::RecordArray::contents)
      .def("setitem_field", [](ak::RecordArray& self, py::object where, py::object what) -> py::object {
        std::shared_ptr<ak::Content> mywhat = unbox_content(what);
        if (where.is(py::none())) {
          return box(self.setitem_field(self.numfields(), mywhat));
        }
        else {
          try {
            std::string mywhere = where.cast<std::string>();
            return box(self.setitem_field(mywhere, mywhat));
          }
          catch (py::cast_error err) {
            try {
              int64_t mywhere = where.cast<int64_t>();
              return box(self.setitem_field(mywhere, mywhat));
            }
            catch (py::cast_error err) {
              throw std::invalid_argument("where must be None, int, or str");
            }
          }
        }
      }, py::arg("where"), py::arg("what"))

      .def("field", [](ak::RecordArray& self, int64_t fieldindex) -> std::shared_ptr<ak::Content> {
        return self.field(fieldindex);
      })
      .def("field", [](ak::RecordArray& self, std::string key) -> std::shared_ptr<ak::Content> {
        return self.field(key);
      })
      .def("fields", [](ak::RecordArray& self) -> py::object {
        py::list out;
        for (auto item : self.fields()) {
          out.append(box(item));
        }
        return out;
      })
      .def("fielditems", [](ak::RecordArray& self) -> py::object {
        py::list out;
        for (auto item : self.fielditems()) {
          py::str key(item.first);
          py::object val(box(item.second));
          py::tuple pair(2);
          pair[0] = key;
          pair[1] = val;
          out.append(pair);
        }
        return out;
      })
      .def_property_readonly("astuple", [](ak::RecordArray& self) -> py::object {
        return box(self.astuple());
      })

  );
}

/////////////////////////////////////////////////////////////// RegularArray

py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content> make_RegularArray(py::handle m, std::string name) {
  return content_methods(py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content>(m, name.c_str())
      .def(py::init([](py::object content, int64_t size, py::object identities, py::object parameters) -> ak::RegularArray {
        return ak::RegularArray(unbox_identities_none(identities), dict2parameters(parameters), std::shared_ptr<ak::Content>(unbox_content(content)), size);
      }), py::arg("content"), py::arg("size"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("size", &ak::RegularArray::size)
      .def_property_readonly("content", &ak::RegularArray::content)
      .def("compact_offsets64", &ak::RegularArray::compact_offsets64)
      .def("broadcast_tooffsets64", &ak::RegularArray::broadcast_tooffsets64)
  );
}

/////////////////////////////////////////////////////////////// UnionArray

template <typename T, typename I>
py::class_<ak::UnionArrayOf<T, I>, std::shared_ptr<ak::UnionArrayOf<T, I>>, ak::Content> make_UnionArrayOf(py::handle m, std::string name) {
  return content_methods(py::class_<ak::UnionArrayOf<T, I>, std::shared_ptr<ak::UnionArrayOf<T, I>>, ak::Content>(m, name.c_str())
      .def(py::init([](ak::IndexOf<T>& tags, ak::IndexOf<I>& index, py::iterable contents, py::object identities, py::object parameters) -> ak::UnionArrayOf<T, I> {
        std::vector<std::shared_ptr<ak::Content>> out;
        for (auto content : contents) {
          out.push_back(std::shared_ptr<ak::Content>(unbox_content(content)));
        }
        return ak::UnionArrayOf<T, I>(unbox_identities_none(identities), dict2parameters(parameters), tags, index, out);
      }), py::arg("tags"), py::arg("index"), py::arg("contents"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_static("regular_index", &ak::UnionArrayOf<T, I>::regular_index)
      .def_property_readonly("tags", &ak::UnionArrayOf<T, I>::tags)
      .def_property_readonly("index", &ak::UnionArrayOf<T, I>::index)
      .def_property_readonly("contents", &ak::UnionArrayOf<T, I>::contents)
      .def_property_readonly("numcontents", &ak::UnionArrayOf<T, I>::numcontents)
      .def("content", &ak::UnionArrayOf<T, I>::content)
      .def("project", &ak::UnionArrayOf<T, I>::project)
      .def("simplify", [](ak::UnionArrayOf<T, I>& self, bool mergebool) -> py::object {
        return box(self.simplify(mergebool));
      }, py::arg("mergebool") = false)

  );
}

template py::class_<ak::UnionArray8_32, std::shared_ptr<ak::UnionArray8_32>, ak::Content> make_UnionArrayOf(py::handle m, std::string name);
template py::class_<ak::UnionArray8_U32, std::shared_ptr<ak::UnionArray8_U32>, ak::Content> make_UnionArrayOf(py::handle m, std::string name);
template py::class_<ak::UnionArray8_64, std::shared_ptr<ak::UnionArray8_64>, ak::Content> make_UnionArrayOf(py::handle m, std::string name);
