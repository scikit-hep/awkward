// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/io.cpp", line)

#include <pybind11/numpy.h>
#include <string>

#include "awkward/io/json.h"
#include "awkward/io/uproot.h"

#include "awkward/python/io.h"

namespace ak = awkward;

////////// fromjson

void
make_fromjson(py::module& m, const std::string& name) {
  m.def(name.c_str(),
        [](const std::string& source,
           ak::ArrayBuilder& builder,
           const char* nan_string,
           const char* infinity_string,
           const char* minus_infinity_string,
           int64_t buffersize) -> int64_t {
    return ak::FromJsonString(source.c_str(),
                              builder,
                              nan_string,
                              infinity_string,
                              minus_infinity_string);
  }, py::arg("source"),
     py::arg("builder"),
     py::arg("nan_string") = nullptr,
     py::arg("infinity_string") = nullptr,
     py::arg("minus_infinity_string") = nullptr,
     py::arg("buffersize") = 65536);
}

void
make_fromjsonfile(py::module& m, const std::string& name) {
  m.def(name.c_str(),
        [](const std::string& source,
           ak::ArrayBuilder& builder,
           const char* nan_string,
           const char* infinity_string,
           const char* minus_infinity_string,
           int64_t buffersize) -> int64_t {
#ifdef _MSC_VER
      FILE* file;
      if (fopen_s(&file, source.c_str(), "rb") != 0) {
#else
      FILE* file = fopen(source.c_str(), "rb");
      if (file == nullptr) {
#endif
        throw std::invalid_argument(
          std::string("file \"") + source
          + std::string("\" could not be opened for reading")
          + FILENAME(__LINE__));
      }
      int num = 0;
      try {
        num = ak::FromJsonFile(file,
                         builder,
                         buffersize,
                         nan_string,
                         infinity_string,
                         minus_infinity_string);
      }
      catch (...) {
        fclose(file);
        throw;
      }
      fclose(file);

      return num;
  }, py::arg("source"),
     py::arg("builder"),
     py::arg("nan_string") = nullptr,
     py::arg("infinity_string") = nullptr,
     py::arg("minus_infinity_string") = nullptr,
     py::arg("buffersize") = 65536);
}

class PythonFileLikeObject : public ak::FileLikeObject {
public:
  PythonFileLikeObject(py::object& obj) : obj_(obj) { }

  int64_t read(int64_t num_bytes, char* buffer) {
    // assuming that this is being called from code in which the GIL has been released
    py::gil_scoped_acquire acquire;

    py::object data = obj_.attr("read")(num_bytes);

    if (!PyBytes_Check(data.ptr())) {
      throw py::type_error("obj.read(num_bytes) should return bytes (is the file mode 'rb'?)");
    }

    int64_t num_bytes_read = PyBytes_Size(data.ptr());

    if (num_bytes_read > num_bytes) {
      throw py::type_error("obj.read(num_bytes) returned a larger bytes object than num_bytes");
    }

    std::strncpy(buffer, PyBytes_AsString(data.ptr()), std::min(num_bytes, num_bytes_read));

    py::gil_scoped_release release;

    return num_bytes_read;
  }

private:
  py::object obj_;
};

void
make_fromjsonobj(py::module& m, const std::string& name) {
  m.def(name.c_str(),
        [](py::object& source,
           ak::ArrayBuilder& builder,
           bool read_one,
           int64_t buffersize,
           const char* nan_string,
           const char* posinf_string,
           const char* neginf_string) -> void {

    PythonFileLikeObject obj(source);

    py::gil_scoped_release release;

    ak::fromjsonobject(&obj,
                       builder,
                       buffersize,
                       read_one,
                       nan_string,
                       posinf_string,
                       neginf_string);

    py::gil_scoped_acquire acquire;

  }, py::arg("source"),
     py::arg("builder"),
     py::arg("read_one"),
     py::arg("buffersize"),
     py::arg("nan_string"),
     py::arg("posinf_string"),
     py::arg("neginf_string"));
}

void
make_fromjsonobj_schema(py::module& m, const std::string& name) {
  m.def(name.c_str(),
        [](py::object& source,
           py::dict& container,
           bool read_one,
           int64_t buffersize,
           const char* nan_string,
           const char* posinf_string,
           const char* neginf_string,
           const char* instructions,
           int64_t initial,
           double resize) -> int64_t {

    PythonFileLikeObject obj(source);

    py::gil_scoped_release release;

    ak::FromJsonObjectSchema out(&obj,
                                 buffersize,
                                 read_one,
                                 nan_string,
                                 posinf_string,
                                 neginf_string,
                                 instructions,
                                 initial,
                                 resize);

    py::gil_scoped_acquire acquire;

    for (int64_t i = 0;  i < out.num_outputs();  i++) {
      py::str key = out.output_name(i);
      std::string dtype = out.output_dtype(i);
      int64_t num_items = out.output_num_items(i);

      py::object value = py::module::import("numpy").attr("empty")(num_items, dtype);
      size_t data_pointer = value.attr("ctypes").attr("data").cast<size_t>();

      out.output_fill(i, reinterpret_cast<void*>(data_pointer));

      container[key] = value;
    }

    return out.length();

  }, py::arg("source"),
     py::arg("container"),
     py::arg("read_one"),
     py::arg("buffersize"),
     py::arg("nan_string"),
     py::arg("posinf_string"),
     py::arg("neginf_string"),
     py::arg("instructions"),
     py::arg("initial"),
     py::arg("resize"));
}

////////// Uproot connector

void
make_uproot_issue_90(py::module& m) {
  m.def("uproot_issue_90", &ak::uproot_issue_90);
}
