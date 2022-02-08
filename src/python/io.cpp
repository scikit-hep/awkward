// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/io.cpp", line)

#include <pybind11/numpy.h>
#include <string>

#include "awkward/builder/ArrayBuilderOptions.h"
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

////////// Uproot connector

void
make_uproot_issue_90(py::module& m) {
  m.def("uproot_issue_90", &ak::uproot_issue_90);
}
