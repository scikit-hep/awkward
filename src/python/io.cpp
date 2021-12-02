// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/io.cpp", line)

#include <pybind11/numpy.h>
#include <string>

#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/io/json.h"
#include "awkward/io/uproot.h"
#include "awkward/python/content.h"

#include "awkward/python/io.h"

namespace ak = awkward;

////////// fromjson

void
make_fromjson(py::module& m, const std::string& name) {
  m.def(name.c_str(),
        [](const std::string& source,
           const char* nan_string,
           const char* infinity_string,
           const char* minus_infinity_string,
           int64_t initial,
           double resize,
           int64_t buffersize) -> py::object {
    auto out = ak::FromJsonString(source.c_str(),
                                  ak::ArrayBuilderOptions(initial, resize),
                                  nan_string,
                                  infinity_string,
                                  minus_infinity_string);
    if (out.first == 1) {
      return box(unbox_content(::builder_snapshot(out.second))->getitem_at_nowrap(0));
    }
    else {
      return ::builder_snapshot(out.second);
    }
  }, py::arg("source"),
     py::arg("nan_string") = nullptr,
     py::arg("infinity_string") = nullptr,
     py::arg("minus_infinity_string") = nullptr,
     py::arg("initial") = 1024,
     py::arg("resize") = 1.5,
     py::arg("buffersize") = 65536);
}

void
make_fromjsonfile(py::module& m, const std::string& name) {
  m.def(name.c_str(),
        [](const std::string& source,
           const char* nan_string,
           const char* infinity_string,
           const char* minus_infinity_string,
           int64_t initial,
           double resize,
           int64_t buffersize) -> py::object {
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
      ak::BuilderPtr out(nullptr);
      try {
        auto out_pair = ak::FromJsonFile(file,
                           ak::ArrayBuilderOptions(initial, resize),
                           buffersize,
                           nan_string,
                           infinity_string,
                           minus_infinity_string);
        num = out_pair.first;
        out = out_pair.second;
      }
      catch (...) {
        fclose(file);
        throw;
      }
      fclose(file);
      if (num == 1) {
        return box(unbox_content(::builder_snapshot(out))->getitem_at_nowrap(0));
      }
      else {
        return ::builder_snapshot(out);
      }
  }, py::arg("source"),
     py::arg("nan_string") = nullptr,
     py::arg("infinity_string") = nullptr,
     py::arg("minus_infinity_string") = nullptr,
     py::arg("initial") = 1024,
     py::arg("resize") = 1.5,
     py::arg("buffersize") = 65536);
}

////////// Uproot connector

void
make_uproot_issue_90(py::module& m) {
  m.def("uproot_issue_90", &ak::uproot_issue_90);
}
