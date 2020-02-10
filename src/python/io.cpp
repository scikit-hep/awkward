// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/python/boxing.h"

#include "awkward/python/io.h"

void make_fromjson(py::module m, std::string name) {
  m.def(name.c_str(), [](std::string source, int64_t initial, double resize, int64_t buffersize) -> py::object {
    bool isarray = false;
    for (char const &x: source) {
      if (x != 9  &&  x != 10  &&  x != 13  &&  x != 32) {  // whitespace
        if (x == 91) {       // opening square bracket
          isarray = true;
        }
        break;
      }
    }
    if (isarray) {
      return box(ak::FromJsonString(source.c_str(), ak::FillableOptions(initial, resize)));
    }
    else {
#ifdef _MSC_VER
      FILE* file;
      if (fopen_s(&file, source.c_str(), "rb") != 0) {
#else
      FILE* file = fopen(source.c_str(), "rb");
      if (file == nullptr) {
#endif
        throw std::invalid_argument(std::string("file \"") + source + std::string("\" could not be opened for reading"));
      }
      std::shared_ptr<ak::Content> out(nullptr);
      try {
        out = FromJsonFile(file, ak::FillableOptions(initial, resize), buffersize);
      }
      catch (...) {
        fclose(file);
        throw;
      }
      fclose(file);
      return box(out);
    }
  }, py::arg("source"), py::arg("initial") = 1024, py::arg("resize") = 2.0, py::arg("buffersize") = 65536);
}

void make_fromroot_nestedvector(py::module m, std::string name) {
  m.def(name.c_str(), [](ak::Index64& byteoffsets, ak::NumpyArray& rawdata, int64_t depth, int64_t itemsize, std::string format, int64_t initial, double resize) -> py::object {
      return box(FromROOT_nestedvector(byteoffsets, rawdata, depth, itemsize, format, ak::FillableOptions(initial, resize)));
  }, py::arg("byteoffsets"), py::arg("rawdata"), py::arg("depth"), py::arg("itemsize"), py::arg("format"), py::arg("initial") = 1024, py::arg("resize") = 2.0);
}
