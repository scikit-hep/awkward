// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/io.cpp", line)

#include <string>

#include "awkward/Content.h"
#include "awkward/Index.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/io/json.h"
#include "awkward/io/root.h"

#include "awkward/python/io.h"

namespace ak = awkward;

////////// fromjson

void
make_fromjson(py::module& m, const std::string& name) {
  m.def(name.c_str(),
        [](const std::string& source,
           int64_t initial,
           double resize,
           int64_t buffersize) -> std::shared_ptr<ak::Content> {
    bool isarray = false;
    bool isrecord = false;
    for (char const &x: source) {
      if (x != 9  &&  x != 10  &&  x != 13  &&  x != 32) {  // whitespace
        if (x == 91) {         // opening square bracket
          isarray = true;
        }
        else if (x == 123) {   // opening curly bracket
          isrecord = true;
        }
        break;
      }
    }
    if (isarray) {
      return ak::FromJsonString(
        source.c_str(), ak::ArrayBuilderOptions(initial, resize));
    }
    if (isrecord) {
      return ak::FromJsonString(
        source.c_str(), ak::ArrayBuilderOptions(initial, resize)
      ).get()->getitem_at_nowrap(0).get()->getitem_at_nowrap(0);
    }
    else {
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
      std::shared_ptr<ak::Content> out(nullptr);
      try {
        out = FromJsonFile(file,
                           ak::ArrayBuilderOptions(initial, resize),
                           buffersize);
      }
      catch (...) {
        fclose(file);
        throw;
      }
      fclose(file);
      return out;
    }
  }, py::arg("source"),
      py::arg("initial") = 1024,
      py::arg("resize") = 1.5,
      py::arg("buffersize") = 65536);
}

////////// fromroot

void
make_fromroot_nestedvector(py::module& m, const std::string& name) {
  m.def(name.c_str(),
        [](const ak::Index64& byteoffsets,
           const ak::NumpyArray& rawdata,
           int64_t depth,
           int64_t itemsize,
           const std::string& format,
           int64_t initial,
           double resize) -> std::shared_ptr<ak::Content> {
      return FromROOT_nestedvector(byteoffsets,
                                   rawdata,
                                   depth,
                                   itemsize,
                                   format,
                                   ak::ArrayBuilderOptions(initial, resize));
  }, py::arg("byteoffsets"),
     py::arg("rawdata"),
     py::arg("depth"),
     py::arg("itemsize"),
     py::arg("format"),
     py::arg("initial") = 1024,
     py::arg("resize") = 1.5);
}
