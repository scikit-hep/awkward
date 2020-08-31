// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/util.cpp", line)

#include <sstream>
#include <set>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "awkward/kernels/identities.h"
#include "awkward/kernels/getitem.h"
#include "awkward/kernels/operations.h"
#include "awkward/kernels/reducers.h"
#include "awkward/kernels/sorting.h"

#include "awkward/util.h"
#include "awkward/Identities.h"

namespace rj = rapidjson;

namespace awkward {
  namespace util {

    dtype
    name_to_dtype(const std::string& name) {
      if (name == "bool") {
        return util::dtype::boolean;
      }
      else if (name == "int8") {
        return util::dtype::int8;
      }
      else if (name == "int16") {
        return util::dtype::int16;
      }
      else if (name == "int32") {
        return util::dtype::int32;
      }
      else if (name == "int64") {
        return util::dtype::int64;
      }
      else if (name == "uint8") {
        return util::dtype::uint8;
      }
      else if (name == "uint16") {
        return util::dtype::uint16;
      }
      else if (name == "uint32") {
        return util::dtype::uint32;
      }
      else if (name == "uint64") {
        return util::dtype::uint64;
      }
      else if (name == "float16") {
        return util::dtype::float16;
      }
      else if (name == "float32") {
        return util::dtype::float32;
      }
      else if (name == "float64") {
        return util::dtype::float64;
      }
      else if (name == "float128") {
        return util::dtype::float128;
      }
      else if (name == "complex64") {
        return util::dtype::complex64;
      }
      else if (name == "complex128") {
        return util::dtype::complex128;
      }
      else if (name == "complex256") {
        return util::dtype::complex256;
      }
      // else if (name == "datetime64") {
      //   return util::dtype::datetime64;
      // }
      // else if (name == "timedelta64") {
      //   return util::dtype::timedelta64;
      // }
      else {
        return util::dtype::NOT_PRIMITIVE;
      }
    }

    const std::string
    dtype_to_name(dtype dt) {
      switch (dt) {
      case util::dtype::boolean:
        return "bool";
      case util::dtype::int8:
        return "int8";
      case util::dtype::int16:
        return "int16";
      case util::dtype::int32:
        return "int32";
      case util::dtype::int64:
        return "int64";
      case util::dtype::uint8:
        return "uint8";
      case util::dtype::uint16:
        return "uint16";
      case util::dtype::uint32:
        return "uint32";
      case util::dtype::uint64:
        return "uint64";
      case util::dtype::float16:
        return "float16";
      case util::dtype::float32:
        return "float32";
      case util::dtype::float64:
        return "float64";
      case util::dtype::float128:
        return "float128";
      case util::dtype::complex64:
        return "complex64";
      case util::dtype::complex128:
        return "complex128";
      case util::dtype::complex256:
        return "complex256";
        // case datetime64:
        //   return "datetime64";
        // case timedelta64:
        //   return "timedelta64";
      default:
        return "unknown";
      }
    }

    dtype
    format_to_dtype(const std::string& format, int64_t itemsize) {
      int32_t test = 1;
      bool little_endian = (*(int8_t*)&test == 1);

      std::string fmt = format;
      if (format.length() > 1) {
        std::string endianness = format.substr(0, 1);
        if ((endianness == ">"  &&  !little_endian)  ||
            (endianness == "<"  &&  little_endian)  ||
            (endianness == "=")) {
          fmt = format.substr(1, format.length() - 1);
        }
        else if ((endianness == ">"  &&  little_endian)  ||
                 (endianness == "<"  &&  !little_endian)) {
          return dtype::NOT_PRIMITIVE;
        }
      }

      if (fmt == std::string("?")) {
        return dtype::boolean;
      }
      else if (fmt == std::string("b")  ||
               fmt == std::string("h")  ||
               fmt == std::string("i")  ||
               fmt == std::string("l")  ||
               fmt == std::string("q")) {
        if (itemsize == 1) {
          return dtype::int8;
        }
        else if (itemsize == 2) {
          return dtype::int16;
        }
        else if (itemsize == 4) {
          return dtype::int32;
        }
        else if (itemsize == 8) {
          return dtype::int64;
        }
        else {
          return dtype::NOT_PRIMITIVE;
        }
      }
      else if (fmt == std::string("c")  ||
               fmt == std::string("B")  ||
               fmt == std::string("H")  ||
               fmt == std::string("I")  ||
               fmt == std::string("L")  ||
               fmt == std::string("Q")) {
        if (itemsize == 1) {
          return dtype::uint8;
        }
        else if (itemsize == 2) {
          return dtype::uint16;
        }
        else if (itemsize == 4) {
          return dtype::uint32;
        }
        else if (itemsize == 8) {
          return dtype::uint64;
        }
        else {
          return dtype::NOT_PRIMITIVE;
        }
      }
      else if (fmt == std::string("e")) {
        return dtype::float16;
      }
      else if (fmt == std::string("f")) {
        return dtype::float32;
      }
      else if (fmt == std::string("d")) {
        return dtype::float64;
      }
      else if (fmt == std::string("g")) {
        return dtype::float128;
      }
      else if (fmt == std::string("Zf")) {
        return dtype::complex64;
      }
      else if (fmt == std::string("Zd")) {
        return dtype::complex128;
      }
      else if (fmt == std::string("Zg")) {
        return dtype::complex256;
      }
      // else if (fmt == std::string("M")) {
      //   return dtype::datetime64;
      // }
      // else if (fmt == std::string("m")) {
      //   return dtype::timedelta64;
      // }
      else {
        return dtype::NOT_PRIMITIVE;
      }
    }

    const std::string
    dtype_to_format(dtype dt) {
      switch (dt) {
      case dtype::boolean:
        return "?";
      case dtype::int8:
        return "b";
      case dtype::int16:
        return "h";
      case dtype::int32:
#if defined _MSC_VER || defined __i386__
        return "l";
#else
        return "i";
#endif
      case dtype::int64:
#if defined _MSC_VER || defined __i386__
        return "q";
#else
        return "l";
#endif
      case dtype::uint8:
        return "B";
      case dtype::uint16:
        return "H";
      case dtype::uint32:
#if defined _MSC_VER || defined __i386__
        return "L";
#else
        return "I";
#endif
      case dtype::uint64:
#if defined _MSC_VER || defined __i386__
        return "Q";
#else
        return "L";
#endif
      case dtype::float16:
        return "e";
      case dtype::float32:
        return "f";
      case dtype::float64:
        return "d";
      case dtype::float128:
        return "g";
      case dtype::complex64:
        return "Zf";
      case dtype::complex128:
        return "Zd";
      case dtype::complex256:
        return "Zg";
      // case dtype::datetime64:
      //   return "M";
      // case dtype::timedelta64:
      //   return "m";
      default:
        return "";
      }
    }

    int64_t
    dtype_to_itemsize(dtype dt) {
      switch (dt) {
      case dtype::boolean:
        return 1;
      case dtype::int8:
        return 1;
      case dtype::int16:
        return 2;
      case dtype::int32:
        return 4;
      case dtype::int64:
        return 8;
      case dtype::uint8:
        return 1;
      case dtype::uint16:
        return 2;
      case dtype::uint32:
        return 4;
      case dtype::uint64:
        return 8;
      case dtype::float16:
        return 2;
      case dtype::float32:
        return 4;
      case dtype::float64:
        return 8;
      case dtype::float128:
        return 16;
      case dtype::complex64:
        return 8;
      case dtype::complex128:
        return 16;
      case dtype::complex256:
        return 32;
      // case dtype::datetime64:
      //   return 8;
      // case dtype::timedelta64:
      //   return 8;
      default:
        return 0;
      }
    }

    bool
    is_integer(dtype dt) {
      switch (dt) {
      case dtype::int8:
      case dtype::int16:
      case dtype::int32:
      case dtype::int64:
      case dtype::uint8:
      case dtype::uint16:
      case dtype::uint32:
      case dtype::uint64:
        return true;
      default:
        return false;
      }
    }

    bool
    is_signed(dtype dt) {
      switch (dt) {
      case dtype::int8:
      case dtype::int16:
      case dtype::int32:
      case dtype::int64:
        return true;
      default:
        return false;
      }
    }

    bool
    is_unsigned(dtype dt) {
      switch (dt) {
      case dtype::uint8:
      case dtype::uint16:
      case dtype::uint32:
      case dtype::uint64:
        return true;
      default:
        return false;
      }
    }

    bool
    is_real(dtype dt) {
      switch (dt) {
      case dtype::float16:
      case dtype::float32:
      case dtype::float64:
      case dtype::float128:
        return true;
      default:
        return false;
      }
    }

    bool
    is_complex(dtype dt) {
      switch (dt) {
      case dtype::complex64:
      case dtype::complex128:
      case dtype::complex256:
        return true;
      default:
        return false;
      }
    }

    void
    handle_error(const struct Error& err,
                 const std::string& classname,
                 const Identities* identities) {
      std::string filename = (err.filename == nullptr ? "" : err.filename);

      if (err.pass_through == true) {
        throw std::invalid_argument(std::string(err.str) + filename);
      }
      else {
        if (err.str != nullptr) {
          std::stringstream out;
          out << "in " << classname;
          if (err.identity != kSliceNone && identities != nullptr) {
            if (0 <= err.identity && err.identity < identities->length()) {
              out << " with identity ["
                  << identities->identity_at(err.identity) << "]";
            } else {
              out << " with invalid identity";
            }
          }
          if (err.attempt != kSliceNone) {
            out << " attempting to get " << err.attempt;
          }
          out << ", " << err.str << filename;
          throw std::invalid_argument(out.str());
        }
      }
    }

    template<typename T>
    IndexOf<T> make_starts(const IndexOf<T> &offsets) {
      return IndexOf<T>(offsets.ptr(),
                        offsets.offset(),
                        offsets.length() - 1,
                        offsets.ptr_lib());
    }

    template<typename T>
    IndexOf<T> make_stops(const IndexOf<T> &offsets) {
      return IndexOf<T>(offsets.ptr(),
                        offsets.offset() + 1,
                        offsets.length() - 1,
                        offsets.ptr_lib());
    }

    template IndexOf<int32_t> make_starts(const IndexOf<int32_t> &offsets);

    template IndexOf<uint32_t> make_starts(const IndexOf<uint32_t> &offsets);

    template IndexOf<int64_t> make_starts(const IndexOf<int64_t> &offsets);

    template IndexOf<int32_t> make_stops(const IndexOf<int32_t> &offsets);

    template IndexOf<uint32_t> make_stops(const IndexOf<uint32_t> &offsets);

    template IndexOf<int64_t> make_stops(const IndexOf<int64_t> &offsets);

    std::string
    quote(const std::string &x) {
      rj::StringBuffer buffer;
      rj::Writer<rj::StringBuffer> writer(buffer);
      writer.String(x.c_str(), x.length());
      return std::string(buffer.GetString());
    }

    RecordLookupPtr
    init_recordlookup(int64_t numfields) {
      RecordLookupPtr out = std::make_shared<RecordLookup>();
      for (int64_t i = 0; i < numfields; i++) {
        out.get()->push_back(std::to_string(i));
      }
      return out;
    }

    int64_t
    fieldindex(const RecordLookupPtr &recordlookup,
               const std::string &key,
               int64_t numfields) {
      int64_t out = -1;
      if (recordlookup.get() != nullptr) {
        for (size_t i = 0; i < recordlookup.get()->size(); i++) {
          if (recordlookup.get()->at(i) == key) {
            out = (int64_t) i;
            break;
          }
        }
      }
      if (out == -1) {
        try {
          out = (int64_t) std::stoi(key);
        }
        catch (std::invalid_argument err) {
          throw std::invalid_argument(
            std::string("key ") + quote(key)
            + std::string(" does not exist (not in record)") + FILENAME(__LINE__));
        }
        if (!(0 <= out && out < numfields)) {
          throw std::invalid_argument(
            std::string("key interpreted as fieldindex ") + key
            + std::string(" for records with only ") + std::to_string(numfields)
            + std::string(" fields") + FILENAME(__LINE__));
        }
      }
      return out;
    }

    const std::string
    key(const RecordLookupPtr &recordlookup,
        int64_t fieldindex,
        int64_t numfields) {
      if (fieldindex >= numfields) {
        throw std::invalid_argument(
          std::string("fieldindex ") + std::to_string(fieldindex)
          + std::string(" for records with only ") + std::to_string(numfields)
          + std::string(" fields") + FILENAME(__LINE__));
      }
      if (recordlookup.get() != nullptr) {
        return recordlookup.get()->at((size_t) fieldindex);
      } else {
        return std::to_string(fieldindex);
      }
    }

    bool
    haskey(const RecordLookupPtr &recordlookup,
           const std::string &key,
           int64_t numfields) {
      try {
        fieldindex(recordlookup, key, numfields);
      }
      catch (std::invalid_argument err) {
        return false;
      }
      return true;
    }

    const std::vector<std::string>
    keys(const RecordLookupPtr &recordlookup, int64_t numfields) {
      std::vector<std::string> out;
      if (recordlookup.get() != nullptr) {
        out.insert(out.end(),
                   recordlookup.get()->begin(),
                   recordlookup.get()->end());
      } else {
        int64_t cols = numfields;
        for (int64_t j = 0; j < cols; j++) {
          out.push_back(std::to_string(j));
        }
      }
      return out;
    }

    bool
    parameter_equals(const Parameters &parameters,
                     const std::string &key,
                     const std::string &value) {
      auto item = parameters.find(key);
      std::string myvalue;
      if (item == parameters.end()) {
        myvalue = "null";
      } else {
        myvalue = item->second;
      }
      rj::Document mine;
      rj::Document yours;
      mine.Parse<rj::kParseNanAndInfFlag>(myvalue.c_str());
      yours.Parse<rj::kParseNanAndInfFlag>(value.c_str());
      return mine == yours;
    }

    bool
    parameters_equal(const Parameters &self, const Parameters &other) {
      std::set<std::string> checked;
      for (auto pair : self) {
        if (!parameter_equals(other, pair.first, pair.second)) {
          return false;
        }
        checked.insert(pair.first);
      }
      for (auto pair : other) {
        if (checked.find(pair.first) == checked.end()) {
          if (!parameter_equals(self, pair.first, pair.second)) {
            return false;
          }
        }
      }
      return true;
    }

    bool
    parameter_isstring(const Parameters &parameters, const std::string &key) {
      auto item = parameters.find(key);
      if (item == parameters.end()) {
        return false;
      }
      rj::Document mine;
      mine.Parse<rj::kParseNanAndInfFlag>(item->second.c_str());
      return mine.IsString();
    }

    bool
    parameter_isname(const Parameters &parameters, const std::string &key) {
      auto item = parameters.find(key);
      if (item == parameters.end()) {
        return false;
      }
      rj::Document mine;
      mine.Parse<rj::kParseNanAndInfFlag>(item->second.c_str());
      if (!mine.IsString()) {
        return false;
      }
      std::string value = mine.GetString();
      if (value.empty()) {
        return false;
      }
      if (!((value[0] >= 'a' && value[0] <= 'z') ||
            (value[0] >= 'A' && value[0] <= 'Z') ||
            (value[0] == '_'))) {
        return false;
      }
      for (size_t i = 1; i < value.length(); i++) {
        if (!((value[i] >= 'a' && value[i] <= 'z') ||
              (value[i] >= 'A' && value[i] <= 'Z') ||
              (value[i] >= '0' && value[i] <= '9') ||
              (value[i] == '_'))) {
          return false;
        }
      }
      return true;
    }

    const std::string
    parameter_asstring(const Parameters &parameters, const std::string &key) {
      auto item = parameters.find(key);
      if (item == parameters.end()) {
        throw std::runtime_error(
          std::string("parameter is null") + FILENAME(__LINE__));
      }
      rj::Document mine;
      mine.Parse<rj::kParseNanAndInfFlag>(item->second.c_str());
      if (!mine.IsString()) {
        throw std::runtime_error(
          std::string("parameter is not a string") + FILENAME(__LINE__));
      }
      return mine.GetString();
    }

    std::string
    gettypestr(const Parameters &parameters, const TypeStrs &typestrs) {
      auto item = parameters.find("__record__");
      if (item != parameters.end()) {
        std::string source = item->second;
        rj::Document recname;
        recname.Parse<rj::kParseNanAndInfFlag>(source.c_str());
        if (recname.IsString()) {
          std::string name = recname.GetString();
          for (auto pair : typestrs) {
            if (pair.first == name) {
              return pair.second;
            }
          }
        }
      }
      item = parameters.find("__array__");
      if (item != parameters.end()) {
        std::string source = item->second;
        rj::Document recname;
        recname.Parse<rj::kParseNanAndInfFlag>(source.c_str());
        if (recname.IsString()) {
          std::string name = recname.GetString();
          for (auto pair : typestrs) {
            if (pair.first == name) {
              return pair.second;
            }
          }
        }
      }
      return std::string();
    }

  }
}
