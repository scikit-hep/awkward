// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/util.cpp", line)

#include <sstream>
#include <set>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "awkward/util.h"

namespace rj = rapidjson;

namespace awkward {
  namespace util {

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
      case util::dtype::datetime64:
        return "datetime64";
      case util::dtype::timedelta64:
        return "timedelta64";
      default:
        return "unknown";
      }
    }

    const std::string
    dtype_to_format(dtype dt, const std::string& format) {
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
      case dtype::datetime64:
        return format.empty() ? "M" : format;
      case dtype::timedelta64:
        return format.empty() ? "m" : format;
      default:
        return "";
      }
    }

    std::string
    quote(const std::string &x) {
      rj::StringBuffer buffer;
      rj::Writer<rj::StringBuffer> writer(buffer);
      writer.String(x.c_str(), (rj::SizeType)x.length());
      return std::string(buffer.GetString());
    }
  }
}
