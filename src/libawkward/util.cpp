// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <sstream>

#include "awkward/util.h"
#include "awkward/Identity.h"

namespace awkward {
  namespace util {
    void handle_error(const Error& err, const std::string classname, const Identity* id) {
      if (err.str != nullptr) {
        std::stringstream out;
        out << "in " << classname;
        if (err.location != kSliceNone  &&  id != nullptr) {
          assert(err.location > 0);
          if (0 <= err.location  &&  err.location < id->length()) {
            out << " at id[" << id->location(err.location) << "]";
          }
          else {
            out << " at id[???]";
          }
        }
        if (err.attempt != kSliceNone) {
          out << " attempting to get " << err.attempt;
        }
        out << ", " << err.str;
        throw std::invalid_argument(out.str());
      }
    }
  }
}
