// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/util.h"
#include "awkward/Identity.h"

namespace awkward {
  namespace util {
    void handle_error(const Error& err, const std::string classname, const Identity* id1, const Identity* id2) {
      if (err.str != nullptr) {
        throw std::invalid_argument(
          (std::string("in ") + classname) +
          (err.where1 >= 0 ? std::string(" at index ") + std::to_string(err.where1) : std::string("")) +
          (err.where1 >= 0  &&  id1 != nullptr ? std::string(" (id ") + id1->position(err.where1) + std::string(")") : std::string("")) +
          (err.where1 >= 0  &&  err.where2 >= 0 ? std::string(" and index ") + std::to_string(err.where2) : std::string("")) +
          (err.where1 >= 0  &&  err.where2 >= 0  &&  id2 != nullptr ? std::string(" (id ") + id2->position(err.where2) + std::string(")") : std::string("")) +
          std::string(": ") + std::string(err.str)
        );
      }
    }
  }
}
