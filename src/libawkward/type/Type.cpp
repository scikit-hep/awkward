// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "rapidjson/document.h"

#include "awkward/util.h"

#include "awkward/type/Type.h"

namespace rj = rapidjson;

namespace awkward {
  std::shared_ptr<Type> Type::nolength() const {
    return shallow_copy();
  }

  const std::string Type::compare(std::shared_ptr<Type> supertype) {
    // FIXME: better side-by-side comparison
    return tostring() + std::string(" versus ") + supertype.get()->tostring();
  }

  bool Type::equal_parameters(const Type::Parameters& other) const {
    if (parameters_FIXME_.size() != other.size()) {
      return false;
    }
    for (auto pair : parameters_FIXME_) {
      auto other_value = other.find(pair.first);
      if (other_value == other.end()) {
        return false;
      }
      rj::Document mine;
      rj::Document yours;
      mine.Parse<rj::kParseNanAndInfFlag>(pair.second.c_str());
      yours.Parse<rj::kParseNanAndInfFlag>(other_value->second.c_str());
      if (mine != yours) {
        return false;
      }
    }
    return true;
  }

  const std::string Type::string_parameters() const {
    std::stringstream out;
    out << "parameters={";
    bool first = true;
    for (auto pair : parameters_FIXME_) {
      if (!first) {
        out << ", ";
      }
      out << util::quote(pair.first, true) << ": " << pair.second;
      first = false;
    }
    out << "}";
    return out.str();
  }
}
