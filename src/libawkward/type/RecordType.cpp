// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>
#include <sstream>

#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/RecordType.h"

namespace awkward {
    std::string RecordType::tostring_part(std::string indent, std::string pre, std::string post) const {
      throw std::runtime_error("FIXME: RecordType::tostring_part");
    }

    const std::shared_ptr<Type> RecordType::shallow_copy() const {
      throw std::runtime_error("FIXME: RecordType::shallow_copy");
    }

    bool RecordType::equal(std::shared_ptr<Type> other) const {
      throw std::runtime_error("FIXME: RecordType::equal");
    }

    bool RecordType::compatible(std::shared_ptr<Type> other, bool bool_is_int, bool int_is_float, bool ignore_null, bool unknown_is_anything) const {
      throw std::runtime_error("FIXME: RecordType::compatible");
    }

    int64_t RecordType::numfields() const {
      throw std::runtime_error("FIXME: RecordType::numfields");
    }

    int64_t RecordType::index(const std::string& key) const {
      throw std::runtime_error("FIXME: RecordType::index");
    }

    const std::string RecordType::key(int64_t index) const {
      throw std::runtime_error("FIXME: RecordType::key");
    }

    bool RecordType::has(const std::string& key) const {
      throw std::runtime_error("FIXME: RecordType::has");
    }

    const std::vector<std::string> RecordType::aliases(int64_t index) const {
      throw std::runtime_error("FIXME: RecordType::aliases");
    }

    const std::vector<std::string> RecordType::aliases(const std::string& key) const {
      throw std::runtime_error("FIXME: RecordType::aliases");
    }

    const std::shared_ptr<Type> RecordType::field(int64_t index) const {
      throw std::runtime_error("FIXME: RecordType::field");
    }

    const std::shared_ptr<Type> RecordType::field(const std::string& key) const {
      throw std::runtime_error("FIXME: RecordType::field");
    }

    const std::vector<std::string> RecordType::keys() const {
      throw std::runtime_error("FIXME: RecordType::keys");
    }

    const std::vector<std::shared_ptr<Type>> RecordType::values() const {
      throw std::runtime_error("FIXME: RecordType::values");
    }

    const std::vector<std::pair<std::string, std::shared_ptr<Type>>> RecordType::items() const {
      throw std::runtime_error("FIXME: RecordType::items");
    }

}
