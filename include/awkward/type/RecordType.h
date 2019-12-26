// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDTYPE_H_
#define AWKWARD_RECORDTYPE_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "awkward/util.h"

#include "awkward/type/Type.h"

namespace awkward {
  class RecordType: public Type {
  public:
    RecordType(const Parameters& parameters, const std::vector<std::shared_ptr<Type>>& types, const std::shared_ptr<util::RecordLookup>& recordlookup);
    RecordType(const Parameters& parameters, const std::vector<std::shared_ptr<Type>>& types);

    const std::vector<std::shared_ptr<Type>> types() const;
    const std::shared_ptr<util::RecordLookup> recordlookup() const;
    bool istuple() const;

    std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    const std::shared_ptr<Type> shallow_copy() const override;
    bool equal(const std::shared_ptr<Type>& other, bool check_parameters) const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

    const std::shared_ptr<Type> field(int64_t fieldindex) const;
    const std::shared_ptr<Type> field(const std::string& key) const;
    const std::vector<std::shared_ptr<Type>> fields() const;
    const std::vector<std::pair<std::string, std::shared_ptr<Type>>> fielditems() const;
    const std::shared_ptr<Type> astuple() const;

    void append(const std::shared_ptr<Type>& type, const std::string& key);
    void append(const std::shared_ptr<Type>& type);

  private:
    std::vector<std::shared_ptr<Type>> types_;
    std::shared_ptr<util::RecordLookup> recordlookup_;
  };
}

#endif // AWKWARD_RECORDTYPE_H_
