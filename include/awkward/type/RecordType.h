// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDTYPE_H_
#define AWKWARD_RECORDTYPE_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "awkward/type/Type.h"

namespace awkward {
  class RecordType: public Type {
  public:
    typedef std::unordered_map<std::string, size_t> Lookup;
    typedef std::vector<std::string> ReverseLookup;

    RecordType(const Parameters& parameters, const std::vector<std::shared_ptr<Type>>& types, const std::shared_ptr<Lookup>& lookup, const std::shared_ptr<ReverseLookup>& reverselookup)
        : Type(parameters)
        , types_(types)
        , lookup_(lookup)
        , reverselookup_(reverselookup) { }
    RecordType(const Parameters& parameters, const std::vector<std::shared_ptr<Type>>& types)
        : Type(parameters)
        , types_(types)
        , lookup_(nullptr)
        , reverselookup_(nullptr) { }

    const std::vector<std::shared_ptr<Type>> types() const { return types_; };
    const std::shared_ptr<Lookup> lookup() const { return lookup_; }
    const std::shared_ptr<ReverseLookup> reverselookup() const { return reverselookup_; }
    bool istuple() const { return lookup_.get() == nullptr; }

    std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    const std::shared_ptr<Type> shallow_copy() const override;
    bool equal(const std::shared_ptr<Type>& other, bool check_parameters) const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keyaliases(int64_t fieldindex) const override;
    const std::vector<std::string> keyaliases(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

    const std::shared_ptr<Type> field(int64_t fieldindex) const;
    const std::shared_ptr<Type> field(const std::string& key) const;
    const std::vector<std::shared_ptr<Type>> fields() const;
    const std::vector<std::pair<std::string, std::shared_ptr<Type>>> fielditems() const;
    const std::shared_ptr<Type> astuple() const;

    void append(const std::shared_ptr<Type>& type);
    void setkey(int64_t fieldindex, const std::string& key);

  private:
    std::vector<std::shared_ptr<Type>> types_;
    std::shared_ptr<Lookup> lookup_;
    std::shared_ptr<ReverseLookup> reverselookup_;
  };
}

#endif // AWKWARD_RECORDTYPE_H_
