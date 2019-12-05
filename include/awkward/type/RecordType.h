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

    RecordType(const std::vector<std::shared_ptr<Type>>& types, const std::shared_ptr<Lookup>& lookup, const std::shared_ptr<ReverseLookup>& reverselookup)
        : types_(types)
        , lookup_(lookup)
        , reverselookup_(reverselookup) { }
    RecordType(const std::vector<std::shared_ptr<Type>>& types)
        : types_(types)
        , lookup_(nullptr)
        , reverselookup_(nullptr) { }

    const std::vector<std::shared_ptr<Type>> types() const { return types_; };
    const std::shared_ptr<Lookup> lookup() const { return lookup_; }
    const std::shared_ptr<ReverseLookup> reverselookup() const { return reverselookup_; }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual const std::shared_ptr<Type> shallow_copy() const;
    virtual bool equal(std::shared_ptr<Type> other) const;
    virtual std::shared_ptr<Type> inner() const;
    virtual std::shared_ptr<Type> inner(std::string key) const;
    virtual int64_t numfields() const;
    virtual int64_t fieldindex(const std::string& key) const;
    virtual const std::string key(int64_t fieldindex) const;
    virtual bool haskey(const std::string& key) const;
    virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const;
    virtual const std::vector<std::string> keyaliases(const std::string& key) const;
    virtual const std::vector<std::string> keys() const;

    const std::shared_ptr<Type> field(int64_t fieldindex) const;
    const std::shared_ptr<Type> field(const std::string& key) const;
    const std::vector<std::shared_ptr<Type>> fields() const;
    const std::vector<std::pair<std::string, std::shared_ptr<Type>>> fielditems() const;

  private:
    const std::vector<std::shared_ptr<Type>> types_;
    const std::shared_ptr<Lookup> lookup_;
    const std::shared_ptr<ReverseLookup> reverselookup_;
  };
}

#endif // AWKWARD_RECORDTYPE_H_
