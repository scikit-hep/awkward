// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONTYPE_H_
#define AWKWARD_UNIONTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  class UnionType: public Type {
  public:
    UnionType(const Parameters& parameters, const std::vector<std::shared_ptr<Type>>& types): Type(parameters), types_(types) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual const std::shared_ptr<Type> shallow_copy() const;
    virtual bool equal(const std::shared_ptr<Type> other, bool check_parameters) const;
    virtual std::shared_ptr<Type> level() const;
    virtual std::shared_ptr<Type> inner() const;
    virtual std::shared_ptr<Type> inner(const std::string& key) const;
    virtual int64_t numfields() const;
    virtual int64_t fieldindex(const std::string& key) const;
    virtual const std::string key(int64_t fieldindex) const;
    virtual bool haskey(const std::string& key) const;
    virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const;
    virtual const std::vector<std::string> keyaliases(const std::string& key) const;
    virtual const std::vector<std::string> keys() const;

    int64_t numtypes() const;
    const std::vector<std::shared_ptr<Type>> types() const;
    const std::shared_ptr<Type> type(int64_t index) const;

  private:
    const std::vector<std::shared_ptr<Type>> types_;
  };
}

#endif // AWKWARD_OPTIONTYPE_H_
