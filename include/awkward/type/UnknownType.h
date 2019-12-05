// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNKNOWNTYPE_H_
#define AWKWARD_UNKNOWNTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class UnknownType: public Type {
  public:
    UnknownType() { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual const std::shared_ptr<Type> shallow_copy() const;
    virtual bool equal(std::shared_ptr<Type> other) const;
    virtual std::shared_ptr<Type> level() const;
    virtual std::shared_ptr<Type> inner() const;
    virtual std::shared_ptr<Type> inner(std::string key) const;
    virtual int64_t numfields() const;
    virtual int64_t fieldindex(const std::string& key) const;
    virtual const std::string key(int64_t fieldindex) const;
    virtual bool haskey(const std::string& key) const;
    virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const;
    virtual const std::vector<std::string> keyaliases(const std::string& key) const;
    virtual const std::vector<std::string> keys() const;

  private:
  };
}

#endif // AWKWARD_LISTTYPE_H_
