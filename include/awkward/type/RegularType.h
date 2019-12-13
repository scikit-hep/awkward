// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REGULARTYPE_H_
#define AWKWARD_REGULARTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  class RegularType: public Type {
  public:
    RegularType(const Parameters& parameters, const std::shared_ptr<Type> type, int64_t size)
        : Type(parameters)
        , type_(type)
        , size_(size) { }

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

    const std::shared_ptr<Type> type() const;
    int64_t size() const;

  private:
    const std::shared_ptr<Type> type_;
    const int64_t size_;
  };
}

#endif // AWKWARD_REGULARTYPE_H_
