// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ARRAYTYPE_H_
#define AWKWARD_ARRAYTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class ArrayType: public Type {
  public:
    ArrayType(const Parameters& parameters, const std::shared_ptr<Type> type, int64_t length)
        : Type(parameters)
        , type_(type)
        , length_(length) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual const std::shared_ptr<Type> shallow_copy() const;
    virtual bool equal(const std::shared_ptr<Type> other, bool check_parameters) const;
    virtual int64_t numfields() const;
    virtual int64_t fieldindex(const std::string& key) const;
    virtual const std::string key(int64_t fieldindex) const;
    virtual bool haskey(const std::string& key) const;
    virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const;
    virtual const std::vector<std::string> keyaliases(const std::string& key) const;
    virtual const std::vector<std::string> keys() const;

    const std::shared_ptr<Type> type() const;
    int64_t length() const;

  private:
    std::shared_ptr<Type> type_;
    int64_t length_;
  };
}

#endif // AWKWARD_ARRAYTYPE_H_
