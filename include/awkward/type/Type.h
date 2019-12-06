// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_TYPE_H_
#define AWKWARD_TYPE_H_

#include <memory>
#include <vector>

#include "awkward/cpu-kernels/util.h"

namespace awkward {
  class Type {
  public:
    virtual ~Type() { }

    static std::shared_ptr<Type> none() { return std::shared_ptr<Type>(nullptr); }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const = 0;
    virtual const std::shared_ptr<Type> shallow_copy() const = 0;
    virtual bool shallow_equal(const std::shared_ptr<Type> other) const = 0;
    virtual bool equal(const std::shared_ptr<Type> other) const = 0;
    virtual std::shared_ptr<Type> nolength() const;
    virtual std::shared_ptr<Type> level() const = 0;
    virtual std::shared_ptr<Type> inner() const = 0;
    virtual std::shared_ptr<Type> inner(const std::string& key) const = 0;
    virtual int64_t numfields() const = 0;
    virtual int64_t fieldindex(const std::string& key) const = 0;
    virtual const std::string key(int64_t fieldindex) const = 0;
    virtual bool haskey(const std::string& key) const = 0;
    virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const = 0;
    virtual const std::vector<std::string> keyaliases(const std::string& key) const = 0;
    virtual const std::vector<std::string> keys() const = 0;

    std::string tostring() const { return tostring_part("", "", ""); };
    const std::string compare(std::shared_ptr<Type> supertype);
  };
}

#endif // AWKWARD_TYPE_H_
