// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_TYPE_H_
#define AWKWARD_TYPE_H_

#include <memory>
#include <vector>
#include <map>

#include "awkward/cpu-kernels/util.h"

namespace awkward {
  class Type {
  public:
    typedef std::map<std::string, std::string> Parameters;

    static std::shared_ptr<Type> none();

    Type(const Parameters& parameters);
    virtual ~Type();

    virtual std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const = 0;
    virtual const std::shared_ptr<Type> shallow_copy() const = 0;
    virtual bool equal(const std::shared_ptr<Type>& other, bool check_parameters) const = 0;
    virtual int64_t numfields() const = 0;
    virtual int64_t fieldindex(const std::string& key) const = 0;
    virtual const std::string key(int64_t fieldindex) const = 0;
    virtual bool haskey(const std::string& key) const = 0;
    virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const = 0;
    virtual const std::vector<std::string> keyaliases(const std::string& key) const = 0;
    virtual const std::vector<std::string> keys() const = 0;

    const Parameters parameters() const;
    void setparameters(const Parameters& parameters);
    std::string parameter(const std::string& key);
    void setparameter(const std::string& key, const std::string& value);
    bool parameter_equals(const std::string& key, const std::string& value);
    std::string tostring() const;
    const std::string compare(std::shared_ptr<Type> supertype);

  protected:
    bool equal_parameters(const Parameters& other) const;
    bool get_typestr(std::string& output) const;
    const std::string string_parameters() const;

    Parameters parameters_;
  };
}

#endif // AWKWARD_TYPE_H_
