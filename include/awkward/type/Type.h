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

    Type(const Parameters& parameters): parameters_(parameters) { }
    virtual ~Type() { }

    static std::shared_ptr<Type> none() { return std::shared_ptr<Type>(nullptr); }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const = 0;
    virtual const std::shared_ptr<Type> shallow_copy() const = 0;
    virtual bool equal(const std::shared_ptr<Type> other, bool check_parameters) const = 0;
    virtual int64_t numfields() const = 0;
    virtual int64_t fieldindex(const std::string& key) const = 0;
    virtual const std::string key(int64_t fieldindex) const = 0;
    virtual bool haskey(const std::string& key) const = 0;
    virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const = 0;
    virtual const std::vector<std::string> keyaliases(const std::string& key) const = 0;
    virtual const std::vector<std::string> keys() const = 0;

    const Parameters parameters() const {
      return parameters_;
    }
    void setparameters(const Parameters& parameters) {
      parameters_ = parameters;
    }
    std::string parameter(const std::string& key) {
      return parameters_[key];
    }
    void setparameter(const std::string& key, const std::string& value) {
      parameters_[key] = value;
    }
    bool parameter_equals(const std::string& key, const std::string& value) {
      auto item = parameters_.find(key);
      if (item == parameters_.end()) {
        return false;
      }
      else {
        return item->second == value;
      }
    }
    std::string tostring() const {
      return tostring_part("", "", "");
    };
    const std::string compare(std::shared_ptr<Type> supertype);

  protected:
    bool equal_parameters(const Parameters& other) const;
    bool get_typestr(std::string& output) const;
    const std::string string_parameters() const;

    Parameters parameters_;
  };
}

#endif // AWKWARD_TYPE_H_
