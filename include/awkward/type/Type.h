// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_TYPE_H_
#define AWKWARD_TYPE_H_

#include <memory>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"

namespace awkward {
  class Content;
  typedef std::shared_ptr<Content> ContentPtr;
  class Type;
  typedef std::shared_ptr<Type> TypePtr;

  class EXPORT_SYMBOL Type {
  public:
    static TypePtr none();

    Type(const util::Parameters& parameters, const std::string& typestr);
    virtual ~Type();

    virtual std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const = 0;
    virtual const TypePtr shallow_copy() const = 0;
    virtual bool equal(const TypePtr& other, bool check_parameters) const = 0;
    virtual int64_t numfields() const = 0;
    virtual int64_t fieldindex(const std::string& key) const = 0;
    virtual const std::string key(int64_t fieldindex) const = 0;
    virtual bool haskey(const std::string& key) const = 0;
    virtual const std::vector<std::string> keys() const = 0;
    virtual const ContentPtr empty() const = 0;

    const util::Parameters parameters() const;
    void setparameters(const util::Parameters& parameters);
    const std::string parameter(const std::string& key) const;
    void setparameter(const std::string& key, const std::string& value);
    bool parameter_equals(const std::string& key, const std::string& value) const;
    bool parameters_equal(const util::Parameters& other) const;
    const std::string tostring() const;
    const std::string compare(TypePtr supertype);

    const std::string typestr() const;

  protected:
    bool get_typestr(std::string& output) const;
    const std::string string_parameters() const;

    util::Parameters parameters_;
    const std::string typestr_;
  };
}

#endif // AWKWARD_TYPE_H_
