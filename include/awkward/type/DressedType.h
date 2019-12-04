// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_DRESSEDTYPE_H_
#define AWKWARD_DRESSEDTYPE_H_

#include <sstream>

#include "awkward/type/Type.h"

namespace awkward {
  template <typename T>
  class DressParameters {
  public:
    virtual const std::vector<std::string> keys() const = 0;
    virtual const T get(const std::string& key) const = 0;
    virtual const std::string get_string(const std::string& key) const = 0;
    virtual bool equal(const DressParameters<T>& other) const = 0;
  };

  template <typename T>
  class Dress {
  public:
    virtual const std::string name() const = 0;
    virtual const std::string typestr(const DressParameters<T>& parameters) const = 0;
    virtual bool equal(const Dress& other) const = 0;
  };

  template <typename D, typename P>
  class DressedType: public Type {
  public:
    DressedType(const std::shared_ptr<Type> type, const D& dress, const P& parameters): type_(type), dress_(dress), parameters_(parameters) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const {
      std::string outstr = dress_.typestr(parameters_);
      if (outstr.size() != 0) {
        return outstr;
      }
      else {
        std::stringstream out;
        out << indent << pre << "dress[" << util::quote(dress_.name(), false) << ", " << type_.get()->tostring_part(indent, "", "");
        for (auto key : parameters_.keys()) {
          out << ", " << key << "=" << parameters_.get_string(key);
        }
        out << "]";
        return out.str();
      }
    }
    virtual const std::shared_ptr<Type> shallow_copy() const {
      return std::shared_ptr<Type>(new DressedType(type_, dress_, parameters_));
    }
    virtual bool equal(std::shared_ptr<Type> other) const {
      if (DressedType<D, P>* raw = dynamic_cast<DressedType<D, P>*>(other.get())) {
        D otherdress = raw->dress();
        if (!dress_.equal(otherdress)) {
          return false;
        }
        P otherparam = raw->parameters();
        if (!parameters_.equal(otherparam)) {
          return false;
        }
        return type_.get()->equal(raw->type());
      }
      else {
        return false;
      }
    }

    const std::shared_ptr<Type> type() const { return type_; };
    const D dress() const { return dress_; };
    const P parameters() const { return parameters_; }

  private:
    const std::shared_ptr<Type> type_;
    const D dress_;
    const P parameters_;
  };
}

#endif // AWKWARD_DRESSEDTYPE_H_
