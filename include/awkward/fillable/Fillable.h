// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"

namespace awkward {
  class Fillable {
  public:
    virtual ~Fillable() { }

    virtual int64_t length() const = 0;
    virtual void clear() = 0;
    virtual const std::shared_ptr<Type> type() const = 0;
    virtual const std::shared_ptr<Content> snapshot() const = 0;

    virtual Fillable* null() = 0;
    virtual Fillable* boolean(bool x) = 0;
    virtual Fillable* integer(int64_t x) = 0;
    virtual Fillable* real(double x) = 0;
    virtual Fillable* beginlist() = 0;
    virtual Fillable* endlist() = 0;
    virtual Fillable* beginrec(const std::vector<std::string>& keys) = 0;
    virtual Fillable* reckey(int64_t index) = 0;
    virtual Fillable* endrec() = 0;
  };
}

#endif // AWKWARD_FILLABLE_H_
