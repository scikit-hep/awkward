// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include <string>
#include <vector>

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

    virtual bool active() const = 0;
    virtual Fillable* null() = 0;
    virtual Fillable* boolean(bool x) = 0;
    virtual Fillable* integer(int64_t x) = 0;
    virtual Fillable* real(double x) = 0;
    virtual Fillable* beginlist() = 0;
    virtual Fillable* endlist() = 0;
    virtual Fillable* begintuple(int64_t numfields) = 0;
    virtual Fillable* index(int64_t index) = 0;
    virtual Fillable* endtuple() = 0;
    virtual Fillable* beginrecord(int64_t disambiguator) = 0;
    virtual Fillable* field_fast(const char* key) = 0;
    virtual Fillable* field_check(const char* key) = 0;
    virtual Fillable* endrecord() = 0;

  };
}

#endif // AWKWARD_FILLABLE_H_
