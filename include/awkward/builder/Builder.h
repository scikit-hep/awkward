// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include <string>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"

namespace awkward {
  class Builder;
  using BuilderPtr = std::shared_ptr<Builder>;

  class EXPORT_SYMBOL Builder {
  public:
    virtual ~Builder();

    virtual const std::string
      classname() const = 0;

    virtual int64_t
      length() const = 0;

    virtual void
      clear() = 0;

    virtual const ContentPtr
      snapshot() const = 0;

    virtual bool
      active() const = 0;

    virtual const BuilderPtr
      null() = 0;

    virtual const BuilderPtr
      boolean(bool x) = 0;

    virtual const BuilderPtr
      integer(int64_t x) = 0;

    virtual const BuilderPtr
      real(double x) = 0;

    virtual const BuilderPtr
      string(const char* x, int64_t length, const char* encoding) = 0;

    virtual const BuilderPtr
      beginlist() = 0;

    virtual const BuilderPtr
      endlist() = 0;

    virtual const BuilderPtr
      begintuple(int64_t numfields) = 0;

    virtual const BuilderPtr
      index(int64_t index) = 0;

    virtual const BuilderPtr
      endtuple() = 0;

    virtual const BuilderPtr
      beginrecord(const char* name, bool check) = 0;

    virtual const BuilderPtr
      field(const char* key, bool check) = 0;

    virtual const BuilderPtr
      endrecord() = 0;

    virtual const BuilderPtr
      append(const ContentPtr& array, int64_t at) = 0;

    void
      setthat(const BuilderPtr& that);

  protected:
    BuilderPtr that_;
  };
}

#endif // AWKWARD_FILLABLE_H_
