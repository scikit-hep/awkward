// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNKNOWNFILLABLE_H_
#define AWKWARD_UNKNOWNFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class UnknownFillable: public Fillable {
  public:
    UnknownFillable(): nullcount_(0) { }

    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot();

    virtual Fillable* null();
    virtual Fillable* boolean(bool x);

  private:
    int64_t nullcount_;
  };
}

#endif // AWKWARD_UNKNOWNFILLABLE_H_
