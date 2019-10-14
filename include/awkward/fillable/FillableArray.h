// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLEARRAY_H_
#define AWKWARD_FILLABLEARRAY_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/BoolFillable.h"

namespace awkward {
  class FillableArray {
  public:
    FillableArray(): fillable_(new BoolFillable()) { }

    std::string tostring() const;
    int64_t length() const;
    void clear();
    const std::shared_ptr<Type> type() const;
    const std::shared_ptr<Content> snapshot();

    void null();
    void boolean(bool x);

  private:
    std::shared_ptr<Fillable> fillable_;

    void maybeupdate(Fillable* tmp);
  };
}

#endif // AWKWARD_FILLABLE_H_
