// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_BOOLFILLABLE_H_
#define AWKWARD_BOOLFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class BoolFillable: public Fillable {
  public:
    BoolFillable(): data_() { }

    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> toarray() const;

    virtual Fillable* boolean(bool x);

  private:
    std::vector<bool> data_;
  };
}

#endif // AWKWARD_BOOLFILLABLE_H_
