// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDFILLABLE_H_
#define AWKWARD_RECORDFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  class FillableArray;

  class RecordFillable: public Fillable {
  public:
    RecordFillable(FillableArray* fillablearray, const FillableOptions& options): fillablearray_(fillablearray), options_(options) { }

    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual Fillable* null();
    virtual Fillable* boolean(bool x);
    virtual Fillable* integer(int64_t x);
    virtual Fillable* real(double x);
    virtual Fillable* beginlist();
    virtual Fillable* endlist();
    virtual Fillable* beginrec(int64_t slotsid);
    virtual Fillable* indexrec(int64_t index);
    virtual Fillable* endrec();

  private:
    FillableArray* fillablearray_;
    const FillableOptions options_;

  };
}

#endif // AWKWARD_RECORDFILLABLE_H_
