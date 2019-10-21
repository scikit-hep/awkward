// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLEARRAY_H_
#define AWKWARD_FILLABLEARRAY_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  class FillableArray {
  public:
    FillableArray(const FillableOptions& options): fillable_(new UnknownFillable(options)) { }

    const std::string tostring() const;
    int64_t length() const;
    void clear();
    const std::shared_ptr<Type> type() const;
    const std::shared_ptr<Content> snapshot() const;
    const std::shared_ptr<Content> getitem_at(int64_t at) const;
    const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const;
    const std::shared_ptr<Content> getitem(const Slice& where) const;

    void null();
    void boolean(bool x);
    void integer(int64_t x);
    void real(double x);

  private:
    std::shared_ptr<Fillable> fillable_;

    void maybeupdate(Fillable* tmp);
  };
}

#endif // AWKWARD_FILLABLE_H_
