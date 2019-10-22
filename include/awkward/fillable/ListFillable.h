// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTFILLABLE_H_
#define AWKWARD_LISTFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  class ListFillable: public Fillable {
  public:
    ListFillable(const FillableOptions& options): options_(options), offsets_(options), content_(new UnknownFillable(options)), begun_(false) {
      offsets_.append(0);
    }
    ListFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& offsets, Fillable* content, bool begun): options_(options), offsets_(offsets), content_(std::shared_ptr<Fillable>(content)), begun_(begun) { }

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

  private:
    const FillableOptions options_;
    GrowableBuffer<int64_t> offsets_;
    std::shared_ptr<Fillable> content_;
    bool begun_;

    Fillable* maybeupdate(Fillable* tmp);
  };
}

#endif // AWKWARD_LISTFILLABLE_H_
