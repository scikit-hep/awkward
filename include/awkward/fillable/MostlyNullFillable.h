// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_MOSTLYNULLFILLABLE_H_
#define AWKWARD_MOSTLYNULLFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class MostlyNullFillable: public Fillable {
  public:
    MostlyNullFillable(const FillableOptions& options, Fillable* content, int64_t nullcount): options_(options), validindex_(), content_(content), length_(nullcount) { }

    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual Fillable* null();
    virtual Fillable* boolean(bool x);

  private:
    const FillableOptions options_;
    std::vector<int64_t> validindex_;
    std::shared_ptr<Fillable> content_;
    int64_t length_;
  };
}

#endif // AWKWARD_MOSTLYNULLFILLABLE_H_
