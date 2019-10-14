// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_MOSTLYVALIDFILLABLE_H_
#define AWKWARD_MOSTLYVALIDFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class MostlyValidFillable: public Fillable {
  public:
    MostlyValidFillable(Fillable* content): nullindex_(), content_(content), length_(content->length()) { }

    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> tolayout();

    virtual Fillable* null();
    virtual Fillable* boolean(bool x);

  private:
    std::vector<int64_t> nullindex_;
    std::shared_ptr<Fillable> content_;
    int64_t length_;
  };
}

#endif // AWKWARD_MOSTLYVALIDFILLABLE_H_
