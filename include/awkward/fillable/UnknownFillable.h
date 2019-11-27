// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNKNOWNFILLABLE_H_
#define AWKWARD_UNKNOWNFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class UnknownFillable: public Fillable {
  public:
    UnknownFillable(const FillableOptions& options, int64_t nullcount): options_(options), nullcount_(nullcount) { }

    static UnknownFillable* fromempty(const FillableOptions& options) {
      return new UnknownFillable(options, 0);
    }

    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual bool active() const;
    virtual Fillable* null();
    virtual Fillable* boolean(bool x);
    virtual Fillable* integer(int64_t x);
    virtual Fillable* real(double x);
    virtual Fillable* beginlist();
    virtual Fillable* endlist();
    virtual Fillable* begintuple(int64_t numfields);
    virtual Fillable* index(int64_t index);
    virtual Fillable* endtuple();
    virtual Fillable* beginrecord(int64_t disambiguator);
    virtual Fillable* field_fast(const char* key);
    virtual Fillable* field_check(const char* key);
    virtual Fillable* endrecord();

  private:
    const FillableOptions options_;
    int64_t nullcount_;

    template <typename T>
    Fillable* prepare() const;
  };
}

#endif // AWKWARD_UNKNOWNFILLABLE_H_
