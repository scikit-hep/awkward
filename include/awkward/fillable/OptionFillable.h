// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_OPTIONFILLABLE_H_
#define AWKWARD_OPTIONFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class OptionFillable: public Fillable {
  public:
    OptionFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& offsets, Fillable* content): options_(options), offsets_(offsets), content_(content) { }

    static OptionFillable* fromnulls(const FillableOptions& options, int64_t nullcount, Fillable* content) {
      GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::full(options, -1, nullcount);
      return new OptionFillable(options, offsets, content);
    }

    static OptionFillable* fromvalids(const FillableOptions& options, Fillable* content) {
      GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::arange(options, content->length());
      return new OptionFillable(options, offsets, content);
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
    GrowableBuffer<int64_t> offsets_;
    std::shared_ptr<Fillable> content_;

    void maybeupdate(Fillable* tmp);
  };
}

#endif // AWKWARD_OPTIONFILLABLE_H_
