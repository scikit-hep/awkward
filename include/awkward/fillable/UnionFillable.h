// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONFILLABLE_H_
#define AWKWARD_UNIONFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class TupleFillable;
  class RecordFillable;

  class UnionFillable: public Fillable {
  public:
    UnionFillable(const FillableOptions& options, const GrowableBuffer<int8_t>& types, const GrowableBuffer<int64_t>& offsets, std::vector<std::shared_ptr<Fillable>> contents): options_(options), types_(types), offsets_(offsets), contents_(contents), current_(-1) { }

    static UnionFillable* fromsingle(const FillableOptions& options, Fillable* firstcontent) {
      GrowableBuffer<int8_t> types = GrowableBuffer<int8_t>::full(options, 0, firstcontent->length());
      GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::arange(options, firstcontent->length());
      std::vector<std::shared_ptr<Fillable>> contents({ std::shared_ptr<Fillable>(firstcontent) });
      return new UnionFillable(options, types, offsets, contents);
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
    GrowableBuffer<int8_t> types_;
    GrowableBuffer<int64_t> offsets_;
    std::vector<std::shared_ptr<Fillable>> contents_;
    int64_t current_;
  };
}

#endif // AWKWARD_UNIONFILLABLE_H_
